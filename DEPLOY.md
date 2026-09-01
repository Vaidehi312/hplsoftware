# Deployment steps

Ordered. Each one is a thing that fails on its own if skipped, and most fail quietly.

`SOCK=/nfs/home/users/vpandya/databases/postgres_hpl_kb/socket` throughout.

---

## 0. Get the code onto the cluster

```bash
git pull            # or copy the folder across
```

`backend/` runs **on the HPC login node** — it shells out to `sbatch`, opens `.svs` files from
scratch, and reaches Postgres over a local socket. `app/` runs on your laptop. They are separate
processes and neither picks up the other's changes.

---

## 1. Back up production

```bash
pg_dump -h $SOCK -d hpl_kb -Fc -f hpl_kb_$(date +%F).dump
```

Nothing below deletes anything, and the migration has been rehearsed against a populated copy of
this exact schema. The backup is what makes that reversible rather than merely unlikely.

---

## 2. Migrate **production** — required even if you only use the Test KB

```bash
psql -h $SOCK -d hpl_kb -f backend/migrate_all.sql
```

**This is the step whose absence is confusing rather than loud.** Run tracking lives in production
for every target — a run is one run whichever Knowledge Bank it filled, and splitting
`slurm_dataset_runs` across two databases would leave the pipeline UI unable to list its own
history. So registering into *test* still writes `registration_kb_target` into *production*.

Without this migration that write fails **after** the rows are already committed to test: the
cohort is registered, and the run says it never was. The registration preview now checks for those
columns and refuses to enable Commit, so you will see it before it happens rather than after — but
running this first is the actual fix.

What it does to production, rehearsed: row counts unchanged, constraints unchanged, columns only
*added*. No `DROP TABLE`, no `TRUNCATE`, no `DELETE` anywhere in the chain, and the normalising
`UPDATE`s in `migrate_indexes.sql` report `UPDATE 0` on an already-normalised database.

---

## 3. Create and migrate the Test KB

```bash
createdb -h $SOCK hpl_kb_test                              # skip if it exists
psql -h $SOCK -d hpl_kb_test -f backend/migrate_all.sql
```

Run `migrate_all.sql` even if you built the database with the schema already — it is idempotent, so
it no-ops where the schema is right and fills in anything missing.

Then check what is actually there:

```bash
psql -h $SOCK -d hpl_kb_test -f backend/check_kb_ready.sql
```

Read-only. It answers the question "I already have the schema" leaves open, which is not whether
the tables exist but whether the **cluster reference tables have data**. A schema-only copy of
production passes every table check and still answers every clinical question with blanks, because
`hpc_dictionary` is empty.

### Copy the reference data — and only that

`migrate_all.sql` does not create those four tables (their `\d` has never been captured, §7), and
the tile overlay, the HPC panels and the chatbot all read them. They are the **only** thing a test
KB needs that registration does not create for itself:

```bash
pg_dump -h $SOCK -d hpl_kb --data-only \
  -t hpc_dictionary -t hpc_malignant_details \
  -t hpc_non_malignant_details -t hpc_survival_analysis \
  -f hpc_ref_data.sql

psql -h $SOCK -d hpl_kb_test -v ON_ERROR_STOP=1 -f hpc_ref_data.sql
```

`--data-only` because the tables are already there; sequences come across with it. **Not
idempotent** — a second run fails on duplicate keys. To redo it, `TRUNCATE` those four first.

Re-run `check_kb_ready.sql` afterwards: the four row counts should read 71 / 27 / 44 / 71.

### Do not clone the whole production database into test

The test KB is useful because what you register is exactly what you see. A full copy puts TCGA's
~500,000 tiles in there too, so the slide list, the HPC panels and the chatbot all answer from a
cohort you did not put there — and it makes "production untouched" harder to verify, not easier.
It is also 4.4 GB of `h_latent_vectors` that nothing reads.

If you ever do want a full clone — to exercise the cross-cohort collision guards against real
neighbours — that is a different exercise from §6:

```bash
pg_dump -h $SOCK -d hpl_kb -Fc --exclude-table-data=h_latent_vectors -f full.dump
pg_restore -h $SOCK -d hpl_kb_test --clean --if-exists full.dump
```

`DB_NAME_TEST` (server) and `HPL_DB_NAME_TEST` (Streamlit) both default to `hpl_kb_test`. Set them
only if you called it something else.

---

## 4. Restart the API server on the cluster

```bash
cd backend && python tile_server_v2_.py      # FastAPI on :8000
```

Check it came up on the right databases:

```bash
curl 'localhost:8000/health'
curl 'localhost:8000/health?kb_target=test'
```

Both should return `slides_loaded` and name the database they read. The second is the one that
proves `hpl_kb_test` exists and is reachable — if step 3 was missed it fails **here**, before any
data moves.

---

## 5. Start the UI on your laptop

```bash
ssh -N -L 8000:localhost:8000 -L 5432:localhost:5432 beatson-hpc
streamlit run app/app_v28.py
```

The second `-L 5432` is not optional: the chatbot and three viewer helpers query Postgres directly
rather than through the API.

---

## 6. Run a Radiogenomics subset against the Test KB

1. **Select `Test — hpl_kb_test`** at the top of the sidebar. The sidebar shows a warning while it
   is selected. Everything downstream follows it — registration, the assignment load, the viewer,
   HPC panels, heatmap overlay, slide list and chatbot.
2. Open **5. Register in the Knowledge Bank**. Set the cohort to `RADIOGENOMICS`, choose
   **subset**, and name two or three slides.
3. **Preview.** Check `slides_without_files` and `ambiguous_slides` are both **0**, and that the
   slide count matches what you asked for. Commit is disabled if anything would be refused.
4. **Register.** Then open **6. Knowledge Bank load**, give it the assignment CSV path, preview,
   and check the match rate is **≥95%**. If it is still low *after* registering, that is a genuine
   slide-naming problem, not a missing step.
5. **Load.** Then open one of those slides in the viewer and ask the chatbot about it.

Every path except the assignment CSV comes off the pipeline run — `raw_dir`, `tile_dir`,
`h5_output_path`, `dataset_name` — and the viewer resolves slide files from that KB's own
`wsi_registry`. Nothing is hardcoded to Radiogenomics.

---

## 7. Confirm production is untouched

```sql
-- against hpl_kb
SELECT dataset_id, count(*) FROM tile_registry       GROUP BY 1 ORDER BY 1;
SELECT dataset_id, count(*) FROM wsi_registry        GROUP BY 1 ORDER BY 1;
SELECT dataset_id, count(*) FROM hpl_profile_summary GROUP BY 1 ORDER BY 1;
```

Every pre-existing `dataset_id` must have exactly the count it had before, and `RADIOGENOMICS`
must not appear at all. The same three queries against `hpl_kb_test` should show it and nothing
else.

---

## 8. Only when that all holds — the real thing

Switch the sidebar back to **Production**, and repeat step 6 against the full cohort.

---

## Expected differences in the Test KB

- **No heatmap overlay.** `tile_hpc_heatmap` is empty there, because nothing writes it anywhere.
  Tile metadata is unaffected.
- **Only Streamlit has the selector.** The React frontend is production-only for now.

## Still outstanding

Capture these and append them to `backend/kb_live_schema_2026-08-26.txt` — they are the reason
step 3 needs a `pg_dump` of the reference tables at all:

```
\d hpc_dictionary
\d hpc_malignant_details
\d hpc_non_malignant_details
\d hpc_survival_analysis
\d+ slide_metadata
```
