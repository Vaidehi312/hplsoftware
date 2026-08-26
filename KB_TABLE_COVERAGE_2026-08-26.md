# Which Knowledge Bank tables a new dataset actually fills

**Date:** 2026-08-26
**Trigger:** the live schema (`\d` against `hpl_kb`, PostgreSQL 18.0) was captured for the first
time, which unblocked the item `CLASSIFIER_TUNING_2026-08-13.md` §8 had been stuck on since
2026-08-13: *"Real column-level schema for `tile_coordinates`, `h_latent_vectors` — only table
names confirmed (`\dt`), not column definitions (`\d`) — blocks writing any real `INSERT`."*
**Method:** every one of the 17 tables and the one view audited independently against the live
code, each gap claim then attacked from three angles (find a writer the audit missed; break the
reachability claim; show the gap does not matter). Findings that survived are below; two did not
and are recorded as overturned.

---

## TL;DR

A cohort that goes through all five pipeline stages ends up with **nothing in the Knowledge Bank**.
Not partially loaded — empty, and Stage 5 refuses at a 0% match rate, correctly, with a message
about slide naming.

The cause was never a bug. `load_hpc_assignments.load()` only ever `UPDATE`s
`tile_registry.hpc_id`; the row it updates has to already exist, and nothing created it. Every row
in `tile_coordinates`, `tile_registry`, `wsi_registry`, `wsi_metadata` and `dataset_config` on the
live database was put there by a hand-run notebook.

`register_dataset.py` (committed 2026-08-19, `4793278`, and never recorded in the session log)
already covered two of those five tables — but only from the command line. It has no endpoint, no
client method and no UI step, so "automated" was never true of it either.

**What changed in this session:** registration now covers all five tables in one transaction,
runs from the UI as pipeline step 5, and the KB load is step 6 and gated on it. The eight tables
that had no `CREATE TABLE` anywhere in git now have one. 46 new tests, 384 passing.

**What is still open, and is a modelling problem rather than plumbing:** `tile_hpc_heatmap` — 149 MB,
71 probability columns, read into memory at server startup, and nothing has ever written it.

---

## 1. The table of record

All 17 tables plus the view. "New dataset" means a cohort taken through tiling → packaging →
extraction → classification → registration → KB load.

| # | Table | What fills it | What reads it | A new dataset gets |
|---|---|---|---|---|
| 1 | `tile_coordinates` | **Stage 5** `register_dataset.py` | `/slide/{id}/tiles_meta` and `/adjacency` — the driving table of both | ✅ filled |
| 2 | `tile_registry` | **Stage 5** (identity) + Stage 6 (`hpc_id`, confidence) | tile overlay, adjacency panel, `cohort_shift.py` | ✅ filled |
| 3 | `wsi_registry` | **Stage 5** — previously only the single-slide upload path | `_load_wsi_map()` at startup → every viewer endpoint | ✅ filled *(was the worst gap)* |
| 4 | `wsi_metadata` | **Stage 5**, opt-in (`--slide-metadata`) | nothing live | ✅ filled when asked for |
| 5 | `dataset_config` | **Stage 5**, from the run's own `tiling_params` | nothing live | ✅ filled |
| 6 | `hpl_profile_summary` | Stage 6 `load_hpc_assignments.py` | chatbot, HPC explorer | ✅ filled |
| 7 | `hpl_profile_proportion` | Stage 6 | chatbot, HPC explorer | ✅ filled |
| 8 | `slurm_dataset_runs` | the server, every stage | the whole pipeline UI | ✅ filled |
| 9 | `slurm_dataset_run_jobs` | the server, on each submission | "what has this run tried" panel | ✅ filled *(Stages 5 and 6 never appear — they submit no Slurm job)* |
| 10 | `hpc_dictionary` | nothing — static reference, 71 clusters | tile overlay, chatbot, `/hpc/{id}/info` | ⚪ correct by standing still |
| 11 | `hpc_malignant_details` | nothing — static, 27 rows | `/hpc/{id}/info`, chatbot | ⚪ correct by standing still |
| 12 | `hpc_non_malignant_details` | nothing — static, 44 rows | `/hpc/{id}/info`, chatbot | ⚪ correct by standing still |
| 13 | `hpc_survival_analysis` | nothing — static Cox coefficients | survival overlay, HPC panels | ⚪ correct by standing still |
| 14 | `tile_hpc_heatmap` | **nothing, ever** | `_load_heatmap_probs()` at startup → merged into every `tiles_meta` response | ❌ **empty for the cohort; globally stale** |
| 15 | `h_latent_vectors` | nothing — 4.4 GB of 2025 hand-loaded rows | nothing live | ❌ stale, and nothing notices |
| 16 | `slide_hpc_membership` | nothing — 19,493 rows from ~501 old slides | nothing live | ❌ stale, and nothing notices |
| 17 | `tile_hpc_heatmap_old` | nothing — 141 MB legacy twin | nothing | ❌ dead storage |
| — | `slide_metadata` (view) | n/a — a view has no storage | nothing | ⚪ n/a |

---

## 2. What already worked, and was not written down

`CLASSIFIER_TUNING_2026-08-13.md` §8 still reads *"nothing in this repository ever `INSERT`s into
`tile_coordinates`, `tile_registry`, or the embeddings table"*. Two-thirds of that stopped being
true six days after it was written.

`backend/register_dataset.py` — commit `4793278`, "Let a new dataset reach the Knowledge Bank at
all" — reads tile identity out of the packaged `.h5` and coordinates out of Stage 1's per-slide
`_tile_metadata.csv`, and writes `tile_coordinates` and `tile_registry` in one transaction, scoped
by `dataset_id`, dry-run by default, refusing on a cross-cohort `slide_tile` collision. It got
`image_index` right for the reason that matters: it reads the row position back out of the `.h5`
rather than deriving it from the metadata CSV, because packaging order is not guaranteed stable
across a re-package.

The log never records it, so the gap looked open when it was half closed. Two lessons, both
already visible elsewhere in this project: a session log that is not updated when the fix lands is
worse than no log, and the thing that made §8's plan look blocked (*"blocks writing any real
`INSERT`"*) had in fact been worked around by writing the `INSERT` against the columns the code
already knew about.

---

## 3. The real gaps

Ordered by whether something live reads the table, then by how hard it is to fill.

### 3.1 `wsi_registry` — the one that would have bitten first · **fixed**

Before this session the only `INSERT INTO wsi_registry` in the codebase was in
`_register_uploaded_slide()` (`tile_server_v2_.py:323`), reached exclusively from
`POST /upload-slide` — the interactive drag-and-drop path, which hard-codes
`dataset_id = 'UPLOADED'`. **No bulk Slurm dataset run has ever registered a slide.**

This is the codebase's signature failure mode one layer up. All stages succeed. `tile_registry`
and `tile_coordinates` fill. `hpl_profile_*` refresh. Every count looks right. And every slide in
the new cohort 404s in the viewer, because `_open_slide()` resolves paths from `_wsi_map`, which
`_load_wsi_map()` builds from `wsi_registry` alone.

Fixed by folding slide registration into `register_dataset.py`, in the **same transaction** as the
tile tables — deliberately, because a separate command is a command someone forgets, and the state
it leaves behind is exactly the invisible-cohort state above.

### 3.2 `tile_coordinates` / `tile_registry` were CLI-only · **fixed**

`register_dataset.py` had no endpoint, no `api_client` method and no UI step. Registering a cohort
meant knowing the script existed, SSH-ing to the login node, and passing four paths by hand — three
of which the server already had on the run record.

Fixed: `POST /dataset-jobs/{id}/register-preview` and `/register` derive every path from
`slurm_dataset_runs` and call `register_dataset.py`'s own functions rather than reimplementing
them, so every guard the CLI enforces applies unchanged. The UI is pipeline step 5, preview-then-commit,
and step 6 is gated on it.

A related correction: **`CLAUDE.md` said Stage 1 produced `tile_coordinates`.** It does not — it
writes a CSV to disk. That single wrong line is the best explanation for why this gap survived so
long: everyone believed the coordinates table filled itself.

### 3.3 `wsi_metadata` and `dataset_config` were unreachable · **fixed, and worth less than it looks**

Neither has a live reader today, so filling them fixes nothing that is currently broken. They are
worth filling anyway for one specific reason: `tile_server_v2_.py:216-218` computes

```python
TILE_SIZE_5X   = 224
SCALE          = 1.8 / 0.252
TILE_SIZE_NATIVE = int(TILE_SIZE_5X * SCALE)   # 1600
```

— that is, **every slide in every cohort is assumed to have been scanned at 0.252 mpp and tiled at
1.8 mpp.** `dataset_config` (`target_mpp`, `tile_size_5x_px`) and `wsi_metadata` (`mpp_x`,
`objective_power`) are precisely the per-cohort numbers that assumption stands in for. Stage 5 now
records them, from the run's own `tiling_params` and from OpenSlide respectively.

**Using them is a separate change and is not made here.** Rewriting the viewer's coordinate maths
on a cohort whose real geometry has never been checked would risk mis-placing every tile on every
existing slide — a strictly worse outcome than a documented assumption.

### 3.4 `tile_hpc_heatmap` — **still open, and not a plumbing problem**

`_load_heatmap_probs()` (`tile_server_v2_.py:1332`) runs `SELECT * FROM tile_hpc_heatmap` at server
startup, keeps `slide_tile` plus every `p_hpc_*` column in a process-global, and merges it into
every `/slide/{id}/tiles_meta` response. All 149 MB, no `dataset_id` filter — the table has no such
column — and no reload endpoint, so the in-memory copy is frozen until the process restarts.

Nothing in this repository has ever written it. It was built by an off-repo notebook path
(`heatmap_table.ipynb`) from a Keras model, and its 141 MB twin `tile_hpc_heatmap_old` came from
the same path at an earlier date.

The reason this cannot simply be automated: **the k-NN classifier does not produce a 71-class
probability distribution.** `vote()` in `assign_hpc_clusters.py` produces a top-1 label and a
`vote_margin`. Filling `p_hpc_0 .. p_hpc_70` needs a decision about what those numbers mean —
normalised k-NN vote fractions at k=10 are not the same quantity the existing rows hold, and
writing them into the same columns would leave two incompatible definitions in one table with
nothing to distinguish them.

Three honest options, none taken here:

1. **Emit vote fractions** from `assign_hpc_clusters.py`, add a provenance column, and treat the
   existing rows as a different (older) quantity. Cheapest, and changes what the overlay means.
2. **Retrain / re-run the classifier that produced the current values**, if its weights and
   annotation CSV can still be found. `Heatmap/hierarchical_classifier_best.keras` is a candidate
   with no recorded provenance.
3. **Drop the probability overlay** and serve the top-1 label plus `vote_margin`, which the
   pipeline does produce and which is calibrated (§ below).

Until one is chosen, a new cohort's tiles simply have no heatmap probabilities, which is what
happens today and degrades to nulls rather than to wrong numbers.

---

## 4. What is not worth fixing

Recommending work on a table nothing reads is worse than saying nothing, so, plainly:

- **`h_latent_vectors`** — 4.4 GB, `vector(1024)` although the encoder emits 128-D. No writer and
  **no live reader**: the only references are in `app/app_v3` … `app_v21`, all superseded by
  `app_v28`. Stage 3 does produce the input (`<set>_h_latent` is in every projections `.h5`), so
  filling it is possible; there is no reason to.
- **`slide_hpc_membership`** — no `dataset_id` column at all, so it cannot be scoped per cohort
  even in principle. Structurally a denormalised index of `tile_registry`
  (`SELECT DISTINCT slides, hpc_id FROM tile_registry`) and referenced by nothing.
- **`tile_hpc_heatmap_old`** — 141 MB, no reader. Note its primary key is named
  `tile_hpc_heatmap_pkey` while the *live* table's is named `hpc_heatmap_pkey`; the names are
  transposed relative to the tables, which is what renaming a live table leaves behind. Anyone
  reading `\di` will draw the wrong conclusion about which is current.
- **`slide_metadata`** — a view, 0 bytes, no reader. Its definition is not recoverable from this
  repository: `hpl_kb_dump.pgsql` predates the `wsi_*` tables and contains no view at all. The 20
  textual occurrences of the name in the codebase are all a Python identifier, never SQL. Recover
  it with `SELECT pg_get_viewdef('public.slide_metadata'::regclass, true);` if it is ever wanted.

---

## 5. Two claims the verification overturned

Recorded because the audit was wrong about them and a report that hides its own corrections is
less useful:

- **`dataset_config` is not a "gap".** It has no live reader, so an empty one breaks nothing today.
  The real finding is the hard-coded geometry in §3.3 that stands in for it — a regression from
  having the table, not a consequence of it being empty.
- **`hpc_survival_analysis` does have a writer**, in `integration.py:107` (a `TRUNCATE` followed by
  a `COPY`). It is unreachable in practice: nothing imports it, no stage calls it, and it raises
  `NameError` on an undefined `z_latent` at module scope before `run()` is even defined. Worth
  knowing separately that the same `TRUNCATE` also names `tile_registry` — if that file is ever
  repaired and run, it would empty the table this pipeline exists to fill.

---

## 6. The schema itself was not in git

Found while transcribing the `\d` output, and independent of the coverage question.

Across the **entire repository**, only 9 of the 17 live tables had a `CREATE TABLE`:

| in git before today | missing entirely |
|---|---|
| `hpc_dictionary`, `hpc_malignant_details`, `hpc_non_malignant_details`, `hpc_survival_analysis`, `hpl_profile_proportion`, `hpl_profile_summary`, `tile_registry` (all in `schema.sql`); `slurm_dataset_runs`, `slurm_dataset_run_jobs` (migrations) | `dataset_config`, `h_latent_vectors`, `slide_hpc_membership`, `tile_coordinates`, `tile_hpc_heatmap`, `tile_hpc_heatmap_old`, `wsi_metadata`, `wsi_registry`, and the `slide_metadata` view |

Also missing: **`dataset_id`** — the column every cohort guard in `register_dataset.py` and
`load_hpc_assignments.py` is scoped by, `NOT NULL` on five tables — appears in no `CREATE TABLE`
and no `ALTER TABLE` anywhere in git. It was added to the live database by hand.

Two consequences.

**`migrate_all.sql`'s central claim was false.** Its header says *"running it against a fresh one
builds the whole schema"*. It stopped at `migrate_indexes.sql:8`, `UPDATE tile_coordinates ...`,
under `ON_ERROR_STOP` — so the error named an index migration rather than a missing table.

**`schema.sql` is worse than absent, because it looks authoritative.** It is a `pg_dump` from
2025-10-23, never regenerated, and it disagrees with the live database on the one table this
pipeline writes most:

| `schema.sql` `tile_registry` | live `tile_registry` |
|---|---|
| `id integer NOT NULL` (primary key) | no `id` column |
| `hpc_id character varying(100)` | `hpc_id integer` → FK `hpc_dictionary(hpc_id)` |
| `samples varchar(100)` | `samples varchar` |
| — | `slide_tile varchar NOT NULL` (primary key) |
| — | `dataset_id text NOT NULL` |
| — | `hpc_vote_margin`, `hpc_neighbor_distance`, `hpc_assigned_at`, `hpc_reference` |

Since the live `tile_registry.hpc_id` is an integer with a foreign key into
`hpc_dictionary(hpc_id)`, `hpc_dictionary.hpc_id` cannot still be the `varchar(100)` `schema.sql`
declares either — so the four cluster reference tables have drifted too, by an amount nobody has
measured.

Fixed by `backend/migrate_kb_base_tables.sql`, transcribed from the capture now kept at
`backend/kb_live_schema_2026-08-26.txt`, run first in `migrate_all.sql`, with
`backend/tests/test_schema_coverage.py` failing if a live table ever again has no DDL.
`schema.sql` now carries a header saying it is stale and what to use instead.

**Still not covered, and deliberately not guessed at:** `hpc_dictionary`,
`hpc_malignant_details`, `hpc_non_malignant_details`, `hpc_survival_analysis` and the
`slide_metadata` view. Their real shape needs five commands against the live database:

```sql
\d hpc_dictionary
\d hpc_malignant_details
\d hpc_non_malignant_details
\d hpc_survival_analysis
\d+ slide_metadata
```

Until those are captured, a fresh checkout still cannot build a complete schema, and this is stated
rather than papered over.

---

## 7. Other things the audit turned up, not acted on

| finding | where | why it matters |
|---|---|---|
| `hpl_profile_summary`'s unique constraint is `(samples, slides)` with **no `dataset_id`** | live schema | Two cohorts holding the same slide id cannot both have a summary row. The FK from `hpl_profile_proportion` inherits the problem. |
| `hpl_profile_proportion.slides` is `varchar(170)`, `hpl_profile_summary.slides` is `varchar(150)` | live schema | The referencing side is wider than the referenced side. A slide id between 151 and 170 characters cannot satisfy the FK. |
| `--allow-unknown-clusters` cannot work | `load_hpc_assignments.py` + the FK | Opting out of the guard does not make the load succeed — the proportion `INSERT` then violates the `hpc_id` FK, and because `load()` is one transaction the rollback takes `tile_registry` with it. The checkbox is exposed in the UI. |
| Stage 6 has no attempt history | `slurm_dataset_run_jobs` | The table's unit is a submitted Slurm job, and Stage 6 submits none. A load run three times, or run against an overridden CSV path, leaves no trace. Stage 5 now has the same shape, deliberately consistent. |
| The chatbot defaults to "non-malignant" on a missing dictionary row | `app/hpc_chat_handlers_v23.py:120-134` | Reads `hpc_dictionary.malignant`; anything not strictly true — including `NULL`, including a cluster with no dictionary row — falls through to "Tile belongs to **non-malignant epithelium**". A clinically loaded silent wrong answer rather than a blank panel. |
| Four of five stages have no completion flag | `slurm_dataset_runs` | Only Stages 5 and 6 persist a boolean. Stages 1–4 recompute "done" on every `/status` from live Slurm state plus a real read of the output file — which is more robust, and worth knowing before adding a sixth. |

---

## 8. What changed

### Code

```
backend/register_dataset.py            slide-level registration: wsi_registry, wsi_metadata
                                       (opt-in), dataset_config; one transaction with the tile
                                       tables; _native() so NaN reaches Postgres as NULL
backend/slide_naming.py                file_uuid_from_raw_path()
backend/tile_server_v2_.py             POST /register-preview, POST /register; registration_*
                                       in /status
backend/migrate_kb_base_tables.sql     NEW — the eight tables nothing else creates, + dataset_id
backend/migrate_dataset_runs_registration.sql   NEW — registration_done/at/dataset_id/raw_dir/rows
backend/migrate_all.sql                base tables first; registration migration added
backend/kb_live_schema_2026-08-26.txt  NEW — the `\d` capture the DDL was transcribed from
schema.sql                             header: STALE, do not build from this
app/api_client.py                      preview_registration(), commit_registration()
app/app_v28.py                         step 5 "Register in the Knowledge Bank", _render_registration_step;
                                       KB load renumbered to 6 and gated on registration
CLAUDE.md                              six stages; Stage 1 does not write tile_coordinates;
                                       KB coverage table
docs/make_architecture_pdf.py          NEW — generates docs/HPL_Pipeline_Architecture.pdf
.gitignore                             ~600 MB of CSV exports, dumps and clinical data named
```

### Tests — 338 → 384

```
backend/tests/test_schema_coverage.py  NEW, 14 — every live table has DDL; the migration is
                                       idempotent; the base tables run before what alters them;
                                       and companions proving each check can fail
backend/tests/test_pipeline_steps.py   NEW, 16 — every step has a renderer; the numbering is
                                       contiguous; the KB load waits for registration and opens
                                       once it is done; the client payload matches the endpoint's
                                       pydantic model
backend/tests/test_register_dataset.py 10 → 26 — slide matching by the same rule Stage 1 used;
                                       upper-casing (every reader looks it up that way);
                                       ambiguous slide refused rather than picked; cross-cohort
                                       slide_id collision refused and rolled back; NaN reaches
                                       the database as NULL
```

Not run against a real PostgreSQL — there is none on the machine this was written on. The DDL is
transcribed and reviewed, not executed. **Preview the first real cohort before committing it.**

---

## 9. What to do next, in order

1. **Apply the migrations** on the HPC:
   `psql -h <socket> -d hpl_kb -f backend/migrate_all.sql` (idempotent; a no-op on an up-to-date
   database).
2. **Deploy `backend/` to the cluster and restart the server.** The API process runs on the login
   node — it shells out to `sbatch` and opens `.svs` files from scratch — so a local run does not
   pick these changes up. `app/` runs on the laptop.
3. **Capture the five missing `\d` outputs** (§6) and append them to
   `backend/kb_live_schema_2026-08-26.txt`.
4. **Preview a registration** on the 10-slide Radiogenomics run before the 14,044-slide one.
   Check `slides_without_files` and `ambiguous_slides` are both zero.
5. **Decide what `tile_hpc_heatmap` should hold** (§3.4), or accept that new cohorts have no
   probability overlay.
