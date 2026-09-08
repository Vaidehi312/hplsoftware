# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline and web UI for assigning **HPL histomorphological phenotype clusters (HPCs)** to H&E
whole-slide-image tiles, running on the Beatson HPC (Slurm). Whole slides are tiled, packaged into
HDF5, encoded by a frozen self-supervised model, classified by k-NN against a reference, and loaded
into a Postgres "Knowledge Bank" that the UI and chatbot read.

## Commands

```bash
# Tests. Every suite also runs standalone (no pytest), which is how they run on the cluster.
python3 -m pytest backend/tests -q
python3 backend/tests/test_extraction_guards.py      # standalone mode
python3 -m pytest backend/tests/test_kb_load.py::test_join_key_matches_the_registrys_format

# Services (both must restart to pick up changes; they are separate processes)
cd backend && python tile_server_v2_.py              # FastAPI on :8000
streamlit run app/app_v28.py                         # UI, talks to localhost:8000

# One-time cluster setup
python backend/submit_feature_extraction.py --bootstrap-extras   # installs container extras
python backend/build_hpc_reference.py --h5ad <leiden .h5ad> --out hpc_reference_leiden_2p5_fold2.npz
```

Dependencies: `backend/requirements.txt`. No lint or build step.

**Pick the highest version number.** `tile_server_v2_.py` (4.7k lines) is current despite the name —
`tile_server_v3.py`/`v4.py` are smaller, older experiments. `app/app_v28.py` is the current UI; v3–v27
are history. Confirm with `ls -lt` before editing.

## Architecture

Six stages, surfaced as numbered steps in the UI sidebar. Stage state lives in Postgres
(`slurm_dataset_runs`), and the UI gates each stage on the previous one having produced a *valid*
output, not merely having run.

```
1 Tiling            submit_mask_tile_slurm.py     → tiles + per-slide _tile_metadata.csv ON DISK
2 Packaging         make_hpl_hdf5.py              → one gzip HDF5 per dataset
3 Feature extract   submit_feature_extraction.py  → projections .h5 (GPU, Singularity)
4 Classification    submit_cluster_assignment.py  → assignments CSV (CPU, faiss k-NN)
5 Registration      register_dataset.py           → wsi_registry, wsi_metadata, dataset_config,
                                                    tile_coordinates, tile_registry (no hpc_id yet)
6 KB load           load_hpc_assignments.py       → tile_registry.hpc_id + per-slide aggregates
```

**Stage 1 does not write `tile_coordinates`.** It writes a per-slide `_tile_metadata.csv` to disk;
nothing puts those rows in Postgres until Stage 5. This file said otherwise until 2026-08-26 and the
mistake is worth naming, because it is what made the gap below invisible for so long.

Those name where each stage's *logic* lives; how it reaches Slurm is not uniform. Stages 3 and 4 have
dedicated `submit_*.py` modules that build and run their own `sbatch`. Stage 1 also has a shell
wrapper, `submit_dataset_tiling.sh`, which expects a `mask_and_tile_array.sbatch` alongside it that is
not in this repo — it lives on the cluster. Stage 2's submission is
driven by the server's `/package` endpoint, and `make_hpl_hdf5.package_slides_to_h5` is additionally
imported and called in-process on the single-slide upload path. Stages 5 and 6 can run **either**
way: `register_dataset.py`'s and `load_hpc_assignments.py`'s functions run in-process inside
`tile_server_v2_.py` (`/register-preview`, `/register`, `/kb-load-preview`, `/kb-load`), and the same
two CLIs are submitted to Slurm by `submit_kb_write.py` via `/register-submit` and `/kb-load-submit`
(`migrate_dataset_runs_kb_slurm.sql` adds their job_id/state pair). The in-process path was never
lost to a closed browser — FastAPI runs a sync endpoint in a threadpool and uvicorn does not cancel it
on client disconnect — but it dies with the server, which is why the Slurm path exists. `registration_done`
and `kb_load_done` mean *committed* on both paths: on the Slurm path the **job** sets them, through
`--record-run` and `run_record.py`, because the job is the only process that knows.

**A Slurm-backed KB write needs Postgres reachable from a compute node,** and nothing about the
server's own connection tells you whether it is. `DB_HOST=127.0.0.1` means the compute node itself,
and a unix socket path is local to the database's machine however shared the filesystem is — so
`submit_kb_write.resolve_job_db_host()` refuses both at submit time and demands `HPL_JOB_DB_HOST`.
Check it once with `python backend/submit_kb_write.py --check-db` (or `GET /kb-job-db-check`), which
sruns a one-second TCP probe from an actual compute node.

**Stage 5 exists because Stage 6 only ever `UPDATE`s.** `load_hpc_assignments.load()` sets
`tile_registry.hpc_id` on rows that must already be there, so for a cohort that has never touched the
KB its match rate is 0% by construction and it refuses — a number that reads like a slide-naming bug
and is in fact a missing step. Registration is gated on Stage 2, not Stage 4: it reads tile identity
out of the packaged `.h5` and the raw slides and needs no cluster labels.

Adding a stage means touching four places: the submitter (or, for a non-Slurm stage like 5 or 6, the
in-process functions it calls), endpoints in `tile_server_v2_.py`, methods in `app/api_client.py`, and a
step entry + render function in `app/app_v28.py`. `backend/tests/test_pipeline_steps.py` checks the
last two agree — a step key with no renderer is a `KeyError` on a screen nobody opens until a run
reaches that stage. Stage 4 is the cleanest Slurm-backed model to copy —
see `_pipeline_steps` and `_render_assignment_step`. Stage 6 (`_render_kb_load_step`) is the model for a
stage that writes straight to the KB: it never auto-commits — a `/kb-load-preview` dry run has to be
pulled up in the UI first, and `/kb-load` enforces the exact same guards (95% match rate, unknown
cluster IDs) the CLI's `--commit` does, because it calls the same functions rather than reimplementing
them.

### Two repositories

`HPL-LATTICeA/` is a **git subtree** of `K-Rakovic/HPL-LATTICeA` (upstream `aec5145`, remote `hpl`),
holding Kai's frozen encoder. Our changes to it are ordinary commits **and** are mirrored in
`backend/patches/hpl-encode-io.patch` — because the HPC has its own separate clone of that repo at
`$HPL_REPO_DIR`, which the subtree does not update. Changing the encoder means updating both.
`backend/tests/test_encode_loop_equivalence.py` fails if a `git subtree pull` reverts the patch.

### The container

Stages 3 and 4 run inside an NGC TensorFlow 1.15 image (`tensorflow-23.03-tf1-py3.sif`, CUDA 12, Python
3.8) because the host conda env cannot drive a Hopper GPU. The image lacks packages the code needs
(`scikit-image`, `faiss-cpu`), and `$HOME` is not writable inside it, so they live in a bound
directory on scratch (`HPL_CONTAINER_EXTRAS`) placed on `PYTHONPATH`. **The package list lives in
`submit_feature_extraction.py`, so that file and `--bootstrap-extras` must always move to the cluster
together** — a stale copy silently installs the wrong set.

### The Knowledge Bank

Postgres `hpl_kb`, 26 relations — 17 tables, 1 view, 8 sequences. Column-level definitions are in
`backend/kb_live_schema_2026-08-26.txt`, transcribed from `\d` against the live database; that capture
is the only complete record of this schema, and `backend/migrate_kb_base_tables.sql` was written from
it. `backend/migrate_all.sql` builds 13 of the 17 tables from an empty database — verified by running
it against a real PostgreSQL — and stops at the four `hpc_*` reference tables, whose `\d` has never
been captured. **`schema.sql` in the repo root is a stale 2025-10-23 `pg_dump`** that declares
`tile_registry.hpc_id` as `varchar(100)` with an `id` primary key and no `slide_tile`, `dataset_id`, or
confidence columns. Do not build a database from it.

Three things matter for classification results:

- `tile_registry` — per-tile `hpc_id` plus confidence columns. Joined via
  `slide_tile`, which is `"<slides>_<tiles>"` upper-cased (`TCGA-55-7574-01Z-00-DX1_18_15.JPEG`).
- `hpl_profile_proportion` / `hpl_profile_summary` — per-slide aggregates **derived** from the same
  assignments. These are what the chatbot and HPC panels read, *not* `tile_registry`. Writing tiles
  without refreshing these leaves the UI internally inconsistent with nothing to signal it.
- `hpc_dictionary` and the `hpc_*_details` tables describe the 71 clusters themselves. Reference data;
  never derived from an assignment.

Which tables a run actually fills, and which nothing fills — audited 2026-08-26, full evidence in
`KB_TABLE_COVERAGE_2026-08-26.md`:

| filled by a run | never filled by anything |
|---|---|
| `tile_coordinates`, `tile_registry`, `wsi_registry`, `wsi_metadata`, `dataset_config` (Stage 5) · `hpl_profile_*`, `slide_hpc_membership` (Stage 6) · `slurm_dataset_run*` (the server) | `tile_hpc_heatmap` — **read live** at server startup, nothing writes it · `h_latent_vectors` (4.4 GB) · `tile_hpc_heatmap_old` |

`tile_hpc_heatmap` is the one gap with a live reader still open. `_load_heatmap_probs()`
(`tile_server_v2_.py:1332`) reads the whole 149 MB table into memory at startup and merges its 71
`p_hpc_*` columns into `/slide/{id}/tiles_meta`. Nothing in this repository has ever written it, and
the k-NN classifier does not produce a 71-class distribution to write — it produces a top-1 label and
a vote margin. Filling it is a modelling decision, not plumbing.

**A grep will not find every reader.** `app/hpc_chat_handlers_v23.py:334` enumerates the whole
database with `insp.get_table_names()`, keeps every table carrying an `hpc_id` or `dominant_hpc`
column — skipping only `hpc_dictionary` and `h_latent_vectors` — and renders up to five matching
rows straight to the user. So any such table is answered out of the chatbot without ever being
named. That is how `slide_hpc_membership` looked unreferenced while serving stale rows for cohorts
nobody was asking about; Stage 6 now refreshes it alongside the aggregates. Before concluding a
table is dead, check whether it has an `hpc_id` column.

`h_latent_vectors` is the one table with genuinely no live reader — it is in that skip list by name,
and nothing else touches it. Leave it alone rather than "completing" it.

## Invariants that cost time to rediscover

**Feature extraction is read-bound, not GPU-bound.** Each tile is its own gzip chunk, so a batch read
is one decompression per tile: ~1.4k tiles/s, below what any available GPU encodes. Consequences:
raising `--batch-size` buys almost nothing; h5py serialises HDF5 calls on a global lock so reader
threads do not parallelise decode (separate processes do — hence `--shards`); and falling back from an
H200 to an H100/A100 costs almost nothing in wall clock.

**`--cleanenv` means the container cannot read Slurm's variables.** Stages 3 and 4 run their
work inside `singularity exec --cleanenv`, which wipes the environment before the inner shell
starts — so a thread count written as `${SLURM_CPUS_PER_TASK:-1}` and expanded in there always
took the fallback, and every cluster assignment ran on one core while Slurm held 16. Nothing
failed and nothing warned: a 2.5M-row reference at 127 dims came to 49 tiles/s, and 18.5M tiles
took three days instead of hours. Thread counts are baked in at submit time now
(`_build_assignment_command(threads=...)`), which means the number and the sbatch's
`--cpus-per-task` live in different strings and must be changed together —
`test_assign_streaming.py` pins both. Anything else the job needs from the submitting
environment has the same problem.

**GPU type names are cluster-specific and matching is exact.** This cluster has `nvidia_h200`,
`nvidia_h100_80gb_hbm3`, `nvidia_h100_pcie`, `nvidia_a100_80gb_pcie`. A preference list naming types
that do not exist is inert, not approximate — it silently falls through to queueing for the first one.
Check with `sinfo -p gpu -o "%N %G %t %D"`; note each type appears twice per node line.

**`--centering query` couples every tile to every other one.** It mirrors `sc.tl.ingest` by subtracting
the mean over *all* queries, so chunking or sharding the assignment changes the labels unless the mean
is computed once and shared (`--precompute-mean` / `--query-mean`). `project()` therefore refuses to
derive a mean from the chunk it was handed. Sharding without the shared mean is rejected outright.

**The encoder has no resume and mishandles its own leftovers.** It creates the output with `mode='w'`
before encoding, and its "output already exists" path crashes on an unbound local. So any interrupted
attempt makes every retry fail in seconds with an error pointing nowhere near the cause — which is why
`validate_extraction_output()` exists and stale outputs are cleared before resubmission.

**Short tile names are repaired on read, not refused — except when mixed.** A `.h5` or
assignments CSV packaged before make_hpl_hdf5.py stored the suffix holds `18_15`, which joins
nothing in a KB keyed on `..._18_15.JPEG`. `register_dataset.py` and `load_hpc_assignments.py`
append `.jpeg` themselves (`slide_naming.normalize_tile_names`) and report the count, because
`auto_tile_from_mask.py:150` writes every tile as `{col}_{row}.jpeg`, which makes the mapping a
bijection rather than a guess. Both sides must be normalised — the `.h5` *and* Stage 1's
metadata CSVs — or the refusal just becomes `tiles_with_coordinates: 0`. A **mixed** file (some
names suffixed, some not) is still refused: that is a resume that straddled the fix, the two
sides are indistinguishable by name, and appending would attach correct cluster IDs to the wrong
tiles. Note `tiles_missing_suffix()` cannot see that case — it samples 100 names and needs them
all short — which is why `tile_name_verdict()` reads every name. `migrate_tile_names.py` still
exists and is still the only fix for the artifacts on disk, and for mixed.

**Reference `.npz` keys are `reference, components, codes, categories, n_neighbors, meta` (+ optional
`mean`).** There is no `labels` key. `build_hpc_reference.save()` is the authority;
`test_reference_keys_match_the_builder` round-trips through it so readers cannot drift.

**k-NN search is faiss-only, exact, with no backend choice.** `Searcher` in `assign_hpc_clusters.py`
requires `faiss` and always builds an exact flat index (`IndexFlatL2`) — there is no numpy fallback and
no approximate (`faiss-ivf`) option anymore. An approximate index was tried and measured against the
real reference: it agreed on the nearest neighbour only 33% of the time, for no speed gain at this
reference size, which would have put an approximation inside the one number this pipeline is judged on.
`faiss-cpu` is in `backend/requirements.txt` and in the container's `CONTAINER_EXTRAS`
(`submit_feature_extraction.py`), so it's expected to always be present; if it's missing, that's a setup
defect to fix, not something to route around.

## The failure mode this codebase is written against

Almost nothing here fails by crashing. A wrong reference, a naming mismatch, a shard with its own mean,
a half-merged output, a stale aggregate — each produces a file or table of the right shape and dtype
with no missing values, which passes every completeness check while every cluster ID is attached to the
wrong tile. This is why guards refuse before the queue rather than inside the job, why validators check
row counts against their source rather than believing the parts, why outputs are written under a
temporary name and renamed only when whole, and why `load_hpc_assignments.py` is dry-run by default and
refuses below a 95% match rate.

Two things follow for anyone extending this. Prefer a loud refusal at submit time over a plausible
result later. And when adding a test, make it prove the guard can *fail* — several bugs here were found
by checking that a validator could come out bad, not that it came out good.

## Validation

```bash
# Does k-NN recover the reference's own labels? Needs only the .npz.
python backend/validate_reference.py --reference hpc_reference_leiden_2p5_fold2.npz

# Full end-to-end: reproduce Kai's TCGA cluster transfer. Needs TCGA projections.
python backend/submit_cluster_assignment.py --projections-h5 <TCGA .h5> --out /tmp/check.csv \
  --validate-against TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv
```

The first covers the search, the vote and whether the clusters are k-NN-separable. It does **not**
cover projecting raw embeddings into the reference space or the centering choice — only the second
does. Below 99% agreement on the second is a defect, not drift.
