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

Five stages, surfaced as numbered steps in the UI sidebar. Stage state lives in Postgres
(`slurm_dataset_runs`), and the UI gates each stage on the previous one having produced a *valid*
output, not merely having run.

```
1 Tiling            submit_mask_tile_slurm.py     → tiles on disk + tile_coordinates
2 Packaging         make_hpl_hdf5.py              → one gzip HDF5 per dataset
3 Feature extract   submit_feature_extraction.py  → projections .h5 (GPU, Singularity)
4 Classification    submit_cluster_assignment.py  → assignments CSV (CPU, faiss k-NN)
5 KB load           load_hpc_assignments.py       → tile_registry + per-slide aggregates
```

Those name where each stage's *logic* lives; how it reaches Slurm is not uniform. Stages 3 and 4 have
dedicated `submit_*.py` modules that build and run their own `sbatch`. Stage 1 also has a shell
wrapper, `submit_dataset_tiling.sh`, which expects a `mask_and_tile_array.sbatch` alongside it that is
not in this repo — it lives on the cluster. Stage 2's submission is
driven by the server's `/package` endpoint, and `make_hpl_hdf5.package_slides_to_h5` is additionally
imported and called in-process on the single-slide upload path. Stage 5 is CLI-only so far.

Adding a stage means touching four places: the submitter, endpoints in `tile_server_v2_.py`, methods in
`app/api_client.py`, and a step entry + render function in `app/app_v28.py`. Stage 4 is the cleanest
model to copy — see `_pipeline_steps` and `_render_assignment_step`.

### Two repositories

`HPL-LATTICeA/` is a **git subtree** of `K-Rakovic/HPL-LATTICeA` (upstream `aec5145`, remote `hpl`),
holding Kai's frozen encoder. Our changes to it are ordinary commits **and** are mirrored in
`backend/patches/hpl-encode-io.patch` — because the HPC has its own separate clone of that repo at
`$HPL_REPO_DIR`, which the subtree does not update. Changing the encoder means updating both.
`backend/tests/test_encode_loop_equivalence.py` fails if a `git subtree pull` reverts the patch.

### The container

Stages 3–5 run inside an NGC TensorFlow 1.15 image (`tensorflow-23.03-tf1-py3.sif`, CUDA 12, Python
3.8) because the host conda env cannot drive a Hopper GPU. The image lacks packages the code needs
(`scikit-image`, `faiss-cpu`), and `$HOME` is not writable inside it, so they live in a bound
directory on scratch (`HPL_CONTAINER_EXTRAS`) placed on `PYTHONPATH`. **The package list lives in
`submit_feature_extraction.py`, so that file and `--bootstrap-extras` must always move to the cluster
together** — a stale copy silently installs the wrong set.

### The Knowledge Bank

Postgres `hpl_kb`, 26 relations. Three things matter for classification results:

- `tile_registry` — per-tile `hpc_id` plus confidence columns. Joined via
  `slide_tile`, which is `"<slides>_<tiles>"` upper-cased (`TCGA-55-7574-01Z-00-DX1_18_15.JPEG`).
- `hpl_profile_proportion` / `hpl_profile_summary` — per-slide aggregates **derived** from the same
  assignments. These are what the chatbot and HPC panels read, *not* `tile_registry`. Writing tiles
  without refreshing these leaves the UI internally inconsistent with nothing to signal it.
- `hpc_dictionary` and the `hpc_*_details` tables describe the 71 clusters themselves. Reference data;
  never derived from an assignment.

## Invariants that cost time to rediscover

**Feature extraction is read-bound, not GPU-bound.** Each tile is its own gzip chunk, so a batch read
is one decompression per tile: ~1.4k tiles/s, below what any available GPU encodes. Consequences:
raising `--batch-size` buys almost nothing; h5py serialises HDF5 calls on a global lock so reader
threads do not parallelise decode (separate processes do — hence `--shards`); and falling back from an
H200 to an H100/A100 costs almost nothing in wall clock.

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

**Reference `.npz` keys are `reference, components, codes, categories, n_neighbors, meta` (+ optional
`mean`).** There is no `labels` key. `build_hpc_reference.save()` is the authority;
`test_reference_keys_match_the_builder` round-trips through it so readers cannot drift.

**`faiss-ivf` is approximate and changes cluster labels.** Measured against the real reference it
matched 23% of the 250 neighbours and agreed on the nearest one 33% of the time, for no speed gain at
this reference size. `auto`, `faiss` and `numpy` are all exact and differ only in speed (~2,500 vs ~59
tiles/s). Default to `auto`.

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
