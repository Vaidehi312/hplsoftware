# WSI packaging pipeline — hardening & performance pass

**Date:** 2026-08-01 / 02
**Scope:** `backend/` packaging pipeline (tiling → `.h5` packaging → feature extraction) and the `app/app_v28.py` pipeline UI.
**Trigger:** a review of `backend/tile_server_v2_.py` looking for anything that would crash a packaging run.

---

## TL;DR

A packaging run that was killed by `TIMEOUT` or OOM could never be recovered — every retry crashed on the truncated `.h5`. Fixing that surfaced a cluster of related problems: silent slide loss, unverifiable outputs, Slurm queries that treated failure as success, and a chunk layout doing up to 3000× the necessary I/O.

Sixteen defects fixed across five commits. The **write stage** is 6.9× faster; **end-to-end is only 1.1×** on local SSD, because the pipeline is decode-bound (see [Measurements](#measurements)). Nothing has run on the HPC yet — see [Before your next run](#before-your-next-run).

---

## Commits

| SHA | Title |
|---|---|
| `c0a70ea` | Harden WSI tile packaging against interruption and silent data loss |
| `94e9cbd` | Speed up .h5 packaging writes and stop over-requesting tiling CPUs |
| `dde415a` | Flush the .h5 before recording tiles as complete in the checkpoint |
| `b4a42ed` | Parse every sacct JobID form, not just completed array tasks |
| *uncommitted* | `packaging-progress` endpoint, `api_client` changes, `app_v28` pipeline stepper |

All commits touch `backend/` only. **`app/` is not tracked in git** — the UI work is untracked on disk.

---

## What was fixed

### Correctness / data integrity

**1. Retrying packaging after a kill crashed permanently** — `make_hpl_hdf5.py`
The resume guard checked `output_h5_path.is_file()` but never that the file *opened*. A `SIGTERM` leaves a truncated `.h5` plus a matching checkpoint, so resume engaged and raised `OSError: truncated file` — on that attempt and every retry after it. New `_h5_resumable()` probes the file in the same `"r+"` mode the write path uses and falls back to a fresh run.

**2. Subset runs discarded their checkpoint on every retry** — `tile_server_v2_.py`
`_effective_h5_dataset_name` numbers `_subset_N` by scanning disk, and was called on *every* `/package` request. A retry got `_subset_3` where the first attempt wrote `_subset_2`, orphaning hours of decoded tiles (checkpoints are keyed off the output path). Now the name is read back from the run's recorded `h5_output_path`.

**3. Corrupt tile metadata silently lost slides** — new `backend/tile_metadata.py`
Resume treated "the CSV exists" as done; packaging tried to parse it. A CSV truncated by a killed tiling task was therefore *never re-tiled* **and** *dropped from the `.h5`* — the slide vanished with nothing reporting it. One shared classifier now returns `ok` / `empty` / `corrupt` / `missing`, with `empty` (genuine zero-tissue) deliberately excluded from retries. Also stops a `ParserError` aborting a whole multi-hour run over one bad file.

**4. Checkpoint recorded tiles before their data was durable** — `dde415a`
*(Caught by Vaidehi, not by the review.)* `img_ds[a:b] = ...` returning only means the data reached HDF5's chunk cache — which commit `94e9cbd` had just enlarged from 1 MiB to 64 MiB (~430 tiles). Recording labels first meant a kill in that window left resume skipping past rows that never reached disk: **permanent zero-filled holes**, never re-decoded because the labels claimed they were done. Now `hdf5.flush()` runs between the writes and the label append.

> **Scope limit:** `H5Fflush` hands data to the filesystem; it is **not** an `fsync`. Covers process death (TIMEOUT/OOM, node up). Does *not* cover node or power loss.

**5. Output written non-atomically** — `94e9cbd`
Packaging now stages to `<final>.h5.partial` and `os.replace()`s on success. Previously the final path existed from the instant packaging *started*, so its presence meant nothing — and a re-run clobbered a previous good `.h5` before knowing whether the new one would succeed. Verified: a mid-write failure leaves the previous good file byte-identical.

**6. Unverified outputs marked ready** — `_validate_h5()`
Readiness was existence + Slurm state. An `.h5` truncated after its header opens cleanly and only fails when the missing chunks are read — which previously happened for the first time *hours into a GPU job*. Now checks datasets present, non-zero, lengths agree, and reads the first **and last** rows.

**7. Feature extraction accepted anything** — `tile_server_v2_.py`
Was `h5_path.is_file()`. Now requires packaging `COMPLETED` **and** a passing HDF5 read, with distinct messages for still-running / failed / unreadable / Slurm-unreachable (503).

**8. Slurm query failures read as success** — `_run_slurm()`
A failing `sacct` writes to stderr and leaves stdout empty — byte-identical to "no rows". And "no rows" is what `_job_output_ready` interpreted as *"aged out of retention, so it finished successfully."* A database hiccup could be promoted into proof a job succeeded. All six query sites now check return codes; `squeue`'s "Invalid job id" is preserved as a legitimate "not live".

**9. sacct parser dropped three of four row types** — `b4a42ed`

| sacct emits | was | now |
|---|---|---|
| `990_[4-9003]\|PENDING` | dropped | counted as 9,000 pending |
| `991_2\|CANCELLED by 40315` | dropped | `CANCELLED`, ×1 |
| `992\|COMPLETED` (non-array) | dropped | counted |
| `990_0.batch\|COMPLETED` (step) | excluded by accident | excluded explicitly |

Consequence: `tiling_complete` could report true while thousands of slides were still queued, and the new `afterok` guard could not see cancellations at all.

**10. `afterany` was packaging datasets with holes** — `94e9cbd`
The docstring justified `afterany` by not wanting zero-tile slides to block packaging — but `run_worker()` **exits 0** for a zero-tile slide. It only exits non-zero on genuine failure. So `afterany` protected nothing and silently packaged incomplete datasets. Now `afterok` + `--kill-on-invalid-dep=yes`, with `allow_incomplete=true` as an explicit opt-out (automatic for `/package-test`).

**11. Fork-unsafe HDF5** — `94e9cbd`
`ProcessPoolExecutor` was created *inside* the open `h5py.File` block, so workers inherited the file descriptor and h5py's atexit handler. HDF5 is not fork-safe. Pool is now created first.

### Performance

**12. Pathological chunk shape** — the single biggest defect
`maxshape` without `chunks` forces a chunked layout and lets h5py guess — and its guess subdivides the *pixel* dimensions:

| dataset size | auto chunk shape | chunks touched per image write | amplification |
|---|---|---|---|
| 400 tiles | `(50, 28, 56, 1)` | 96 | 50× |
| 4,000 tiles | `(250, 14, 28, 1)` | 384 | 250× |
| 100,000 tiles | `(3125, 7, 14, 1)` | 1,536 | **3125×** |

Now `chunks=(1, tile_size, tile_size, 3)` — one image write = one chunk write, and the right shape for training reads too.

**13. Row-by-row writes → batched slices**, **14. chunk cache 1 MiB → 64 MiB**, **15. tiling `cpus` 4 → 1** (neither `auto_tile_from_mask.py` nor `tile_mask.py` has any internal parallelism), **16. `max_concurrent` divided across batches** — Slurm's `%N` throttle is *per array job*, so `12` across 14 batches meant **168** concurrent tasks at 672 cores, not 12.

---

## Measurements

All on local SSD (laptop), 3,000 synthetic 224px tiles, 4 processes.

**Stage breakdown**

| stage | before | after |
|---|---|---|
| JPEG decode | 3.09s (146 MB/s) | 3.09s — unchanged |
| HDF5 write | 3.25s (139 MB/s) | **0.96s (471 MB/s)** |
| **end-to-end** | **6.08s** | **5.39s — 1.1×** |

**Write stage vs dataset size**

| tiles | before | after | speedup |
|---|---|---|---|
| 2,000 | 141 MB/s | 795 MB/s | 5.7× |
| 6,000 | 132 MB/s | 910 MB/s | 6.9× |
| 12,000 | 134 MB/s | 921 MB/s | 6.9× |

**Durability cost of the per-chunk flush:** +14% (12 flushes of ~37 MB over 3,000 tiles).

### Honest reading of these numbers

An earlier claim in this session that the chunking fix would give "10×+ end-to-end" was **wrong**, and the correction matters:

- **Decode dominates and overlaps the writes.** The rolling window already hides most write time behind decoding, so removing write time from a decode-bound pipeline buys little.
- **The page cache absorbs the amplification on local disk.** I predicted the "before" case would degrade as chunks grew taller. It doesn't — it's flat at ~135 MB/s, because those hundreds of chunk touches hit RAM.

That second point is the load-bearing assumption, and it is **false on a laptop but should not hold on network-mounted scratch**, where each chunk touch is a round trip no page cache absorbs. Unverified — measure it on the HPC.

**The real bottleneck is JPEG decode (~146 MB/s on 4 processes).** Two levers, in order of value:
1. **More cores.** `submit_packaging_job` requests `cpus=8` and passes that as `--processes`. Raising to 16–32 is one line with near-linear return.
2. **Stop decoding at all.** You decode JPEGs that tiling wrote minutes earlier, to store raw uint8. If tiling wrote raw tiles or per-slide `.h5` shards, packaging becomes a concatenation.

---

## UI changes (`app/app_v28.py`, uncommitted)

The old renderer walked a nested if/else and drew **only the current stage's widget**, so "what's finished / what's left" could not be answered from the screen.

Now a three-step pipeline, always fully listed:

```
✅ 1. Tiling — 2431/2431 slides have tiles
🟠 2. Packaging (.h5) — interrupted (TIMEOUT)      ← auto-expanded
⚪ 3. Feature extraction — waiting on packaging
```

`✅` done · `🔄` running · `🔵` ready to start · `🟠` needs a decision · `❌` failed · `⚪` blocked. Steps needing you expand automatically.

- **Each step opens to `Full dataset` / `Test on a subset`.** The test sections existed in `app_v27` and were dropped in v28; restored per-step, wired to client methods that already existed with zero callers.
- **Resume for interrupted packaging** — new `GET /dataset-jobs/{id}/packaging-progress` reads the `.partial` and checkpoint:
  ```
  [███████░░░░░░░░░░░░] 412,388 / 1,204,551 tiles (34.2%)
  Unfinished .partial on disk: 58.3 GB
  [Resume packaging (412,388 tiles already done)]
  ```
  Called only when a step is actually interrupted — never on the 10s poll, since it counts lines in a file that can reach hundreds of MB. Says so plainly when there is no resumable checkpoint.
- **`400 tiling_incomplete` handled** — renders the failed task states and offers "Package anyway (accept missing slides)".

---

## Testing

**76 assertions**, all passing:

| suite | count | covers |
|---|---|---|
| `test_hardening.py` | 42 | CSV integrity, HDF5 validation, atomic publish, readiness gate, afterok, Slurm return codes |
| `test_sacct.py` | 11 | all four sacct JobID forms, range expansion, state normalisation |
| `test_stepper.py` | 12 | step classification across 11 lifecycle scenarios |
| endpoint test | 5 | `packaging-progress` against real checkpoint files |
| `test_flush_order.py` | 4 | checkpoint never runs ahead of a flush; no zero-filled holes |
| `test_resume.py` | 2 | resume on intact `.partial`; no crash on truncated |

### ⚠️ These tests live in the session scratchpad, not the repo

They will be lost. The flush-ordering and chunk-shape tests especially are worth keeping — both guard against regressions that are *invisible* without them (silent holes; "packaging just got slow again").

### What is NOT tested

- **Nothing has run on the HPC.** All testing used synthetic tiles, fake `squeue`/`sacct` shell scripts, a stubbed `skimage`, no Postgres, no Slurm, no real `.svs`, local SSD instead of network scratch.
- **The Streamlit widget code is unexercised.** `_pipeline_steps` was verified by exec'ing the real function; rendering paths, session-state keys and button handlers were not run.

---

## Before your next run

### Will likely stop you cold

1. **Raise `max_concurrent` to ~150.** The semantics fix made the default far more conservative than what you were running — `12 // 14 batches = 1` per batch means **14 concurrent tasks** for a 14,000-slide run.
2. **Check `MaxMemPerCPU`.** `--mem=24G` at `--cpus-per-task=1` may be rejected or silently inflated:
   ```bash
   scontrol show config | grep MaxMemPerCPU
   ```
3. **Verify the DB columns.** The packaging path needs 16; twelve come from four separate migration files and none are in `schema.sql`:
   ```sql
   SELECT column_name FROM information_schema.columns WHERE table_name='slurm_dataset_runs';
   ```
   Required: `dataset_name`, `is_subset`, `h5_job_id`, `h5_output_path`, `partition`, `notify_email`, `extraction_job_id`, `extraction_output_path`, `extraction_checkpoint`, `submission_id`, `status`, `error`, plus `job_id`, `manifest_path`, `raw_dir`, `tile_dir`, `total_slides`.
4. **Deploy `backend/tile_metadata.py`** — new file. Missing it is an `ImportError` that takes the whole server down.

### Expect on first run

- **`afterok` will probably block your first packaging attempt.** Across 14,000 slides at least one failed task is near-certain, so expect `400 tiling_incomplete`. Intended — resume the failed slides, or use the "Package anyway" button.

### Recommended

Run **20–50 slides end-to-end** (tile → package → extract) before the full dataset. About an hour, exercises every changed path against real Slurm/Postgres/network I/O, and gives the honest read on whether the chunking fix helps on your filesystem.

---

## Outstanding

| item | notes |
|---|---|
| **`.gitignore`** | Still absent after 4 commits. A 400 MB `.svs`, `.h5` files and `hpl_kb_dump.pgsql` are one `git add -A` from permanent history. |
| **Move tests into the repo** | 76 assertions currently in scratchpad only. |
| **Commit the UI work** | `app/` is untracked entirely; needs `.gitignore` first (`app/__pycache__`, `app/.DS_Store`, `.streamlit/` may hold secrets). |
| **Packaging `cpus` 8 → 16–32** | One line, near-linear return on the decode bottleneck. Measure first. |
| **Sharded packaging + HDF5 Virtual Dataset** | The scale answer: a Slurm array packaging disjoint slide shards, stitched with a VDS (no data copy). Several hundred lines; deserves its own pass with real numbers. |
| **Flush amortisation** | If the +14% matters: accumulate labels across *K* chunks, flush once, record all *K*. Same invariant, fewer flushes, more redone work on resume. |

---

## Files touched

```
backend/tile_metadata.py          NEW  — shared tile-metadata classifier
backend/make_hpl_hdf5.py               — resume guard, .partial publish, chunks,
                                         batched writes, chunk cache, fork order, flush
backend/tile_server_v2_.py             — HDF5 validation, extraction gate, Slurm return
                                         codes, sacct parsing, subset naming,
                                         packaging-progress endpoint, allow_incomplete
backend/submit_mask_tile_slurm.py      — afterok, kill-on-invalid-dep, cpus, max_concurrent
backend/find_missing_slides.py         — integrity-aware, detailed breakdown
app/api_client.py                      — allow_incomplete, get_packaging_progress, POST params
app/app_v28.py                         — pipeline stepper, test sections, resume UI
```
