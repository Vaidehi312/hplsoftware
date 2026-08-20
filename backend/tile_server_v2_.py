"""
FastAPI Tile Server — runs on HPCC near the .svs files and PostgreSQL.

Endpoints
---------
GET  /health                         → liveness check
GET  /slides                         → list all slide IDs
GET  /slide/{slide_id}/info          → slide metadata (dimensions, levels, mpp)
GET  /dzi/{slide_id}.dzi             → Deep Zoom metadata for OpenSeadragon
GET  /dzi/{slide_id}_files/{z}/{x}_{y}.jpeg → Deep Zoom JPEG tile
GET  /debug/routes                   → list active FastAPI routes
GET  /slide/{slide_id}/thumbnail     → whole-slide JPEG preview
GET  /slide/{slide_id}/tile          → single tile at (level, x, y, w, h)
GET  /slide/{slide_id}/region        → arbitrary region in native coords
GET  /slide/{slide_id}/tiles_meta    → tile_coordinates + tile_registry + hpc join
GET  /slide/{slide_id}/adjacency     → precomputed adjacency pairs
GET  /hpc/{hpc_id}/info              → HPC dictionary + malignant/non-malignant details
GET  /hpc/{hpc_id}/survival          → survival analysis row
POST /upload-slide                   → upload + preprocess WSI, register it, and queue mask+tile pipeline
GET  /slide/{slide_id}/processing-status → poll background mask/tiling status after upload
GET  /dataset-roots                  → list submittable dataset directories under LONG_TERM_SCRATCH
POST /dataset-jobs                   → submit a dataset-wide masking+tiling Slurm array job
GET  /dataset-jobs                   → list past/active dataset job submissions
POST /dataset-jobs/{submission_id}/resume → resubmit whatever slides from a run never got tiled
POST /dataset-jobs/{submission_id}/package → user-triggered: package this run's tiles into .h5
POST /dataset-jobs/{submission_id}/extract-features → user-triggered: run the .h5 through the model
POST /dataset-jobs/{submission_id}/cancel → scancel every Slurm job for this run and mark it cancelled
GET  /dataset-jobs/{submission_id}/status → discovery/Slurm/per-slide tiling status for a dataset job
POST /query                          → full NL query → structured answer
GET  /tile_image/{slide_tile}        → H5-backed tile image by slide_tile key
"""

import getpass
import hashlib
import inspect
import io
import os
import re
import time
import json
import random
import subprocess
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import openslide
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openslide.deepzoom import DeepZoomGenerator
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from tile_cache import TileCache
from tile_mask import run_tissue_detection
from auto_tile_from_mask import tile_slide_from_mask
from slide_naming import slide_id_from_raw_path, tiles_missing_suffix
from submit_mask_tile_slurm import submit_array as submit_dataset_array
from submit_mask_tile_slurm import (
    submit_packaging_job,
    discover_slides,
    write_manifest,
    tiling_output_complete,
)
from make_hpl_hdf5 import _checkpoint_paths, package_slides_to_h5, hpl_h5_output_path
from find_missing_slides import find_missing_slides_detailed
from dataset_rollup import (
    coarse_run_state as _coarse_run_state,
    group_runs_by_dataset,
    job_ids as _split_job_ids,
    rollup_dataset,
)
from submit_feature_extraction import (
    submit_feature_extraction_job,
    expected_extraction_output_path,
    validate_extraction_output as _validate_extraction_output,
    _input_h5_rows as _packaged_h5_rows,
    HPL_REPO_DIR,
)

from submit_cluster_assignment import submit_cluster_assignment_job

from load_hpc_assignments import (
    read_assignments as _read_kb_assignments,
    inspect as _inspect_kb_load,
    load as _write_kb_load,
    compute_profiles as _compute_kb_profiles,
    _MIN_MATCH_RATE as _KB_MIN_MATCH_RATE,
)

# Columns assign_hpc_clusters.py writes. The cluster column itself is named
# after the reference's groupby (e.g. 'leiden_2.5'), so it is matched by
# elimination rather than by name — hardcoding a name here would break the
# moment the reference changes resolution, which is the kind of coupling that
# makes a validator call a healthy file broken.
_ASSIGNMENT_REQUIRED_COLUMNS = (
    "samples", "slides", "tiles", "vote_margin", "neighbor_distance", "hpc_reference",
)


def _validate_assignment_output(path: Path, expected_rows: int | None = None):
    """Confirm an assignment CSV is a full set of cluster IDs, not a stub.

    Same role as validate_extraction_output plays for Stage 3: a file existing
    at the right path is not evidence the job produced anything usable. A run
    killed partway leaves a CSV with a header and some rows, which reads as
    success to anything that only checks existence.
    """
    if not path.is_file():
        return False, "no output file"
    try:
        with path.open() as fh:
            header = fh.readline().strip()
            if not header:
                return False, "the file is empty"
            columns = [c.strip() for c in header.split(",")]
            missing = [c for c in _ASSIGNMENT_REQUIRED_COLUMNS if c not in columns]
            if missing:
                return False, f"missing column(s): {', '.join(missing)}"
            if len(columns) <= len(_ASSIGNMENT_REQUIRED_COLUMNS):
                return False, "no cluster-ID column alongside the metadata columns"
            rows = sum(1 for _ in fh)
    except OSError as e:
        return False, f"could not be read: {e}"

    if rows == 0:
        return False, "holds a header but no assignments"
    if expected_rows is not None and rows != expected_rows:
        return False, (
            f"holds {rows:,} assignments but the projections file has "
            f"{expected_rows:,} embeddings"
        )
    return True, ""

# sacct states that mean "still queued or actively running" — anything else
# (COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, ...) is
# terminal. Used to decide when a "start next stage" button should appear.
# COMPLETING is included even though the job's steps have finished — Slurm's
# epilogue (and any of the job's own writes still flushing to a network
# filesystem) may not be done yet, so a retry-guard or tiling_complete check
# that treated COMPLETING as terminal could let a second packaging/extraction
# job start writing the same output file while the first is still finishing.
# CONFIGURING is a squeue-only state (nodes allocated, prologue still running)
# that sacct rarely surfaces — it only started mattering once squeue became the
# first source consulted in _get_slurm_job_state.
IN_FLIGHT_SLURM_STATES = {
    "PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING",
}


# Configuration — edit these to match your HPCC environment

DB_USER = os.getenv("DB_USER", "vpandya")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hpl_kb")

WSI_ROOT = os.getenv("WSI_ROOT", "/hpc-home/home/users/vpandya/long-term-scratch/tcga_wsi")
H5_PATH = os.getenv("H5_PATH", "/hpc-home/home/users/vpandya/long-term-scratch/Vaidehi/TCGA/hdf5_TCGA_LUAD_5x_he_train_tiles.h5")

CACHE_DIR = os.getenv("TILE_CACHE_DIR", "/tmp/hpc_tile_cache")
UPLOAD_ROOT = Path(os.getenv(
    "UPLOAD_ROOT",
    "/hpc-home/home/users/vpandya/long-term-scratch/uploaded_wsi"
))
UPLOAD_RAW_DIR = UPLOAD_ROOT / "raw"
UPLOAD_METADATA_DIR = UPLOAD_ROOT / "metadata"

# Post-upload pipeline output — same defaults tile_mask.py / auto_tile_from_mask.py use standalone.
TISSUE_MASK_DIR = Path(os.getenv(
    "TISSUE_MASK_DIR",
    "/hpc-home/home/users/vpandya/long-term-scratch/tissue_masks"
))
PROCESSED_TILES_DIR = Path(os.getenv(
    "PROCESSED_TILES_DIR",
    "/hpc-home/home/users/vpandya/long-term-scratch/processed_tiles"
))
# Fraction of a tile's area that must be tissue to keep it. Tiles are now read
# at the paper's exact 403.2um physical size (~40x the area of the old fixed
# 256-native-px tiles), so a threshold tuned for those smaller tiles may need
# lowering here — tune via env var rather than editing code.
MIN_TISSUE_PERCENT = float(os.getenv("MIN_TISSUE_PERCENT", "30.0"))

# Where packaged .h5 files land — same default submit_packaging_job() uses
# (a sibling of backend/), so single-slide and dataset-wide packaging share
# one datasets/ root.
HPL_DATASETS_ROOT = Path(os.getenv(
    "HPL_DATASETS_ROOT", str(Path(__file__).resolve().parent.parent / "model_input")
))

# Dataset-wide Slurm masking+tiling jobs (submit_mask_tile_slurm.py). A
# submittable "dataset" is any direct subdirectory of LONG_TERM_SCRATCH other
# than the pipeline's own output/working directories below — this is
# recomputed on every request rather than hand-maintained, so it can't drift.
LONG_TERM_SCRATCH = Path(os.getenv(
    "LONG_TERM_SCRATCH",
    "/hpc-home/home/users/vpandya/long-term-scratch"
))

TILE_SIZE_5X = 224
SCALE = 1.8 / 0.252
TILE_SIZE_NATIVE = int(TILE_SIZE_5X * SCALE)


# Initialising Globals 

engine = None
cache: TileCache = None
_h5_handle = None
_wsi_handles: dict[str, openslide.OpenSlide] = {}
_dz_handles: dict[str, DeepZoomGenerator] = {}
_wsi_map: dict[str, str] = {}  # slide_id → hpc_path on HPCC
_heatmap_probs: pd.DataFrame | None = None
_processing_status: dict[str, dict] = {}  # slide_id → {status, stage, error, ...}
# submission_id → (succeeded, zero_tile, not_attempted) slide-id lists, once
# computed for the first time after that run's tiling is fully complete. A
# run's manifest never changes after it's written, so once tiling_complete
# is true this breakdown can never change either — safe to cache forever
# rather than re-scanning up to one file per slide on every 10s status poll,
# which was blowing past even a 60s HTTP timeout for a 14,000+ slide run.
# Lost on server restart, which just means one slow recompute, not a
# correctness issue.
_tiling_breakdown_cache: dict[str, tuple[list[str], list[str], list[str]]] = {}

# Serializes "check whether a packaging/extraction attempt is already in
# flight, then submit a new one if not" across all four submission
# endpoints below. Without this, that check-then-submit was two separate
# steps with nothing stopping two near-simultaneous requests (a double
# click, two browser tabs, or — what actually happened — repeated retries
# each hitting the guard before Slurm's own accounting had caught up on
# the previous attempt) from both passing the check and both submitting.
# Every one of these submissions writes to the exact same deterministic
# output path (real runs) or "<name>_test_sample" path (test runs), so two
# concurrent writers isn't just wasted compute, it's a genuine data race on
# the same file. Serializing all of them (even across different runs)
# costs nothing that matters here — these are rare, human-triggered
# actions, not a throughput-sensitive path.
#
# A threading.Lock() used to guard this, which only serializes within a
# single process. This server runs under multiple Uvicorn worker
# processes (see __main__ below), each with its own separate memory, so
# a plain in-process lock left the exact race above wide open across
# workers — two requests landing on different workers both saw "no
# attempt in flight" and both submitted. A Postgres advisory lock is
# process-agnostic (it's coordinated by the DB, not by memory this
# process happens to own), so it actually serializes across every worker
# talking to the same database, using the same `engine` already used
# everywhere else in this file.
_SLURM_LOCK_KEY = 927341  # arbitrary constant; identifies this one lock


@contextmanager
def _slurm_submission_lock():
    eng = _get_engine()
    conn = eng.raw_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_lock(%s)", (_SLURM_LOCK_KEY,))
        cur.close()
        conn.commit()
        try:
            yield
        finally:
            cur = conn.cursor()
            cur.execute("SELECT pg_advisory_unlock(%s)", (_SLURM_LOCK_KEY,))
            cur.close()
            conn.commit()
    finally:
        conn.close()


def _get_engine():
    global engine
    if engine is None:
        engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
            pool_pre_ping=True,
            pool_size=5,
        )
    return engine


def _get_h5():
    global _h5_handle
    if _h5_handle is None:
        _h5_handle = h5py.File(H5_PATH, "r", swmr=True)
    return _h5_handle


def _load_wsi_map():
    global _wsi_map
    eng = _get_engine()
    df = pd.read_sql("SELECT slide_id, hpc_path FROM wsi_registry", eng)
    df["slide_id"] = df["slide_id"].astype(str).str.strip().str.upper()
    _wsi_map = dict(zip(df["slide_id"], df["hpc_path"]))
 
UPLOADED_DATASET_ID = "UPLOADED"  # tags ad-hoc uploads apart from the bulk TCGA_LUAD_5x cohort


def _register_uploaded_slide(slide_id: str, hpc_path: str):
    """Register or update an uploaded WSI so existing viewer endpoints can open it."""
    slide_id = slide_id.strip().upper()
    eng = _get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO wsi_registry (slide_id, hpc_path, dataset_id)
                VALUES (:slide_id, :hpc_path, :dataset_id)
                ON CONFLICT (slide_id)
                DO UPDATE SET hpc_path = EXCLUDED.hpc_path
            """),
            {"slide_id": slide_id, "hpc_path": hpc_path, "dataset_id": UPLOADED_DATASET_ID},
        )

    _load_wsi_map()
    _wsi_handles.pop(slide_id, None)
    _dz_handles.pop(slide_id, None)


def _set_processing_status(slide_id: str, status: str, error: str | None = None):
    """Write-through: Postgres is the source of truth (survives restarts /
    would survive multiple workers); the in-memory dict is just a fast local
    cache for the common case of a single long-lived worker.
    """
    slide_id = slide_id.strip().upper()
    _processing_status[slide_id] = {"status": status, "error": error}
    try:
        eng = _get_engine()
        with eng.begin() as conn:
            conn.execute(
                text("""
                    UPDATE wsi_registry
                    SET processing_status = :status,
                        processing_error = :error,
                        processing_updated_at = now()
                    WHERE slide_id = :slide_id
                """),
                {"slide_id": slide_id, "status": status, "error": error},
            )
    except Exception as e:
        print(f"[{slide_id}] failed to persist processing_status={status}: {e}")


def _get_processing_status(slide_id: str) -> dict:
    slide_id = slide_id.strip().upper()
    try:
        eng = _get_engine()
        with eng.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT processing_status, processing_error
                    FROM wsi_registry WHERE slide_id = :slide_id
                """),
                {"slide_id": slide_id},
            ).fetchone()
        if row and row[0]:
            return {"status": row[0], "error": row[1]}
    except Exception as e:
        print(f"[{slide_id}] failed to read processing_status from DB: {e}")

    return _processing_status.get(slide_id, {"status": "not_started", "error": None})


def _run_postupload_pipeline(slide_id: str, raw_path: str):
    """Background job: tissue mask -> 224px tiles at Kai's 1.8um/px resolution.

    Runs after /upload-slide returns so the HTTP request doesn't block on a
    full-slide tiling pass. Progress is tracked via _set_processing_status so
    the UI can poll GET /slide/{slide_id}/processing-status.
    """
    try:
        _set_processing_status(slide_id, "masking")
        # Scoped under UPLOADED_DATASET_ID rather than flat under
        # TISSUE_MASK_DIR/PROCESSED_TILES_DIR directly — those same flat
        # dirs are also where bulk dataset submissions (TCGA, Radiogenomics,
        # ...) write, scoped by their own dataset_name (see run_worker() in
        # submit_mask_tile_slurm.py). Without this, an ad-hoc upload whose
        # user-supplied slide_id happens to match a real dataset's slide_id
        # would silently overwrite that dataset's mask/tiles on disk — a
        # second, filesystem-level version of the wsi_registry collision
        # /upload-slide already guards against; that guard alone doesn't
        # cover this, since it only checks the DB row, not these directories.
        # slide_id passed explicitly rather than left for these to derive
        # from raw_path themselves — raw_path is now stored under a
        # server-generated UUID directory (see /upload-slide) with no
        # slide_id embedded in the filename for slide_id_from_raw_path to
        # recover, unlike bulk/GDC datasets' own naming convention.
        upload_mask_dir = TISSUE_MASK_DIR / UPLOADED_DATASET_ID
        mask_result = run_tissue_detection(
            slide_path=raw_path,
            output_dir=str(upload_mask_dir),
            slide_id=slide_id,
        )
        # Backstop: confirm masking actually wrote where it was told to,
        # not just that slide_id's charset was valid — see _resolve_within.
        _resolve_within(upload_mask_dir, Path(mask_result["mask_path"]))
        _resolve_within(upload_mask_dir, Path(mask_result["overlay_path"]))

        _set_processing_status(slide_id, "tiling")
        upload_tile_dir = PROCESSED_TILES_DIR / UPLOADED_DATASET_ID
        tile_summary = tile_slide_from_mask(
            slide_path=raw_path,
            mask_path=mask_result["mask_path"],
            output_dir=str(upload_tile_dir),
            min_tissue_percent=MIN_TISSUE_PERCENT,
            slide_id=slide_id,
        )
        _resolve_within(upload_tile_dir, Path(tile_summary["output_dir"]))

        if tile_summary.get("saved_tiles", 0) == 0:
            # package_slides_to_h5 raises RuntimeError when total_tiles == 0
            # (no per-slide fallback there — it's shared with the multi-slide
            # dataset path, where "every slide produced zero tiles" is a real
            # error worth stopping on). For a single upload, tissue below
            # MIN_TISSUE_PERCENT is an expected, non-fatal outcome — skip
            # packaging instead of letting that raise get caught below and
            # reported as a packaging failure when nothing was actually wrong.
            _set_processing_status(
                slide_id, "done",
                error="Tiling found no tissue above the tissue threshold — nothing to package.",
            )
            return

        _set_processing_status(slide_id, "packaging")
        try:
            package_slides_to_h5(
                raw_paths=[raw_path],
                tile_dir=PROCESSED_TILES_DIR,
                # tile_dataset_name has no default and was missing entirely
                # here before — this call raised a bare TypeError on every
                # single-slide upload, always caught by the except below and
                # reported as a generic packaging failure. Matches the
                # tile_slide_from_mask output_dir above, which is where
                # tiles are actually written.
                tile_dataset_name=UPLOADED_DATASET_ID,
                output_root=HPL_DATASETS_ROOT,
                dataset_name=slide_id,
                # Same reason as the slide_id passed to run_tissue_detection
                # / tile_slide_from_mask above — package_slides_to_h5 would
                # otherwise derive the wrong slide_id from raw_path itself
                # (there's no parseable one embedded in it anymore) and look
                # for tiles under a slide_id nothing was actually written to.
                slide_ids=[slide_id],
            )
            _set_processing_status(slide_id, "done")
        except Exception as e:
            # Tiles are real and usable either way — a packaging failure
            # shouldn't be reported as if masking/tiling itself failed.
            _set_processing_status(
                slide_id, "done", error=f"Tiling succeeded, but .h5 packaging failed: {e}"
            )
    except Exception as e:
        _set_processing_status(slide_id, "error", error=str(e))


# Dataset-wide Slurm job submission and status


def _list_dataset_roots() -> list[str]:
    """Direct subdirectories of LONG_TERM_SCRATCH that are valid dataset inputs.

    Excludes the pipeline's own output/working directories so they can never
    be picked as "a dataset to tile" — recomputed fresh each call from the
    same constants those directories are actually built from, so it can't
    drift out of sync if those env vars change.
    """
    excluded = {TISSUE_MASK_DIR.name, PROCESSED_TILES_DIR.name, UPLOAD_ROOT.name}
    if not LONG_TERM_SCRATCH.is_dir():
        return []
    return sorted(
        p.name for p in LONG_TERM_SCRATCH.iterdir()
        if p.is_dir() and p.name not in excluded and not p.name.startswith(".")
    )


# Shapes sacct emits in its JobID column, all of which have to be told apart:
#
#   12345              a plain, non-array job
#   12345_7            one task of an array job
#   12345_[5-100]      a *pending* array range — one row standing for many
#                      tasks, optionally "%throttle" or a comma list
#   12345.batch        a job step; duplicates its parent's accounting and must
#                      never be counted
#
# The previous single pattern (^\d+_(\d+)\|(\S+)$) matched only the second of
# these. It silently dropped pending array ranges — so tiling could look
# finished while tasks were still queued — and, because "(\S+)$" cannot span a
# space, it also dropped every "CANCELLED by <uid>" row, hiding cancelled
# tasks from the packaging guard that is supposed to block on them. Steps were
# excluded only by accident, since "12345.batch" happens not to match; that is
# now explicit, because the patterns below deliberately accept bare job IDs.
_SACCT_ARRAY_TASK_RE = re.compile(r"^\d+_\d+$")
_SACCT_ARRAY_RANGE_RE = re.compile(r"^\d+_\[(.+?)\]$")
_SACCT_PLAIN_JOB_RE = re.compile(r"^\d+$")


def _array_range_size(spec: str) -> int:
    """How many array tasks a pending-range JobID stands for.

    "5-100" -> 96, "5,7,9" -> 3, "5-10%2" -> 6 (the %N concurrency throttle
    is not part of the task set). Counting the real span rather than treating
    the row as a single task keeps the state totals meaningful — a run with
    9,000 tasks still queued should not report one PENDING.
    """
    spec = spec.split("%", 1)[0]
    total = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        low, dash, high = part.partition("-")
        if dash:
            try:
                total += int(high) - int(low) + 1
                continue
            except ValueError:
                pass
        total += 1
    return max(total, 1)


def _normalise_slurm_state(state: str) -> str:
    """Bare state name, dropping any trailing detail sacct appends.

    The one that matters is "CANCELLED by 1234" — sacct records who cancelled
    a job, and callers compare against plain state names, so the suffix has to
    come off for a cancelled task to be recognised as cancelled at all.
    """
    state = state.strip()
    return state.split()[0] if state else ""

# squeue exits non-zero when asked about a job ID it has no record of. That's
# a real answer ("not live"), not a failure, and has to be told apart from an
# actual problem — see _slurm_jobs_live_states.
_SQUEUE_UNKNOWN_JOB_RE = re.compile(r"invalid job id|invalid user", re.I)


def _run_slurm(cmd: list[str], timeout: int) -> subprocess.CompletedProcess | None:
    """Run a read-only Slurm query. Returns None if the result can't be trusted.

    Every one of these calls used to read result.stdout without ever looking
    at returncode. A failing sacct/squeue writes its error to stderr and
    leaves stdout empty — byte-identical to "ran fine, nothing matched" — and
    callers assign those opposite meanings. "Nothing matched" is specifically
    what _job_output_ready reads as "this job aged out of the accounting
    retention window, so it finished long ago and an existing output file can
    be trusted", so a controller hiccup or an auth failure could be silently
    promoted into evidence that a job succeeded. Distinguishing them is the
    whole job of this wrapper.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[slurm] {cmd[0]} unavailable: {e}")
        return None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:200]
        print(f"[slurm] {' '.join(cmd)} exited {result.returncode}: {detail}")
        return None
    return result


def _slurm_jobs_live_states(job_ids: list[str]) -> list[str] | None:
    """States squeue reports for these job IDs right now, straight from
    the live scheduler queue — not sacct's accounting database.

    sacct goes through slurmdbd, which syncs on its own schedule and can
    lag behind a fresh sbatch submission by anywhere from seconds to
    longer; a job that was just submitted can legitimately have zero sacct
    rows yet even though it's sitting right there in the queue. squeue has
    no such lag, so it's the only reliable way to tell "sacct just hasn't
    caught up" apart from "this job genuinely isn't live" (finished long
    ago, aged out of retention). Returns None if squeue itself couldn't be
    reached (missing/timed out) — genuinely unknown, don't guess. Returns
    [] if squeue ran fine and simply has no rows for these job IDs (not
    currently queued or running, by any name).
    """
    if not job_ids:
        return []
    cmd = ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%T"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[{','.join(job_ids)}] squeue unavailable: {e}")
        return None
    if result.returncode != 0:
        # Not handled by _run_slurm, because this one call has a non-zero exit
        # that is a legitimate answer rather than a failure: squeue rejects a
        # job ID it has no record of with "Invalid job id specified", which is
        # precisely the "not currently live" result this function exists to
        # report. Everything else (controller unreachable, auth) is genuinely
        # unknown and must stay None so callers don't mistake it for proof.
        if _SQUEUE_UNKNOWN_JOB_RE.search(result.stderr or ""):
            # Parse stdout anyway rather than returning [] outright. With a
            # list of IDs, squeue reports the unknown ones on stderr and still
            # prints rows for the valid ones, exiting non-zero for the whole
            # call. Returning [] here therefore threw away live rows whenever
            # a single ID in the batch had aged out of the controller —
            # claiming "nothing is running" while tiling was demonstrably
            # still going. Empty stdout still yields [], the intended answer
            # when every ID really is unknown.
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        print(f"[{','.join(job_ids)}] squeue exited {result.returncode}: "
              f"{(result.stderr or '').strip()[:200]}")
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _slurm_controller_known_jobs(job_ids: list[str]) -> list[str] | None:
    """The subset of job_ids that slurmctld still holds a record of.

    This exists because --dependency is resolved by the controller and
    nothing else. sacct reads slurmdbd, whose retention is days to weeks;
    the controller forgets a job MinJobAge seconds after it ends (default
    300). So sacct's view and sbatch's view of "does this job exist" diverge
    within minutes of a run finishing, and a dependency on a job only sacct
    remembers is rejected outright with "Job dependency problem". Asking
    scontrol is asking the same component sbatch is about to ask, which is
    the only view that actually predicts whether the submission succeeds.

    Queried one ID at a time rather than as a list, because the answer we
    need is per-ID: a single unknown ID among live ones is exactly the mixed
    case worth resolving precisely, and a combined query collapses it into
    one pass/fail. That costs one local RPC per batch (~15 for a large
    dataset), which only happens on an explicit submit.

    A non-zero exit is a real answer here, not a failure — scontrol rejects
    an ID it has no record of with "Invalid job id specified", which is
    precisely the "controller has forgotten this" result being asked for.
    Same distinction _slurm_jobs_live_states draws, and the same regex.
    Returns None if scontrol itself is unusable, so callers can tell "the
    controller does not know these jobs" from "we could not ask" — only the
    former is grounds for dropping a dependency.
    """
    known: list[str] = []
    for job_id in job_ids:
        cmd = ["scontrol", "show", "job", job_id]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[{job_id}] scontrol unavailable: {e}")
            return None
        if result.returncode == 0:
            known.append(job_id)
            continue
        if _SQUEUE_UNKNOWN_JOB_RE.search(result.stderr or ""):
            continue
        print(f"[{job_id}] scontrol exited {result.returncode}: "
              f"{(result.stderr or '').strip()[:200]}")
        return None
    return known


def _get_slurm_array_state_counts(job_ids: list[str]) -> dict[str, int] | None:
    """Aggregate task-state counts across one or more (possibly array)
    Slurm jobs, via a single combined sacct call instead of one call per
    job. A large dataset split into ~15 batches used to call sacct once
    per batch sequentially here — each querying a big array job's full
    task list — which was slow enough (SLURM controller + accounting DB
    load, thousands of task rows per call) to blow past the UI's own
    30s HTTP read timeout while a run was still actively tiling.

    Returns None if sacct itself couldn't be reached (missing binary or
    timed out) — genuinely unknown state, the caller should NOT treat this
    the same as an empty {} result. {} means BOTH sacct ran fine and
    returned zero rows for these job IDs AND squeue confirms none of them
    are currently live — for an old-enough run that combination means it's
    aged out of Slurm's accounting-DB retention window (commonly a few
    days), i.e. long finished, not "still pending." sacct alone returning
    nothing isn't enough to conclude that on its own (see
    _slurm_jobs_live_states) — conflating "aged out" with "just submitted,
    not indexed yet" used to either leave old runs permanently stuck
    showing "tiling in progress," or (worse) make a job submitted moments
    ago look instantly "complete."
    """
    if not job_ids:
        return {}
    result = _run_slurm(
        ["sacct", "-j", ",".join(job_ids), "--format=JobID,State", "--parsable2", "--noheader"],
        timeout=45,
    )
    if result is None:
        return None

    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        job_field, separator, state_field = line.strip().partition("|")
        if not separator:
            continue
        job_field = job_field.strip()
        if "." in job_field:
            # A job step ("12345.batch", "12345_5.extern"). These repeat their
            # parent's state, so counting them would inflate every total.
            continue
        state = _normalise_slurm_state(state_field)
        if not state:
            continue
        if _SACCT_ARRAY_TASK_RE.match(job_field) or _SACCT_PLAIN_JOB_RE.match(job_field):
            counts[state] = counts.get(state, 0) + 1
            continue
        pending_range = _SACCT_ARRAY_RANGE_RE.match(job_field)
        if pending_range:
            counts[state] = counts.get(state, 0) + _array_range_size(pending_range.group(1))
    if counts:
        if any(state in IN_FLIGHT_SLURM_STATES for state in counts):
            # sacct says some tasks are still queued/running. Check that
            # against the live queue before believing it: slurmdbd lags, and a
            # task that died (TIMEOUT, OOM, node failure) keeps its last
            # RUNNING row until accounting catches up. tiling_complete is
            # computed as "no in-flight state present", so one stale row kept
            # the whole run pinned at "tiling in progress" — which in turn
            # left the packaging step blocked, hiding its Full-dataset /
            # subset options entirely.
            #
            # An empty squeue result is proof of absence here (not merely a
            # missing answer): _slurm_jobs_live_states returns None when it
            # could not ask, and only [] when the scheduler answered and holds
            # none of these IDs. In that case drop the stale in-flight rows.
            # Dropping rather than guessing an outcome is deliberate — the
            # caller's fallback for thin/empty counts is to read each slide's
            # _tiling_summary.json off disk, which is ground truth about what
            # actually finished, and far better evidence than a state we would
            # otherwise have to invent.
            live_states = _slurm_jobs_live_states(job_ids)
            if live_states == []:
                stale = {s: n for s, n in counts.items() if s in IN_FLIGHT_SLURM_STATES}
                print(
                    f"[{','.join(job_ids)}] sacct reports {stale} but squeue holds none of "
                    f"these jobs — treating the accounting rows as stale."
                )
                counts = {s: n for s, n in counts.items() if s not in IN_FLIGHT_SLURM_STATES}
        return counts

    live_states = _slurm_jobs_live_states(job_ids)
    if live_states is None:
        return None
    for state in live_states:
        counts[state] = counts.get(state, 0) + 1
    return counts


def _slurm_states_by_job(
    job_ids: list[str], *, timeout: int = 45
) -> dict[str, set[str]] | None:
    """Every state seen per job ID, from ONE sacct call.

    _get_slurm_job_state answers for a single job and _get_slurm_array_state_counts
    aggregates across jobs while discarding which job each state came from.
    Neither can label a *list* of runs: the first needs one call per run (ten
    sequential sacct calls to draw a ten-row list, on a controller already slow
    enough to have blown the UI's 30s read timeout mid-run), and the second
    cannot tell them apart afterwards.

    Keyed by base job ID, so array tasks ("123_5") and job steps ("123.batch")
    both fold into "123" — a caller asking "how is run X doing" wants the whole
    array's states together, not one row per task.

    Returns None if sacct could not be reached at all: genuinely unknown, which
    callers must not render as "finished". An empty set for a job ID means sacct
    ran and had nothing for it — aged out of retention, or too fresh to have
    been written yet, and those two are only distinguishable via squeue.

    timeout is a parameter because "one call" does not bound the work: sacct's
    cost scales with the *tasks* behind the IDs, not the IDs themselves, and
    tiling records one array job per ~1000-slide batch. A whole-history listing
    reached 321 IDs standing for tens of thousands of tasks, which took longer
    than the 45s default every single time and therefore returned None — the
    entire UI reading "can't reach Slurm" while Slurm was perfectly healthy.
    Callers spanning many runs must pass a smaller timeout and a smaller batch
    (see _listing_job_states) rather than inheriting a default sized for one.
    """
    if not job_ids:
        return {}
    result = _run_slurm(
        ["sacct", "-j", ",".join(job_ids), "--format=JobID,State",
         "--parsable2", "--noheader"],
        timeout=timeout,
    )
    if result is None:
        return None

    states: dict[str, set[str]] = {job_id: set() for job_id in job_ids}
    for line in result.stdout.splitlines():
        job_field, separator, state_field = line.strip().partition("|")
        if not separator:
            continue
        job_field = job_field.strip()
        # "123.batch"/"123_5.extern" repeat their parent's state.
        job_field = job_field.split(".", 1)[0]
        base = job_field.split("_", 1)[0]
        state = _normalise_slurm_state(state_field)
        if not state:
            continue
        if base in states:
            states[base].add(state)
    return states


# Bounds for the dataset listing's Slurm lookup. The listing is polled and spans
# every run ever recorded, so it cannot afford the per-run treatment: it asks
# about recent jobs precisely and lets older ones be answered by the queue plus
# whatever is on disk.
#
# WINDOW_DAYS is Slurm's accounting retention as configured here. Asking sacct
# about a job older than that is not merely wasted work — it returns nothing, so
# the answer is identical to not having asked, at the price of the slowest part
# of the call. Set it *shorter* than the real retention rather than longer; the
# cost of being wrong in that direction is one extra squeue row, and in the
# other direction it is a job whose failure we never notice.
_LISTING_SACCT_WINDOW_DAYS = 10
_LISTING_SACCT_CHUNK = 40
_LISTING_SACCT_CHUNK_TIMEOUT = 12
_LISTING_SACCT_BUDGET_S = 24.0
_LISTING_SQUEUE_CHUNK = 100


def _squeue_states_by_job(job_ids: list[str]) -> dict[str, set[str]] | None:
    """Live queue state per base job ID — squeue only, no accounting DB.

    _slurm_jobs_live_states answers the same question but discards which job
    each state belonged to, which is fine for one run and useless for a listing
    of fifty. Chunked because the ID list runs to the hundreds here and a single
    argument that long is worth avoiding regardless of what the shell tolerates.

    A chunk that fails is skipped rather than fatal: squeue exits non-zero for a
    batch containing any ID the controller has already forgotten (see
    _slurm_jobs_live_states), which for a listing of historic runs is the normal
    case, not an error. Returns None only if *every* chunk failed, i.e. squeue
    itself is unreachable.
    """
    if not job_ids:
        return {}

    states: dict[str, set[str]] = {job_id: set() for job_id in job_ids}
    any_answered = False
    for start in range(0, len(job_ids), _LISTING_SQUEUE_CHUNK):
        chunk = job_ids[start:start + _LISTING_SQUEUE_CHUNK]
        try:
            result = subprocess.run(
                ["squeue", "-j", ",".join(chunk), "-h", "-o", "%i|%T"],
                capture_output=True, text=True, timeout=15,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[listing] squeue unavailable: {e}")
            continue
        if result.returncode != 0 and not _SQUEUE_UNKNOWN_JOB_RE.search(result.stderr or ""):
            print(f"[listing] squeue exited {result.returncode}: "
                  f"{(result.stderr or '').strip()[:200]}")
            continue
        any_answered = True
        for line in result.stdout.splitlines():
            job_field, separator, state_field = line.strip().partition("|")
            if not separator:
                continue
            base = job_field.strip().split(".", 1)[0].split("_", 1)[0]
            state = _normalise_slurm_state(state_field)
            if state and base in states:
                states[base].add(state)
    return states if any_answered else None


def _listing_job_states(
    datasets: list[dict],
) -> tuple[dict[str, set[str]] | None, bool]:
    """Job states for every run in a dataset listing, on a time budget.

    Returns (states, complete). complete is False when some jobs were left to
    squeue alone — the answer is still usable, but a recent job that *failed*
    can read as merely "no longer queued", so the caller should say so rather
    than present it as the last word. The per-run /dataset-jobs/{id}/status
    endpoint remains the precise view, and is where the UI sends anyone who
    opens a single run.

    Two sources, deliberately:

      * squeue for every ID, because it is cheap at any list length (controller
        memory, no accounting DB) and it alone can say "this is running right
        now" — the one thing a polled listing must never get wrong.
      * sacct only for jobs from recent runs, in bounded chunks, because it is
        the expensive one and is the only way to tell COMPLETED from FAILED.

    An ID that neither source reports gets an empty set, which coarse_run_state
    reads as "no record" — aged out of retention, i.e. long finished. That is
    the same inference _get_slurm_array_state_counts already documents, and it
    is only sound because squeue was asked: without it, "nothing came back"
    would equally describe a job submitted ten seconds ago.
    """
    recent_ids: list[str] = []
    every_id: list[str] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=_LISTING_SACCT_WINDOW_DAYS)

    for dataset in datasets:
        for run in dataset["runs"]:
            ids: list[str] = []
            for field in ("job_id", "h5_job_id", "extraction_job_id", "test_h5_job_id"):
                ids.extend(_split_job_ids(run.get(field)))
            if not ids:
                continue
            every_id.extend(ids)
            submitted = _parse_timestamp(run.get("submitted_at"))
            # Unparseable timestamps count as recent: a row we cannot date is
            # more safely treated as one whose outcome still matters than as
            # one old enough to assume finished.
            if submitted is None or submitted >= cutoff:
                recent_ids.extend(ids)

    every_id = sorted(set(every_id))
    recent_ids = sorted(set(recent_ids))
    if not every_id:
        return {}, True

    live = _squeue_states_by_job(every_id)

    accounted: dict[str, set[str]] = {}
    deadline = time.monotonic() + _LISTING_SACCT_BUDGET_S
    complete = True
    sacct_reached = False
    for start in range(0, len(recent_ids), _LISTING_SACCT_CHUNK):
        chunk = recent_ids[start:start + _LISTING_SACCT_CHUNK]
        if time.monotonic() >= deadline:
            print(f"[listing] sacct budget spent; {len(recent_ids) - start} recent "
                  f"job ids left to squeue alone")
            complete = False
            break
        chunk_states = _slurm_states_by_job(
            chunk, timeout=_LISTING_SACCT_CHUNK_TIMEOUT
        )
        if chunk_states is None:
            complete = False
            continue
        sacct_reached = True
        accounted.update(chunk_states)

    if live is None and not sacct_reached:
        return None, False
    if len(recent_ids) < len(every_id):
        # Older jobs were never asked about. Honest, but not the whole story.
        complete = False

    states = {job_id: set(live.get(job_id, set()) if live else set()) for job_id in every_id}
    for job_id, seen in accounted.items():
        states.setdefault(job_id, set()).update(seen)
    return states, complete


def _parse_timestamp(value) -> datetime | None:
    """A tz-aware datetime from whatever the runs table hands back.

    Values arrive as ISO strings via pandas' to_json (which renders naive
    timestamps with a trailing Z) or as datetimes when read directly. A naive
    value is read as UTC, matching how submitted_at is written.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# _coarse_run_state is imported from dataset_rollup rather than defined here:
# the dataset rollup has to classify the same job states this server does, and
# two copies of that ordering would drift the moment one of them learned about
# a new Slurm state.


def _get_slurm_job_state(job_id: str) -> str | None:
    """Single (non-array) job's current Slurm state, e.g. for the h5-packaging
    job. squeue is asked first, sacct only as a fallback.

    That order is the whole point of this function and it used to be the other
    way round. sacct reads slurmdbd, which syncs on its own schedule; squeue
    reads slurmctld, which *is* the scheduler. Whenever the two disagree,
    squeue is right and sacct is merely stale, so consulting sacct first meant
    the UI reported a lagging accounting record in preference to the live
    queue — in both directions:

      * a job that had already died (TIMEOUT, OOM, a crash like the h5 file
        lock failure) kept its last sacct row of RUNNING, so the pipeline
        stepper showed "Running (Slurm state: RUNNING)" and hid the retry
        button, sometimes for minutes after the job was gone;
      * a job submitted moments ago has no sacct row at all yet, which the
        old code only rescued via the fallback below.

    Asking the queue first collapses both cases: if slurmctld still holds the
    job, its state is authoritative and current, full stop. squeue is also the
    cheaper of the two (no accounting DB), so this is not a cost for the
    common "is it still running?" poll.

    Whatever squeue reports is returned as-is rather than assumed to be a
    live state — a job that finished very recently can still appear in the
    queue as COMPLETED/FAILED, and that is a perfectly good terminal answer.

    Return contract is unchanged. None means neither source could be reached
    (genuinely unknown — callers must not guess). "" means squeue confirms the
    job is not in the queue AND sacct has no record of it, which for an
    old-enough job means it aged out of the accounting-DB retention window,
    i.e. long finished (see _job_output_ready).
    """
    live_states = _slurm_jobs_live_states([job_id])
    if live_states:
        return _normalise_slurm_state(live_states[0])

    result = _run_slurm(
        ["sacct", "-j", job_id, "--format=JobID,State", "--parsable2", "--noheader"],
        timeout=15,
    )
    if result is None:
        # squeue said "not in the queue" but sacct can't say how it ended.
        # Not-in-queue is not by itself an outcome, and "" would be read as
        # "aged out, trust the output file" — so stay honestly unknown.
        return None

    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        # Normalised for the same reason as the array counts above: an
        # un-normalised "CANCELLED by 1234" matches neither "COMPLETED" nor
        # any entry in IN_FLIGHT_SLURM_STATES, so it happened to be treated as
        # failed — right answer, but by accident, and it surfaced the raw uid
        # in user-facing messages.
        if len(parts) == 2 and parts[0] == job_id:
            return _normalise_slurm_state(parts[1])

    # sacct ran and has no row for this job. Only call that "aged out" if
    # squeue actually answered; if squeue itself was unreachable (None) we
    # have two non-answers, not evidence.
    if live_states is None:
        return None
    return ""


def _find_job_id_by_name(job_name: str) -> str | None:
    """Most recent Slurm job ID currently known under this exact job name,
    or None if nothing was found (or Slurm couldn't be reached).

    Used only as a reconciliation check before submitting a *new*
    packaging/extraction job for a run whose DB row has no job_id on
    record. That "no job_id" state is ambiguous — it either means nothing
    was ever attempted, or it means sbatch already ran and the server
    crashed/restarted in the narrow window between that call returning and
    the follow-up _update_dataset_run() call persisting the id. job_name is
    scoped to this one submission (see start_packaging_job /
    start_feature_extraction_job) specifically so this lookup can tell the
    two cases apart instead of risking a duplicate sbatch.

    Unlike _get_slurm_job_state, callers here don't need to distinguish
    "unreachable" from "genuinely not found" — either way the safe, honest
    fallback is "don't know of one, go ahead and submit as normal."
    squeue is tried first (covers anything still queued/running with no
    accounting-DB lag), sacct as a fallback for a job that already
    finished before this check ran.
    """
    result = _run_slurm(["squeue", "-n", job_name, "-h", "-o", "%A"], timeout=15)
    if result is not None:
        job_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if job_ids:
            return job_ids[-1]

    result = _run_slurm(
        ["sacct", "--name", job_name, "--format=JobID,State", "--parsable2", "--noheader"],
        timeout=15,
    )
    if result is None:
        return None

    # sacct also emits job-step rows ("12345.batch", "12345.extern") for
    # the same job — only the bare-digit parent JobID identifies the actual
    # sbatch submission we'd want to reuse.
    job_ids = [
        parts[0].strip()
        for line in result.stdout.splitlines()
        if (parts := line.split("|", 1)) and parts[0].strip().isdigit()
    ]
    return job_ids[-1] if job_ids else None


def _find_job_ids_by_name_prefix(prefix: str) -> list[str]:
    """Every Slurm job ID (live or recent) whose job name is exactly
    `prefix` or starts with `prefix + "_"` — the latter to catch a
    dataset-wide tiling run split into multiple array batches, each
    suffixed "_1", "_2", ... by _submit_one_array's job_name.

    Used the same way _find_job_id_by_name is for packaging/extraction,
    but for the dataset-wide tiling submission (see _run_dataset_submission)
    — that one's job_name is only known as a prefix ahead of time, since a
    large dataset's actual batch count isn't known until submit_dataset_array
    has already discovered and split the slide list.

    Scoped to the current OS user (via -u) rather than every job on the
    cluster, both to keep squeue/sacct fast on a busy shared controller and
    because this server only ever submits jobs as this one user in the
    first place — matches every other Slurm submission in this file.
    Returns [] if nothing matches or Slurm couldn't be reached; callers
    already treat "found nothing" and "couldn't check" the same way (fall
    back to whatever state the DB already has), so this doesn't need to
    distinguish them the way _get_slurm_job_state does.
    """
    user = getpass.getuser()
    job_ids: set[str] = set()

    def _matches(name: str) -> bool:
        return name == prefix or name.startswith(prefix + "_")

    result = _run_slurm(["squeue", "-u", user, "-h", "-o", "%A|%j"], timeout=15)
    if result is not None:
        for line in result.stdout.splitlines():
            parts = line.split("|", 1)
            if len(parts) == 2 and _matches(parts[1].strip()):
                job_ids.add(parts[0].strip())

    result = _run_slurm(
        ["sacct", "-u", user, "--format=JobID,JobName", "--parsable2", "--noheader"],
        timeout=15,
    )
    if result is not None:
        for line in result.stdout.splitlines():
            parts = line.split("|", 1)
            # Job-step rows ("12345.batch") share their parent's job name —
            # only the bare-digit parent JobID is a real sbatch submission.
            if len(parts) == 2 and parts[0].strip().isdigit() and _matches(parts[1].strip()):
                job_ids.add(parts[0].strip())

    return sorted(job_ids)


def _attempt_signature(*parts) -> str:
    """Short deterministic fingerprint of a test-run's actual parameters.

    /package-test and /extract-features-test are deliberately not written
    to slurm_dataset_runs (see their docstrings — a test attempt succeeding
    or failing has no bearing on the real run), so unlike the tracked
    endpoints there's no DB row to check "is one already in flight?"
    against. This fingerprint stands in for that: it's used both to scope
    the test job's own Slurm job name (so _find_job_id_by_name can look up
    "was *this exact* test already submitted?" directly against Slurm
    itself, no DB needed) and its output filename (so two genuinely
    *different* test requests — e.g. different sample_size — land on
    different output paths instead of silently racing on the same file,
    which used to be the only thing this endpoint's docstring assumed away
    "no state to guard against").
    """
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


_HPL_H5_DATASETS = ("img", "samples", "slides", "tiles")


def _validate_h5(path: Path) -> tuple[bool, str]:
    """Confirm a packaged .h5 is a complete, readable dataset — not merely a
    file sitting at the right path.

    Readiness used to be inferred from existence plus a Slurm state, which
    can't detect a file that is present and non-empty but unusable: an .h5
    truncated after its header opens without complaint and only fails when
    the missing chunks are read, which previously happened for the first time
    inside feature extraction, hours into a GPU job. Everything checked here
    is cheap (metadata plus two row reads) and runs on a human-triggered
    request, not a hot path.

    Returns (ok, reason) so callers can tell the user *why* it was rejected
    rather than just refusing.
    """
    try:
        with h5py.File(path, "r") as f:
            absent = [name for name in _HPL_H5_DATASETS if name not in f]
            if absent:
                return False, f"missing dataset(s) {absent}"

            rows = f["img"].shape[0]
            if rows == 0:
                return False, "contains zero tiles"

            mismatched = {
                name: f[name].shape[0]
                for name in _HPL_H5_DATASETS
                if f[name].shape[0] != rows
            }
            if mismatched:
                return False, f"dataset lengths disagree with img={rows}: {mismatched}"

            # Actually touch the first and last row. HDF5 validates the
            # superblock on open, so a file truncated partway through the data
            # still opens cleanly — reading the final row is what forces the
            # missing chunk to be resolved, and is the cheapest check that
            # distinguishes "complete" from "cut short".
            f["img"][0]
            f["img"][rows - 1]
            f["slides"][rows - 1]

    except (OSError, KeyError, ValueError) as e:
        return False, f"unreadable HDF5: {e}"
    return True, ""


def _h5_has_legacy_tile_names(path: Path) -> bool:
    """Whether this .h5 stores tile names without the ".jpeg" suffix.

    Deliberately *not* part of _validate_h5. The file is perfectly usable — the
    images are right, and feature extraction and cluster assignment both read it
    without caring what the name column says. The suffix only matters where the
    name becomes a join key, which is the Knowledge Bank load, and
    migrate_tile_names.py fixes it there in one command.

    Treating it as invalid blocked packaging, extraction and assignment for a
    defect none of them are affected by. So it is reported as an advisory the UI
    can show, and the refusal lives at the one boundary where a wrong key
    produces a wrong result.
    """
    try:
        with h5py.File(path, "r") as f:
            if "tiles" not in f:
                return False
            rows = f["tiles"].shape[0]
            return bool(rows) and tiles_missing_suffix(f["tiles"][: min(rows, 100)])
    except (OSError, KeyError, ValueError):
        return False


def _extraction_expected_rows(row: dict) -> int | None:
    """Tiles in the .h5 this run's extraction was given, or None if that can't
    be established.

    None means "cannot check", and the row-count comparison is skipped rather
    than failed — a run whose packaged input has since been moved or deleted
    should not have a previously-good extraction start reading as incomplete.
    The narrower checks in validate_extraction_output still apply.
    """
    packaged = row.get("h5_output_path")
    if not packaged:
        return None
    packaged_path = Path(packaged)
    if not packaged_path.is_file():
        return None
    return _packaged_h5_rows(packaged_path)


def _job_output_ready(
    output_path: Path | None,
    slurm_state: str | None,
    validator: Callable[[Path], tuple[bool, str]] | None = None,
) -> bool:
    """Whether a Slurm job's output file (packaging's .h5, extraction's
    features file) is safe to treat as finished and readable.

    slurm_state == "COMPLETED" (a live sacct record) is the normal case.
    slurm_state == "" means neither sacct nor squeue (see
    _slurm_jobs_live_states) has any record of this job — for an old-enough
    job that means it's aged out of Slurm's accounting-DB retention window
    (long finished), not that it's still running or was just submitted a
    moment ago (squeue would have caught that). A non-empty file is safe to
    trust in that case too — without this, an old completed run would show
    "waiting on tiling to finish" and an eligible-for-retry prompt forever,
    since sacct can no longer vouch for it either way. slurm_state is None
    (sacct/squeue themselves failed to run) is left as not-ready — that's
    genuinely ambiguous, not a case to guess through.

    `validator` (pass _validate_h5 for packaging output) gets the final say: a
    job Slurm reports as COMPLETED can still have left a file nothing can
    read, and nothing downstream should be told it's ready until something has
    actually opened it.
    """
    if not output_path or not output_path.is_file():
        return False

    # slurm_state is None means Slurm itself could not be reached. That is
    # normally not enough to call an output ready — but it must not be an
    # automatic "no" either, or an unreachable sacct makes a *finished* run
    # look interrupted and puts a Retry button in front of a perfectly good
    # output. For packaging that retry would resubmit over a complete .h5 and
    # throw away hours of work, which is a far worse outcome than the
    # over-cautious "not ready" was ever protecting against.
    #
    # There is independent, on-disk evidence available in exactly that case:
    # packaging writes to a ".partial" sibling and os.replace()s it into place
    # only after a successful run, so the real path existing at all already
    # means a run finished, and the absence of a leftover ".partial" means no
    # other attempt is mid-write. Combined with a validator that opens the file
    # and reads its first and last row, that is strictly stronger proof than
    # the sacct row we could not fetch. Requiring a validator keeps this narrow:
    # callers with no way to check their output's integrity (no validator) get
    # the old conservative answer.
    partial_sibling = output_path.with_name(output_path.name + ".partial")
    unverifiable_but_complete = (
        slurm_state is None
        and validator is not None
        and not partial_sibling.exists()
    )

    if not unverifiable_but_complete and slurm_state != "COMPLETED" and not (
        slurm_state == "" and output_path.stat().st_size > 0
    ):
        return False
    if validator is not None:
        ok, reason = validator(output_path)
        if not ok:
            print(f"[{output_path}] output rejected as not ready: {reason}")
            return False
    if unverifiable_but_complete:
        print(
            f"[{output_path}] Slurm unreachable, but the output validates and no "
            f".partial is present — treating it as complete."
        )
    return True


def _load_heatmap_probs():
    
    global _heatmap_probs
    try:
        eng = _get_engine()
        df = pd.read_sql("SELECT * FROM tile_hpc_heatmap", eng)
        df.columns = df.columns.astype(str).str.strip()
        if "slide_tile" in df.columns:
            df["slide_tile"] = df["slide_tile"].astype(str).str.strip().str.upper()
        keep = ["slide_tile"] + [c for c in df.columns if c.startswith("p_hpc_")]
        _heatmap_probs = df[keep].copy()
    except Exception as e:
        print(f"Heatmap load failed: {e}")
        _heatmap_probs = None





def _open_slide(slide_id: str) -> openslide.OpenSlide:
    slide_id = slide_id.strip().upper()
    if slide_id in _wsi_handles:
        return _wsi_handles[slide_id]
    hpc_path = _wsi_map.get(slide_id)
    if not hpc_path:
        raise HTTPException(404, f"Slide {slide_id} not in wsi_registry")
    if not os.path.isfile(hpc_path):
        raise HTTPException(404, f"SVS file not found on disk: {hpc_path}")
    slide = openslide.OpenSlide(hpc_path)
    _wsi_handles[slide_id] = slide
    return slide


def _get_deepzoom(slide_id: str) -> DeepZoomGenerator:
    slide_id = slide_id.strip().upper()

    if slide_id in _dz_handles:
        return _dz_handles[slide_id]

    slide = _open_slide(slide_id)

    dz = DeepZoomGenerator(
        slide,
        tile_size=256,
        overlap=1,
        limit_bounds=False,
    )

    _dz_handles[slide_id] = dz
    return dz

def _img_to_jpeg_bytes(img, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf.getvalue()


def _jpeg_response(data: bytes) -> StreamingResponse:
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.dtype == np.uint8:
        return arr
    x = arr.astype(np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if 0.0 <= mn and mx <= 1.0:
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    lo = float(np.nanpercentile(x, 1))
    hi = float(np.nanpercentile(x, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(x, 0, 255).astype(np.uint8)
    x = (x - lo) / (hi - lo) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)



# Grid / adjacency helpers 


def _grid_xy(x_native, y_native):
    gx = int(float(x_native) // float(TILE_SIZE_NATIVE))
    gy = int(float(y_native) // float(TILE_SIZE_NATIVE))
    return gx, gy


def _neighbors_8(gx, gy):
    return [
        (gx - 1, gy - 1), (gx, gy - 1), (gx + 1, gy - 1),
        (gx - 1, gy),                     (gx + 1, gy),
        (gx - 1, gy + 1), (gx, gy + 1), (gx + 1, gy + 1),
    ]


def _compute_adjacency(df_slide: pd.DataFrame):
    df2 = df_slide.copy()
    df2["hpc_id"] = pd.to_numeric(df2["hpc_id"], errors="coerce")
    df2 = df2.dropna(subset=["hpc_id", "x_native", "y_native"])
    df2["hpc_id"] = df2["hpc_id"].astype(int)
    id_col = "slide_tile" if "slide_tile" in df2.columns else "tiles"

    pos_to_row = {}
    for _, r in df2.iterrows():
        gx, gy = _grid_xy(r["x_native"], r["y_native"])
        if (gx, gy) not in pos_to_row:
            pos_to_row[(gx, gy)] = r

    pair_edge_counts: dict[tuple, int] = {}
    tile_has_neighbor_pair: dict[tuple, dict] = {}
    visited_edges = set()

    for (gx, gy), r in pos_to_row.items():
        a = int(r["hpc_id"])
        tile_a = str(r[id_col])
        for nb in _neighbors_8(gx, gy):
            r2 = pos_to_row.get(nb)
            if r2 is None:
                continue
            b = int(r2["hpc_id"])
            if a == b:
                continue
            tile_b = str(r2[id_col])
            p = (a, b) if a < b else (b, a)
            edge_key = tuple(sorted([(gx, gy), nb]))
            if (p, edge_key) in visited_edges:
                continue
            visited_edges.add((p, edge_key))
            pair_edge_counts[p] = pair_edge_counts.get(p, 0) + 1
            if p not in tile_has_neighbor_pair:
                tile_has_neighbor_pair[p] = {"a_touch": set(), "b_touch": set()}
            if a < b:
                tile_has_neighbor_pair[p]["a_touch"].add(tile_a)
                tile_has_neighbor_pair[p]["b_touch"].add(tile_b)
            else:
                tile_has_neighbor_pair[p]["a_touch"].add(tile_b)
                tile_has_neighbor_pair[p]["b_touch"].add(tile_a)

    # Convert sets → lists for JSON serialisation
    serialisable = {}
    for pair_key, sets in tile_has_neighbor_pair.items():
        k = f"{pair_key[0]}_{pair_key[1]}"
        serialisable[k] = {
            "a_touch": sorted(sets["a_touch"]),
            "b_touch": sorted(sets["b_touch"]),
        }
    pair_counts = {f"{a}_{b}": cnt for (a, b), cnt in pair_edge_counts.items()}
    return pair_counts, serialisable


# App lifecycle


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LOADED TILE SERVER WITH DZI SUPPORT:", __file__)
    _load_wsi_map()
    _load_heatmap_probs()
    yield
    global _h5_handle
    if _h5_handle:
        _h5_handle.close()
    _dz_handles.clear()
    for s in _wsi_handles.values():
        s.close()
    _wsi_handles.clear()

app = FastAPI(title="HPC Tile Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = TileCache(cache_dir=CACHE_DIR)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "slides_loaded": len(_wsi_map)}


@app.get("/debug/routes")
def debug_routes():
    return {
        "file": __file__,
        "routes": sorted([getattr(route, "path", str(route)) for route in app.routes]),
    }


@app.get("/slides")
def list_slides():
    return {"slides": sorted(_wsi_map.keys())}

@app.post("/upload-slide")
async def upload_slide(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    slide_id: str = Form(None),
    confirm_overwrite: bool = Form(False),
):
    original_filename = Path(file.filename or "uploaded_slide").name
    suffix = Path(original_filename).suffix.lower()

    allowed_suffixes = {".svs", ".ndpi", ".tif", ".tiff", ".isyntax"}

    if suffix not in allowed_suffixes:
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'. Allowed types: {sorted(allowed_suffixes)}",
        )

    safe_user_slide_id = (slide_id or "").strip().upper()

    if not safe_user_slide_id:
        safe_user_slide_id = f"UPLOAD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif not _DATASET_NAME_RE.match(safe_user_slide_id):
        # slide_id still becomes a literal filename component below (not the
        # directory itself — see internal_id further down) and a primary
        # key in wsi_registry, and it's used as-is by mask/tile output paths
        # for this upload (tissue_masks/UPLOADED/<slide_id>_..., etc.) — a
        # value containing "/" or ".." would let those land outside their
        # intended directories. Same charset _sanitize_dataset_name enforces
        # for dataset folder names below, for the same reason. Belt-and-
        # suspenders: _resolve_within also asserts this below rather than
        # relying on this check alone.
        raise HTTPException(
            400,
            f"Invalid slide_id '{safe_user_slide_id}': use letters, numbers, '.', '_', or '-' "
            "only, and don't start with one of those.",
        )

    # A slide_id colliding with an existing registry entry from outside this
    # ad-hoc-upload pool (e.g. a real curated TCGA slide) would otherwise let
    # any upload silently repoint that ID's hpc_path at whatever file was just
    # uploaded — every existing viewer/pipeline call for that ID would then
    # serve the wrong slide, with nothing in the response signaling that a
    # collision (not a fresh registration) just happened. Not overridable by
    # confirm_overwrite — this is never the intended "replace my own test
    # upload" case, so there's no confirmation that makes it safe to proceed.
    eng = _get_engine()
    with eng.connect() as conn:
        existing = conn.execute(
            text("SELECT dataset_id FROM wsi_registry WHERE slide_id = :slide_id"),
            {"slide_id": safe_user_slide_id},
        ).fetchone()
    if existing and existing[0] != UPLOADED_DATASET_ID:
        raise HTTPException(
            409,
            {
                "error": "slide_id_conflict",
                "slide_id": safe_user_slide_id,
                "existing_dataset_id": existing[0],
                "message": (
                    f"slide_id '{safe_user_slide_id}' is already registered under dataset "
                    f"'{existing[0]}' — choose a different slide_id instead of overwriting it."
                ),
            },
        )

    # Re-using an ID from a previous ad-hoc upload is allowed (DO UPDATE in
    # _register_uploaded_slide) — that's the intended "replace my own test
    # upload" path — but it silently overwrote that upload's file, tissue
    # mask, tiles, and packaged .h5 with no warning. Now a hard stop unless
    # the caller already confirmed: first attempt gets a structured 409 the
    # UI can render as a warning with an explicit "yes, overwrite" action,
    # rather than the overwrite just happening.
    if existing and not confirm_overwrite:
        raise HTTPException(
            409,
            {
                "error": "slide_id_exists",
                "slide_id": safe_user_slide_id,
                "message": (
                    f"slide_id '{safe_user_slide_id}' already exists from a previous upload. "
                    "Uploading again will overwrite its saved file, tissue mask, tiles, and "
                    "packaged .h5. Resubmit with confirm_overwrite=true to proceed."
                ),
            },
        )

    internal_id = str(uuid.uuid4())
    safe_filename = original_filename.replace(" ", "_")

    UPLOAD_RAW_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Storage is keyed by internal_id — server-generated, never user
    # input — not by safe_user_slide_id. Charset validation above already
    # makes slide_id safe to use as a path segment, but the actual
    # collision-safety/traversal guarantee here comes from a UUID directory
    # neither the user nor a future bug in that validation can influence;
    # slide_id is embedded only in the filename *inside* that directory, so
    # anyone browsing UPLOAD_RAW_DIR can still immediately tell which slide
    # a given upload is — it's just no longer what determines *where* the
    # file lands. _resolve_within is the explicit backstop: even given all
    # of the above, refuse rather than silently writing outside
    # UPLOAD_RAW_DIR if something upstream is ever wrong.
    try:
        save_dir = _resolve_within(UPLOAD_RAW_DIR, UPLOAD_RAW_DIR / internal_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = _resolve_within(save_dir, save_dir / f"{safe_user_slide_id}_{safe_filename}")
    except PathEscapeError as e:
        raise HTTPException(500, f"Refusing to save upload: {e}")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Failed to save uploaded file: {e}")
    finally:
        await file.close()

    validation = {
        "openslide_readable": False,
        "error": None,
    }

    slide_info_payload = None
    metadata_path = None
    status = "uploaded"
    try:
        # Context manager, not a bare OpenSlide(...) + a .close() call at
        # the end of the block — several things between open and close can
        # legitimately raise (metadata_dir.mkdir()/json.dump() on a full
        # disk, _resolve_within() on a PathEscapeError, even a malformed
        # slide's own property/dimension reads), and a bare .close() placed
        # after all of that is simply never reached if any of it throws —
        # the outer except below still catches the error and returns
        # normally, but the OpenSlide handle (an open fd plus, for a large
        # WSI, a substantial mmap'd region) leaks until GC eventually gets
        # to it, if it ever does. __exit__ runs regardless of how the block
        # exits, so this can't leak the same way.
        with openslide.OpenSlide(str(save_path)) as slide:
            slide_info_payload = {
                "slide_id": safe_user_slide_id,
                "filename": original_filename,
                "stored_path": str(save_path),
                "level_count": slide.level_count,
                "level_dimensions": [
                    {"width": int(w), "height": int(h)}
                    for w, h in slide.level_dimensions
                ],
                "mpp_x": slide.properties.get("openslide.mpp-x"),
                "mpp_y": slide.properties.get("openslide.mpp-y"),
                "vendor": slide.properties.get("openslide.vendor"),
                "objective_power": slide.properties.get("openslide.objective-power"),
                "uploaded_at": datetime.now().isoformat(),
            }

            # No thumbnail generated here — tile_mask.py's masking step (the
            # very next thing the background pipeline does) already produces
            # one as a byproduct of computing the tissue mask
            # (tissue_masks/UPLOADED/{slide_id}_thumbnail.png), and that step
            # starts essentially immediately after this request returns.
            # Generating a second one here would just duplicate it for a few
            # seconds' head start that isn't worth the redundancy.
            # Same internal_id-keyed layout as save_path above, for the same
            # reason — slide_id stays readable in the filename, but isn't what
            # determines the path.
            metadata_dir = _resolve_within(UPLOAD_METADATA_DIR, UPLOAD_METADATA_DIR / internal_id)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = _resolve_within(metadata_dir, metadata_dir / f"{safe_user_slide_id}.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(slide_info_payload, f, indent=2)

        _register_uploaded_slide(safe_user_slide_id, str(save_path))

        validation["openslide_readable"] = True

        _set_processing_status(safe_user_slide_id, "queued")
        background_tasks.add_task(_run_postupload_pipeline, safe_user_slide_id, str(save_path))

    except Exception as e:
        validation["error"] = str(e)
        status = "viewer_ready"
    return {
        "internal_id": internal_id,
        "slide_id": safe_user_slide_id,
        "filename": original_filename,
        "stored_path": str(save_path),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "status": status,
        "processing_status": _get_processing_status(safe_user_slide_id),
        "validation": validation,
        "slide_info": slide_info_payload,
        "next_step": "Tissue masking + tiling started in the background — poll /slide/{slide_id}/processing-status.",
    }


@app.get("/slide/{slide_id}/processing-status")
def slide_processing_status(slide_id: str):
    return _get_processing_status(slide_id)


# Every submit_array() argument that changes what the tiles themselves look
# like — as opposed to how the job is scheduled (cpus, memory, partition,
# batch_size) or which slides are picked (sample_size, slide_names). These are
# what has to be identical between an original run and any resume of it, since
# a dataset half-tiled at one threshold and half at another is not one dataset.
#
# jpeg_quality is included: it only affects the on-disk JPEGs and not the .h5's
# uncompressed pixels, but it does change the image data those pixels are
# decoded from, so a resume at a different quality is still a mixed dataset.
_TILING_PARAM_NAMES = (
    "min_tissue",
    "target_mpp",
    "target_tile_px",
    "level",
    "jpeg_quality",
    "mask_max_size",
    "mask_saturation",
    "mask_value",
)


def _default_tiling_params() -> dict:
    """Current defaults, read off submit_array()'s own signature.

    Introspection rather than a hardcoded copy specifically so this cannot
    drift from the function it feeds: changing a default in
    submit_mask_tile_slurm.py updates what gets recorded here automatically,
    and a renamed or removed parameter fails loudly at import instead of
    silently recording a value nothing uses.
    """
    signature = inspect.signature(submit_dataset_array)
    defaults = {}
    for name in _TILING_PARAM_NAMES:
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"submit_array() no longer has a defaulted '{name}' parameter — "
                f"_TILING_PARAM_NAMES needs updating."
            )
        defaults[name] = parameter.default
    return defaults


def _resolve_tiling_params(req: "DatasetJobRequest") -> dict:
    """The tiling parameters this submission will actually run with.

    Precedence: an explicit tiling_params block (how a resume passes the
    original run's recorded values through) over the defaults, and an
    explicitly-set min_tissue over both — min_tissue predates tiling_params as
    a top-level request field and the UI still sends it that way.

    model_fields_set is what makes that last part work: min_tissue has a
    default on the model, so its presence in the request is the only way to
    tell "the user chose 30.0" from "the user said nothing and 30.0 is the
    default". Without that distinction a resume could not avoid overriding the
    recorded value with a default that merely looks deliberate.
    """
    params = _default_tiling_params()
    if req.tiling_params:
        params.update(
            {k: v for k, v in req.tiling_params.items() if k in _TILING_PARAM_NAMES}
        )
    if "min_tissue" in req.model_fields_set:
        params["min_tissue"] = req.min_tissue
    return params


def _row_tiling_params(row) -> dict | None:
    """Tiling parameters recorded for a run, or None if it predates the column.

    None is returned rather than the defaults so callers can tell "this run
    used these values" from "nobody knows what this run used" — only the former
    is grounds for claiming a resume reproduces the original.
    """
    try:
        recorded = row["tiling_params"]
    except (KeyError, IndexError):
        # Server pointed at a database without the migration applied.
        return None
    if not recorded:
        return None
    if isinstance(recorded, str):
        # psycopg2 without the JSONB adapter registered hands back raw text.
        try:
            recorded = json.loads(recorded)
        except json.JSONDecodeError:
            return None
    if not isinstance(recorded, dict):
        return None
    return {k: v for k, v in recorded.items() if k in _TILING_PARAM_NAMES} or None


class DatasetJobRequest(BaseModel):
    dataset_path: str
    max_concurrent: int = 10
    min_tissue: float = 30.0
    # Set by resume_dataset_job (and the UI's full-directory run) to reproduce
    # an earlier run's tiling exactly. Unset on a fresh submission, which then
    # takes the current defaults plus whatever min_tissue the user chose.
    tiling_params: Optional[dict] = None
    sample_size: Optional[int] = None
    slide_names: Optional[list[str]] = None
    partition: Optional[str] = None  # None -> Slurm's own default partition
    notify_email: Optional[str] = None  # Slurm's own END/FAIL notification, no attachments
    # Folder tiles land in under PROCESSED_TILES_DIR, e.g. "TCGA" or
    # "Radiogenomics" — lets the UI reuse an existing dataset folder or name
    # a new one. None falls back to dataset_path's own folder name.
    dataset_name: Optional[str] = None


@app.get("/dataset-roots")
def list_dataset_roots():
    """Top-level directories currently under LONG_TERM_SCRATCH, for reference
    only — POST /dataset-jobs accepts any path (including nested ones), it
    doesn't require picking from this list.
    """
    return {"root": str(LONG_TERM_SCRATCH), "datasets": _list_dataset_roots()}


@app.get("/tile-dataset-names")
def list_tile_dataset_names():
    """Existing folders directly under PROCESSED_TILES_DIR, e.g. ["TCGA",
    "Radiogenomics"] — lets the UI offer "add to an existing dataset folder"
    as a dropdown instead of everyone retyping the name by hand and risking
    a near-miss (e.g. "Radiogenomic" splitting off a second folder).
    """
    if not PROCESSED_TILES_DIR.is_dir():
        return {"dataset_names": []}
    return {
        "dataset_names": sorted(
            p.name for p in PROCESSED_TILES_DIR.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    }


_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sanitize_dataset_name(name: str) -> str:
    """Validate a user-supplied dataset folder name.

    This becomes a literal path segment under PROCESSED_TILES_DIR
    (tile_dir/<name>/<slide_id>/...), so it's restricted to a safe charset
    rather than trusted as-is — a name like "../../etc" or containing "/"
    would otherwise let a submission write tiles outside PROCESSED_TILES_DIR.
    """
    name = name.strip()
    if not name:
        raise ValueError("Dataset folder name cannot be empty.")
    if not _DATASET_NAME_RE.match(name):
        raise ValueError(
            f"Invalid dataset folder name '{name}': use letters, numbers, "
            "'.', '_', or '-' only, and don't start with one of those."
        )
    return name


class PathEscapeError(RuntimeError):
    """A resolved path landed outside the directory it was supposed to be
    confined to. Should be unreachable in practice — every caller of
    _resolve_within already validates its inputs before building the path
    (charset-checked slide_id/dataset_name, or a server-generated UUID) —
    but this is the structural backstop for when that validation has a bug,
    gets skipped, or a future code path forgets it. Whoever catches this
    should treat it as "refuse and report," never "strip and continue."
    """


def _resolve_within(base: Path, path: Path) -> Path:
    """Resolve `path` and assert it's actually inside `base`.

    This is the last line of defense, not the primary one — charset
    validation (_sanitize_dataset_name, the slide_id check in
    /upload-slide) is what should actually stop a "../" or absolute-path
    value from ever reaching here. This exists for the case that
    validation misses something: even if a bad value somehow got this far,
    the write/read still can't land outside where it's supposed to.
    """
    resolved_base = base.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_base and resolved_base not in resolved_path.parents:
        raise PathEscapeError(
            f"Resolved path {resolved_path} escapes permitted directory {resolved_base}"
        )
    return resolved_path


def _resolve_dataset_path(user_path: str) -> Path:
    """Resolve a user-supplied absolute path.

    Not restricted to any particular root (by request) — accepts any
    location the server process can read. Still rejects empty input and
    pipeline-owned output directories, since submitting those as "a
    dataset" is always a mistake regardless of where they live. Submitting
    this path triggers a real sbatch job, so treat this input as trusted —
    anyone with UI access can point it at any directory the server can see.
    """
    user_path = user_path.strip()
    if not user_path:
        raise ValueError("Dataset path cannot be empty.")

    expanded = Path(user_path).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"Dataset path must be absolute (start with '/'): got '{user_path}'. "
            "A relative path would silently resolve against the server's own "
            "working directory, not where you meant."
        )

    candidate = expanded.resolve()

    excluded_dirs = {TISSUE_MASK_DIR.resolve(), PROCESSED_TILES_DIR.resolve(), UPLOAD_ROOT.resolve()}
    for excluded in excluded_dirs:
        if candidate == excluded or excluded in candidate.parents:
            raise ValueError(
                f"'{user_path}' is a pipeline-owned output directory, not a dataset input."
            )

    if not candidate.is_dir():
        raise ValueError(f"Directory not found: {candidate}")

    return candidate


def _record_run_job(
    submission_id: str,
    stage: str,
    job_id: str | None,
    output_path: str | None = None,
    params: dict | None = None,
) -> None:
    """Append one submitted Slurm job to a run's history.

    Purely additive alongside the single-slot columns on slurm_dataset_runs,
    which stay authoritative for gating (see migrate_dataset_run_jobs.sql).

    Never raises. Every caller is on the far side of a successful sbatch, so a
    bookkeeping failure must not become a 500 — that would tell the caller
    nothing was queued while the job runs anyway, which is worse than a gap in
    the history. ON CONFLICT DO NOTHING makes it safe to call again for a job
    already recorded, which the tiling path does after recovering job ids by
    name.
    """
    if not job_id:
        return
    try:
        eng = _get_engine()
        with eng.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO slurm_dataset_run_jobs
                        (submission_id, stage, job_id, output_path, params)
                    VALUES (:submission_id, :stage, :job_id, :output_path, :params)
                    ON CONFLICT (submission_id, stage, job_id) DO NOTHING
                """),
                {
                    "submission_id": submission_id,
                    "stage": stage,
                    "job_id": str(job_id),
                    "output_path": str(output_path) if output_path else None,
                    # Serialised here rather than relying on the driver's dict
                    # adaptation, which differs between psycopg2 and psycopg3 —
                    # same reason tiling_params is written this way.
                    "params": json.dumps(params) if params is not None else None,
                },
            )
    except Exception as e:
        print(f"[warn] could not record {stage} job {job_id} for {submission_id}: {e}")


def _run_job_history(submission_id: str) -> list[dict]:
    """Every recorded Slurm job for a run, newest first, with live state.

    States for the whole history come from one sacct call via
    _slurm_states_by_job — a per-row lookup would be one call per attempt, and
    this is rendered inside a run's panel where several attempts are normal.
    """
    try:
        eng = _get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT stage, job_id, output_path, params, submitted_at
                    FROM slurm_dataset_run_jobs
                    WHERE submission_id = :submission_id
                    ORDER BY submitted_at DESC, id DESC
                """),
                {"submission_id": submission_id},
            ).mappings().fetchall()
    except Exception as e:
        # The table may not exist yet if the code is deployed before the
        # migration is run. An empty history is the right degradation; failing
        # the whole status response is not.
        print(f"[warn] could not read job history for {submission_id}: {e}")
        return []

    every_id: list[str] = []
    for row in rows:
        every_id.extend(j for j in str(row["job_id"]).split(",") if j)
    states = _slurm_states_by_job(sorted(set(every_id)))

    history = []
    for row in rows:
        ids = [j for j in str(row["job_id"]).split(",") if j]
        if states is None:
            state = "unknown"
        else:
            seen: set[str] = set()
            for job_id in ids:
                seen |= states.get(job_id.split("_", 1)[0], set())
            state = _coarse_run_state(seen)
        params = row["params"]
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = None
        history.append({
            "stage": row["stage"],
            "job_id": row["job_id"],
            "batch_count": len(ids),
            "output_path": row["output_path"],
            "params": params,
            "submitted_at": row["submitted_at"].isoformat() if row["submitted_at"] else None,
            "slurm_state": state,
        })
    return history


def _update_dataset_run(submission_id: str, **fields):
    eng = _get_engine()
    set_clause = ", ".join(f"{k} = :{k}" for k in fields)
    with eng.begin() as conn:
        conn.execute(
            text(f"UPDATE slurm_dataset_runs SET {set_clause} WHERE submission_id = :submission_id"),
            {**fields, "submission_id": submission_id},
        )


def _effective_h5_dataset_name(base_name: str, is_subset: bool) -> str:
    """Full-dataset runs keep the plain name; subset runs (random sample or
    specific slides) get a distinct "_subset_N" suffix so a validation run
    doesn't silently overwrite a previous run's .h5 at the same path —
    h5py.File(path, 'w') always clobbers whatever's already there.

    Numbered by scanning what's already on disk under HPL_DATASETS_ROOT
    rather than querying the DB, since the filesystem is the actual source
    of truth for "would this collide with something that already exists."
    """
    if not is_subset:
        return base_name

    existing = 0
    if HPL_DATASETS_ROOT.is_dir():
        for entry in HPL_DATASETS_ROOT.iterdir():
            match = re.match(rf"^{re.escape(base_name)}_subset_(\d+)$", entry.name)
            if match:
                existing = max(existing, int(match.group(1)))
    return f"{base_name}_subset_{existing + 1}"


def _run_dataset_submission(submission_id: str, raw_dir: str, req: DatasetJobRequest):
    """Background job: discover every slide under raw_dir (can be slow on a
    large or network-mounted dataset) and submit the Slurm array, updating
    slurm_dataset_runs as it progresses. Runs after /dataset-jobs already
    returned submission_id, so the HTTP request isn't held open for however
    long the directory walk takes.
    """
    try:
        _update_dataset_run(submission_id, status="discovering")

        def _persist_plan(plan: dict):
            # Runs the instant submit_dataset_array has written the combined
            # manifest, before any sbatch call — i.e. before the part of that
            # call that can take minutes on a large dataset split into
            # several batches (each retrying through Slurm controller
            # congestion, see _run_sbatch_with_retry). Persisting the
            # manifest/slide-count now, not only after submit_dataset_array
            # fully returns, means a crash mid-submission still leaves this
            # row with everything dataset_job_status() needs except job_id
            # — which the job-name lookup below can recover directly from
            # Slurm — instead of the run looking permanently lost with no
            # manifest to even resume from.
            _update_dataset_run(
                submission_id,
                status="submitting",
                manifest_path=plan["manifest_path"],
                total_slides=plan["slides_found"],
            )

        result = submit_dataset_array(
            raw_dir=Path(raw_dir),
            mask_dir=TISSUE_MASK_DIR,
            tile_dir=PROCESSED_TILES_DIR,
            max_concurrent=req.max_concurrent,
            # Every tile-affecting parameter, from the same resolved dict that
            # was written to the row — including min_tissue, which used to be
            # passed on its own. Passing them as a unit is what guarantees a
            # resume runs the original values: there is no second code path
            # here that could quietly reintroduce a default.
            **(req.tiling_params or _default_tiling_params()),
            sample_size=req.sample_size,
            slide_names=req.slide_names,
            partition=req.partition,
            notify_email=req.notify_email,
            dataset_name=req.dataset_name,
            # Scoped to this submission so _find_job_ids_by_name_prefix can
            # recover it (and any per-batch job under this name, e.g.
            # "..._1", "..._2") if this background task never gets to write
            # job_id itself.
            job_name=f"wsi_mask_tile_{submission_id}",
            on_planned=_persist_plan,
        )
        job_ids = result.get("job_ids") or []
        failed_batch_count = result.get("failed_batch_count", 0)
        if job_ids:
            # Stored comma-joined in the existing job_id column — one batch
            # (the common case) looks exactly as it always did; multiple
            # batches (a large dataset split to avoid overwhelming Slurm's
            # controller with one giant array) list all of them. A batch
            # can fail even after its own retries without aborting the rest
            # — surface that here as a warning-style note, not a hard error,
            # since whatever did submit is still real, running work.
            partial_failure_note = (
                f"{failed_batch_count} of {result['batch_count']} batches failed to submit "
                f"after retries — {len(job_ids)} succeeded and are running; missing slides "
                f"would need a follow-up submission."
                if failed_batch_count else None
            )
            _update_dataset_run(
                submission_id,
                status="submitted",
                job_id=",".join(job_ids),
                manifest_path=result["manifest_path"],
                total_slides=result["slides_found"],
                error=partial_failure_note,
            )
            _record_run_job(
                submission_id, "tiling", ",".join(job_ids),
                output_path=result["manifest_path"],
                params={
                    "slides": result["slides_found"],
                    "batches": len(job_ids),
                    "failed_batches": failed_batch_count,
                    "tiling_params": req.tiling_params,
                },
            )
            # Packaging (and later, feature extraction) no longer auto-chain
            # from here — each stage now needs an explicit "start" click from
            # the UI once the previous stage is confirmed done. See
            # POST /dataset-jobs/{id}/package and .../extract-features.
        else:
            batch_errors = "; ".join(
                b.get("error") or b.get("sbatch_stdout") or "no job id"
                for b in result.get("batches", [])
            )
            _update_dataset_run(
                submission_id, status="error",
                error=f"sbatch did not return any job IDs. {batch_errors}".strip(),
            )
    except Exception as e:
        _update_dataset_run(submission_id, status="error", error=str(e))


def _record_resume_lineage(submission_id: str, parent_submission_id: str) -> None:
    """Note that this run was created by resuming another.

    A separate best-effort UPDATE rather than a column in the INSERT, and it
    never raises, for the same reason _record_run_job doesn't: this is
    bookkeeping on the far side of a decision that has already been made. If
    migrate_dataset_runs_lineage.sql has not been applied yet, folding this
    into the INSERT would make every resume fail outright — trading a missing
    display detail for a broken pipeline stage.

    GET /datasets does not depend on this. It groups runs by
    (raw_dir, dataset_name), which resume reuses, so lineage only sharpens the
    picture from "these runs share a directory" to "this one continued that
    one".
    """
    try:
        eng = _get_engine()
        with eng.begin() as conn:
            conn.execute(
                text(
                    "UPDATE slurm_dataset_runs "
                    "SET resumed_from_submission_id = :parent "
                    "WHERE submission_id = :submission_id"
                ),
                {"parent": parent_submission_id, "submission_id": submission_id},
            )
    except Exception as e:
        print(
            f"[warn] could not record that {submission_id} resumed "
            f"{parent_submission_id}: {e}"
        )


def _start_dataset_submission(
    raw_dir: Path,
    req: DatasetJobRequest,
    background_tasks: BackgroundTasks,
    resumed_from_submission_id: str | None = None,
) -> dict:
    """Create a new slurm_dataset_runs row and kick off the background
    pipeline for it. Shared by POST /dataset-jobs (a fresh submission) and
    the /resume endpoint (a follow-up submission for whatever a previous
    run's slides are still missing) — both are "start a submission for this
    raw_dir with this req," just with req.slide_names populated differently.

    resumed_from_submission_id is what tells those two apart afterwards. Both
    land in the same table looking identical, so without it a resume and a
    deliberate second run over the same directory are indistinguishable.
    """
    submission_id = str(uuid.uuid4())
    is_subset = bool(req.sample_size or req.slide_names)

    # Falls back to raw_dir's own folder name when the caller (or the UI's
    # "use default" state) didn't pick one — same default submit_array()
    # itself uses, kept here too so the resolved name gets stored on the row
    # and every later stage (resume, packaging, status) reads it back
    # instead of recomputing it and potentially drifting if raw_dir's own
    # name ever gets reused for a different dataset.
    try:
        dataset_name = _sanitize_dataset_name(req.dataset_name) if req.dataset_name else raw_dir.name
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Resolved once, here, then both persisted and handed to submit_array — so
    # the row records exactly what ran rather than a second, independently
    # computed guess at it. A resume reads these back and passes them straight
    # through, which is the whole point: nothing downstream re-derives them.
    tiling_params = _resolve_tiling_params(req)
    req = req.model_copy(
        update={"dataset_name": dataset_name, "tiling_params": tiling_params}
    )

    eng = _get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO slurm_dataset_runs
                    (submission_id, raw_dir, mask_dir, tile_dir, status,
                     is_subset, partition, notify_email, dataset_name,
                     tiling_params)
                VALUES
                    (:submission_id, :raw_dir, :mask_dir, :tile_dir, 'queued',
                     :is_subset, :partition, :notify_email, :dataset_name,
                     :tiling_params)
            """),
            {
                "submission_id": submission_id,
                "raw_dir": str(raw_dir),
                "mask_dir": str(TISSUE_MASK_DIR),
                "tile_dir": str(PROCESSED_TILES_DIR),
                "is_subset": is_subset,
                "partition": req.partition,
                "notify_email": req.notify_email,
                "dataset_name": dataset_name,
                # Serialised here rather than relying on the driver's dict
                # adaptation, which differs between psycopg2 and psycopg3.
                "tiling_params": json.dumps(tiling_params),
            },
        )

    # After the INSERT, so a database without the lineage migration still gets
    # a fully working run out of this function.
    if resumed_from_submission_id:
        _record_resume_lineage(submission_id, resumed_from_submission_id)

    background_tasks.add_task(_run_dataset_submission, submission_id, str(raw_dir), req)

    return {"submission_id": submission_id, "status": "queued", "raw_dir": str(raw_dir), "dataset_name": dataset_name}


@app.post("/dataset-jobs")
def create_dataset_job(req: DatasetJobRequest, background_tasks: BackgroundTasks):
    """Queue a dataset-wide masking+tiling Slurm array job.

    dataset_path is resolved and validated server-side via
    _resolve_dataset_path — never trust that the client only ever sends a
    safe path, since this ultimately triggers a real sbatch submission.

    Discovering slides and submitting to Slurm both happen in a background
    task, not here — a recursive walk of a large/network-mounted dataset can
    take far longer than a client is willing to hold an HTTP request open
    for. This returns immediately with a submission_id to poll instead.
    """
    try:
        raw_dir = _resolve_dataset_path(req.dataset_path)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return _start_dataset_submission(raw_dir, req, background_tasks)


@app.post("/dataset-jobs/{submission_id}/resume")
def resume_dataset_job(submission_id: str, background_tasks: BackgroundTasks):
    """Find whatever slides from a previous submission never got tiled
    (checked against the filesystem — a slide counts as done if it has a
    real _tile_metadata.csv, regardless of what Slurm's job records say)
    and queue a new submission for just those, reusing the same raw_dir.

    This is how "the whole dataset ended partway through" gets resolved
    without manually figuring out which Slurm batch failed — every run,
    however it stopped, can be resumed the same way.
    """
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM slurm_dataset_runs WHERE submission_id = :submission_id"),
            {"submission_id": submission_id},
        ).mappings().fetchone()

    if not row:
        raise HTTPException(404, f"No dataset job found with id {submission_id}")
    if not row["manifest_path"]:
        raise HTTPException(400, "This run never got far enough to have a manifest to resume from.")

    manifest_path = Path(row["manifest_path"])
    tile_dir = Path(row["tile_dir"])
    if not manifest_path.is_file():
        raise HTTPException(400, f"Manifest no longer exists on disk: {manifest_path}")

    dataset_name = _row_dataset_name(row)
    # Detailed variant so the response can distinguish slides that were never
    # attempted from ones whose tile metadata is corrupt (both get re-tiled)
    # and from ones that legitimately produced no tissue (which never should
    # be, since re-running yields the same empty result).
    breakdown = find_missing_slides_detailed(manifest_path, tile_dir, dataset_name)
    missing_raw_paths, total = breakdown["missing"], breakdown["total"]

    if not missing_raw_paths:
        # "Missing" here only means "never produced a _tile_metadata.csv at
        # all" — it does NOT mean every slide has real tiles. A slide that
        # ran and legitimately saved zero tiles (no tissue passed the
        # threshold) still writes that file, so it doesn't show up as
        # missing here, but it's not "done" in any meaningful sense either.
        # Say so explicitly instead of claiming they "have tiles," which is
        # false for that case and contradicts the zero-tile breakdown shown
        # elsewhere in the status view.
        return {
            "resumed": False,
            "message": (
                f"Nothing to resubmit — every one of the {total} slides in this run was "
                f"already attempted. (Some may have saved zero tiles; resubmitting won't "
                f"change that — see the zero-tile breakdown in the status view above.)"
            ),
        }

    missing_slide_ids = [slide_id_from_raw_path(p) for p in missing_raw_paths]

    # The parameters the original run actually tiled with, so the resumed
    # slides come out identical to the ones already on disk. Note min_tissue is
    # deliberately NOT passed as a top-level field: doing so would put it in
    # model_fields_set and let it override the recorded block (see
    # _resolve_tiling_params). None here — a run predating the tiling_params
    # column — falls back to current defaults, reported below so the caller
    # knows the resume is not a guaranteed reproduction.
    recorded_tiling_params = _row_tiling_params(row)

    resume_req = DatasetJobRequest(
        dataset_path=row["raw_dir"],
        slide_names=missing_slide_ids,
        partition=row["partition"],
        notify_email=row["notify_email"],
        tiling_params=recorded_tiling_params,
        # Re-run into the same folder tiles already live in, not whatever
        # raw_dir.name would resolve to by default — matters if the
        # original submission was given a custom dataset_name.
        dataset_name=dataset_name,
    )
    result = _start_dataset_submission(
        Path(row["raw_dir"]),
        resume_req,
        background_tasks,
        # Persisted, not just echoed back: this reply is the only place the
        # relationship existed before, so reloading the page lost it.
        resumed_from_submission_id=submission_id,
    )
    result.update({
        "resumed": True,
        "resumed_from_submission_id": submission_id,
        # What the resumed slides will be tiled with, and whether that is the
        # original run's own recorded settings or a fallback. The caller should
        # surface the fallback case: mixing thresholds within one dataset is
        # exactly the failure this is meant to prevent, and silently defaulting
        # would reintroduce it for every pre-migration run.
        "tiling_params": _resolve_tiling_params(resume_req),
        "tiling_params_source": "original_run" if recorded_tiling_params else "defaults",
        "missing_slide_count": len(missing_raw_paths),
        "never_attempted_count": len(breakdown["never_attempted"]),
        "corrupt_metadata_count": len(breakdown["corrupt"]),
        "zero_tile_count": len(breakdown["zero_tile"]),
        "total_in_original_manifest": total,
    })
    return result


def _row_test_packaging_params(row) -> dict | None:
    """What the recorded test packaging job actually sampled.

    Lets the UI say "the job below is from an earlier setup" after a reload has
    thrown away the form state it used to compare against — previously that
    warning only worked within a single session, which is the one case where the
    user could still remember what they had typed.
    """
    recorded = row.get("test_h5_params")
    if not recorded:
        return None
    if isinstance(recorded, str):
        try:
            recorded = json.loads(recorded)
        except json.JSONDecodeError:
            return None
    return recorded if isinstance(recorded, dict) else None


def _row_dataset_name(row) -> str:
    """The dataset folder this run's tiles live under. Rows created before
    the dataset_name column existed have it NULL — fall back to raw_dir's
    own folder name, which is what those runs actually used at the time.
    """
    return row["dataset_name"] or Path(row["raw_dir"]).name


def _get_dataset_run_row(submission_id: str) -> dict:
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM slurm_dataset_runs WHERE submission_id = :submission_id"),
            {"submission_id": submission_id},
        ).mappings().fetchone()
    if not row:
        raise HTTPException(404, f"No dataset job found with id {submission_id}")
    return dict(row)


def _tiled_coverage(
    raw_dir: Path, tile_dir: Path, dataset_name: str, *, strict: bool = False
) -> dict:
    """What is actually tiled on disk right now, for every slide in raw_dir.

    The run's manifest says what a submission set out to do; this says what
    exists. They diverge constantly and the difference is what the UI needs:
    a subset run's manifest lists 30 slides, but the dataset folder may hold
    tiles for all 400 because earlier runs (or a resume, or a hand-run job)
    filled it in. Deciding what can be packaged from the manifest alone means
    refusing to package tiles that are sitting right there.

    "Tiled" delegates to tiling_output_complete() — the same function
    submit_mask_tile_slurm.py's worker uses to decide whether to skip a slide —
    so the two cannot drift apart as that test is tightened.

    strict=False (the default) is the one deliberate difference. The full test
    also parses the tile-metadata CSV to confirm its row count matches the
    summary's saved_tiles, which costs a whole-file parse per slide: fine for
    the single slide a worker is deciding about, ruinous across 14,000 of them
    on cephfs while a user waits. The JSON checks (parseable summary,
    saved_tiles present and sane) are kept, since those files are small.
    #
    The asymmetry that leaves is worth being explicit about: a slide whose CSV
    stopped short of its summary counts as tiled here, and the worker would
    re-tile it. That is the safe direction — the worker has the final say and
    redoes the work — but it means this count can be marginally optimistic
    after an interrupted tiling run. strict=True removes that gap at the cost
    of reading every CSV, which is why it's opt-in and never used by anything
    polling: it's for the one moment someone wants to sign off on a dataset
    being finished before committing GPU hours to it.

    Cost is a stat plus a small JSON read per slide over a network filesystem.
    That's why this lives behind its own endpoint rather than in the 10s poll.
    """
    slide_dataset_dir = tile_dir / dataset_name
    tiled: list[str] = []
    untiled: list[str] = []
    for slide_path in discover_slides(raw_dir):
        slide_id = slide_id_from_raw_path(slide_path)
        slide_tile_dir = slide_dataset_dir / slide_id
        metadata = slide_tile_dir / f"{slide_id}_tile_metadata.csv"
        summary = slide_tile_dir / f"{slide_id}_tiling_summary.json"
        complete = tiling_output_complete(
            metadata, summary, slide_tile_dir, verify_row_count=strict
        )
        (tiled if complete else untiled).append(str(slide_path))
    return {
        "raw_dir": str(raw_dir),
        "dataset_name": dataset_name,
        "strict": strict,
        "slides_in_directory": len(tiled) + len(untiled),
        "slides_tiled": len(tiled),
        "slides_untiled": len(untiled),
        "tiled_paths": tiled,
        # Capped: this is for showing the user which slides still need work,
        # and a full list on a 14,000-slide directory is neither useful in the
        # UI nor cheap to ship on every poll.
        "untiled_sample": [Path(p).name for p in untiled[:50]],
    }


@app.get("/dataset-jobs/{submission_id}/jobs")
def dataset_job_history(submission_id: str):
    """Every Slurm job this run has submitted, across all stages, newest first.

    Separate from /status because it costs an sacct call and answers a different
    question: /status says what the run can do next, this says what it has
    already tried. Repackaging and repeated test packaging are invisible in
    /status by design — those fields hold only the latest attempt.
    """
    _get_dataset_run_row(submission_id)      # 404s for an unknown run
    return {"submission_id": submission_id, "jobs": _run_job_history(submission_id)}


@app.get("/dataset-jobs/{submission_id}/tiled-coverage")
def tiled_coverage(submission_id: str):
    """Live, on-disk answer to "how much of this directory is actually tiled?"

    Deliberately not folded into /status: it stats two files per slide over
    cephfs, and /status is polled every 10s by every open tab.
    """
    row = _get_dataset_run_row(submission_id)
    return _tiled_coverage(
        Path(row["raw_dir"]), Path(row["tile_dir"]), _row_dataset_name(row)
    )


def _runs_for_directory(raw_dir: Path, dataset_name: str) -> list[dict]:
    """Every recorded run that tiled into this (raw_dir, dataset_name) pair.

    Filtering on dataset_name in Python rather than SQL because
    _row_dataset_name() has to resolve a NULL column back to raw_dir's folder
    name for rows predating that column — a WHERE clause would silently drop
    exactly those older runs, which are the ones most likely to hold the
    tiles nobody remembers submitting.
    """
    eng = _get_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM slurm_dataset_runs WHERE raw_dir = :raw_dir "
                "ORDER BY submitted_at ASC"
            ),
            {"raw_dir": str(raw_dir)},
        ).mappings().fetchall()
    return [dict(row) for row in rows if _row_dataset_name(row) == dataset_name]


def _tiling_readiness(
    raw_dir: Path, tile_dir: Path, dataset_name: str, *, strict: bool = False
) -> dict:
    """One answer to "is tiling finished for this whole directory?", across
    every run that ever tiled into it.

    The per-run endpoints cannot answer this, and that is the point. A run's
    /status reports the run's own manifest — correct, but a 30-slide subset
    run reporting 30/30 reads as "the dataset is tiled" when 14,000 slides
    sit beside it untouched. Directories here get filled in by several runs
    plus resumes plus hand-run jobs, so the only trustworthy scope is the
    directory, and the only trustworthy authority is the disk.

    Disk decides what is *done*; Slurm decides what that means about what is
    *left*. Untiled slides with jobs still running is a wait; the same
    untiled slides with nothing running is a stall needing resubmission, and
    those two need to be distinguishable without reading sacct by hand.

    sacct being unreachable is reported as its own verdict rather than
    folded into either. _get_slurm_array_state_counts returns None for
    "genuinely unknown" and {} for "ran fine, nothing in flight" — collapsing
    those would let a controller outage read as "nothing is running, so this
    has stalled" and send someone off to resubmit work that is mid-flight.
    """
    coverage = _tiled_coverage(raw_dir, tile_dir, dataset_name, strict=strict)
    runs = _runs_for_directory(raw_dir, dataset_name)

    job_ids: list[str] = []
    for run in runs:
        job_ids.extend(j for j in (run["job_id"] or "").split(",") if j)
    job_ids = sorted(set(job_ids))

    state_counts = _get_slurm_array_state_counts(job_ids)
    slurm_known = state_counts is not None
    in_flight = bool(
        slurm_known and set(state_counts) & IN_FLIGHT_SLURM_STATES
    )

    untiled = coverage["slides_untiled"]
    total = coverage["slides_in_directory"]

    if not total:
        verdict = "no_slides"
        message = f"No supported WSI files found under {raw_dir}."
    elif not untiled:
        verdict = "complete"
        message = (
            f"All {total} slides in {raw_dir} have tiles on disk"
            f"{' (row counts verified)' if strict else ''}."
        )
    elif in_flight:
        verdict = "in_progress"
        message = (
            f"{coverage['slides_tiled']} of {total} slides tiled; "
            f"tiling jobs are still running."
        )
    elif not slurm_known:
        verdict = "unknown"
        message = (
            f"{coverage['slides_tiled']} of {total} slides tiled, but sacct "
            f"could not be reached — cannot confirm whether the remaining "
            f"{untiled} are still being worked on. Retry before resubmitting."
        )
    elif not runs:
        # No tracked run has ever targeted this directory, so there is nothing
        # to have stalled and nothing to resume. Kept distinct from "stalled"
        # because the two need opposite actions — submit a first run here,
        # versus resume an existing one — and because a brand-new dataset
        # being told it "needs resubmitting" is the kind of wrong-but-plausible
        # message that sends someone hunting for a run that never existed.
        verdict = "not_started"
        if coverage["slides_tiled"]:
            message = (
                f"{coverage['slides_tiled']} of {total} slides already have "
                f"tiles, but no tracked run targeted this directory — they came "
                f"from a hand-run job or a different raw_dir. Submit a run "
                f"(POST /dataset-jobs) to tile the remaining {untiled}."
            )
        else:
            message = (
                f"None of the {total} slides in {raw_dir} have been tiled, and "
                f"no run has been submitted for it. Submit one with "
                f"POST /dataset-jobs."
            )
    else:
        verdict = "stalled"
        # Name a real run to resume rather than a {submission_id} placeholder —
        # the most recent one, since _runs_for_directory orders oldest-first.
        message = (
            f"{coverage['slides_tiled']} of {total} slides tiled and no tiling "
            f"job is running — the remaining {untiled} need resubmitting "
            f"(POST /dataset-jobs/{runs[-1]['submission_id']}/resume)."
        )

    return {
        "raw_dir": str(raw_dir),
        "dataset_name": dataset_name,
        # The plain yes/no. Deliberately only true on full coverage, so it
        # cannot be satisfied by a subset run finishing its own manifest.
        "tiling_done": verdict == "complete",
        "verdict": verdict,
        "message": message,
        "strict": strict,
        "slides_in_directory": total,
        "slides_tiled": coverage["slides_tiled"],
        "slides_untiled": untiled,
        "untiled_sample": coverage["untiled_sample"],
        # tiled_paths is omitted: 14,000 absolute paths is a payload nobody
        # reading a verdict wants. /tiled-coverage still returns it for the
        # callers that package from it.
        "runs_checked": len(runs),
        "runs": [
            {
                "submission_id": run["submission_id"],
                "status": run["status"],
                "is_subset": run["is_subset"],
                "total_slides": run["total_slides"],
                "job_id": run["job_id"],
                "submitted_at": str(run["submitted_at"]) if run["submitted_at"] else None,
            }
            for run in runs
        ],
        "slurm_job_ids": job_ids,
        "slurm_state_counts": state_counts,
        "slurm_reachable": slurm_known,
    }


@app.get("/tiling-readiness")
def tiling_readiness(
    raw_dir: Optional[str] = Query(
        None, description="Directory of WSIs to check. Omit if passing submission_id."
    ),
    dataset_name: Optional[str] = Query(
        None, description="Tile folder under PROCESSED_TILES_DIR. Defaults to raw_dir's name."
    ),
    submission_id: Optional[str] = Query(
        None, description="Resolve raw_dir/dataset_name from an existing run instead."
    ),
    strict: bool = Query(
        False,
        description="Also verify each slide's tile-metadata row count against its "
                    "summary. Authoritative but reads every CSV — minutes on a "
                    "14,000-slide directory. Use before committing to GPU hours.",
    ),
):
    """Is tiling actually finished for a whole directory, across every run?

    Not scoped to one submission, unlike /dataset-jobs/{id}/tiled-coverage —
    pass submission_id only as a convenient way to name the directory, and
    the answer still covers everything in it.

    Same cost profile as /tiled-coverage (two files stat'd per slide over
    cephfs, more with strict=true), so this is a button, not a poll.
    """
    if submission_id:
        row = _get_dataset_run_row(submission_id)
        resolved_raw = Path(row["raw_dir"])
        resolved_tile = Path(row["tile_dir"])
        resolved_dataset = dataset_name or _row_dataset_name(row)
    elif raw_dir:
        # Through the same resolver POST /dataset-jobs uses, so the path this
        # looks up is byte-identical to what that endpoint stored. A bare
        # Path(raw_dir) matched nothing for "~/x", a trailing slash, or a
        # symlink — _runs_for_directory compares raw_dir as an exact string, so
        # every run would silently drop out and a fully-tiled directory could
        # report as never started.
        try:
            resolved_raw = _resolve_dataset_path(raw_dir)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # PROCESSED_TILES_DIR is this server's current default. A run records
        # its own tile_dir, so if one was submitted when that pointed
        # elsewhere, pass submission_id instead and the row decides.
        resolved_tile = PROCESSED_TILES_DIR
        resolved_dataset = dataset_name or resolved_raw.name
    else:
        raise HTTPException(400, "Provide raw_dir or submission_id.")

    return _tiling_readiness(
        resolved_raw, resolved_tile, resolved_dataset, strict=strict
    )


@app.post("/dataset-jobs/{submission_id}/package")
def start_packaging_job(
    submission_id: str,
    allow_incomplete: bool = Query(
        False,
        description="Package even though some slides failed tiling. Switches the Slurm "
                    "dependency from afterok to afterany and accepts a dataset with holes.",
    ),
    scope: str = Query(
        "run",
        description="'run' packages this run's own manifest (the default, unchanged). "
                    "'tiled' packages every slide in the raw directory that has tiles on "
                    "disk right now, regardless of which run produced them.",
    ),
    resume: Optional[bool] = Query(
        None,
        description="true continues a previous attempt's checkpoint (and fails if there "
                    "is none); false discards it and repackages from scratch. Omit only "
                    "for non-interactive callers — the UI always sends an explicit "
                    "choice, because silently continuing an earlier attempt is a "
                    "decision the user should be making.",
    ),
):
    """Manually start .h5 packaging for a run whose tiling has already been
    submitted. This used to auto-chain via Slurm's --dependency the moment
    tiling was submitted; now it only happens when the user clicks the
    "Start packaging" button in the UI, once they've confirmed tiling is
    actually done. The submitted packaging job still carries its own
    --dependency on every tiling batch job as a safety net, in case this gets
    clicked slightly before the last batch finishes.

    That dependency is afterok by default (see submit_packaging_job), so a run
    with failed tiling tasks is refused here with a 400 rather than submitted
    as a job Slurm could never start. Pass allow_incomplete=true to package
    anyway, knowing the resulting .h5 will be missing those slides.

    Not gated on row["status"] == "submitted" — cancelling a run only flips
    its status column, it doesn't touch the manifest or the tiles already
    written to disk, so packaging the whole thing is still valid (and
    useful) for a cancelled run whose tiling had already finished. job_id
    existing is what actually means tiling was submitted in the first place.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        if not row["job_id"]:
            raise HTTPException(400, "Tiling hasn't been submitted yet for this run.")

        tile_dataset_name = _row_dataset_name(row)

        manifest_path = Path(row["manifest_path"]) if row["manifest_path"] else None
        if not manifest_path:
            raise HTTPException(400, "This run never got far enough to have a manifest.")
        # Only scope="run" actually reads this file (below, for the sbatch job
        # and the pool-size count). scope="tiled" writes a brand new manifest
        # from live disk coverage a few lines down, using only this path's
        # parent directory — checking the original file exists here refused a
        # "package the whole directory" request over a manifest it was never
        # going to open, whenever the run's own manifest had since been
        # deleted or moved.
        if scope != "tiled" and not manifest_path.is_file():
            raise HTTPException(400, f"Manifest no longer exists on disk: {manifest_path}")
        job_ids = [j for j in row["job_id"].split(",") if j]

        # Scope is resolved up front, before the retry guard below, because it
        # decides *which output file* this submission is about. Resolving it
        # afterwards meant the guard always judged the run's recorded path: a
        # subset run whose own 30-slide .h5 had completed refused a scope="tiled"
        # request with "Packaging has already completed for this run", even
        # though the full-coverage .h5 it was actually asking for did not exist.
        if scope == "tiled":
            # Package what is on disk, not what this run set out to do. A
            # subset run's manifest is 30 slides even when the dataset folder
            # holds tiles for the whole directory — put there by an earlier
            # run, a resume, or a hand-run job. Packaging from the manifest
            # then ignores tiles sitting right there, and no amount of
            # re-running this run widens it, because its manifest was fixed at
            # submission time.
            coverage = _tiled_coverage(
                Path(row["raw_dir"]), Path(row["tile_dir"]), tile_dataset_name
            )
            if not coverage["slides_tiled"]:
                raise HTTPException(
                    400, f"No slide in {row['raw_dir']} has tiles on disk yet."
                )
            # The real-time completeness gate. Disk is the authority, not
            # sacct: a slide either has its tiles or it does not, whatever
            # Slurm remembers about the job that was meant to produce them.
            if coverage["slides_untiled"] and not allow_incomplete:
                raise HTTPException(
                    400,
                    {
                        "error": "tiling_incomplete",
                        "submission_id": submission_id,
                        "slides_tiled": coverage["slides_tiled"],
                        "slides_untiled": coverage["slides_untiled"],
                        "slides_in_directory": coverage["slides_in_directory"],
                        "untiled_sample": coverage["untiled_sample"],
                        "message": (
                            f"{coverage['slides_untiled']} of "
                            f"{coverage['slides_in_directory']} slides in this directory "
                            f"have no tiles on disk. Tile them first, or package the "
                            f"{coverage['slides_tiled']} that do."
                        ),
                    },
                )
            manifest_path = manifest_path.parent / (
                f"wsi_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_tiled_{tile_dataset_name}.txt"
            )
            write_manifest([Path(p) for p in coverage["tiled_paths"]], manifest_path)
            # Full coverage of the dataset folder is by definition not a
            # subset, so it takes the unsuffixed name and its own output
            # directory — deliberately leaving any "_subset_N" .h5 this run
            # already produced untouched instead of overwriting it.
            dataset_name = _effective_h5_dataset_name(tile_dataset_name, False)
            # Nothing to wait on: the tiles already exist. Keeping this run's
            # tiling job IDs as a dependency would only expose the submission
            # to slurmctld having forgotten them.
            job_ids = []

        # Only pick a *new* "_subset_N" the first time this run packages.
        # _effective_h5_dataset_name numbers by scanning HPL_DATASETS_ROOT, so
        # calling it again on a retry returns the *next* N — i.e. a different
        # output directory than the one this run already started writing to.
        # make_hpl_hdf5.py's resume checkpoint sidecars are keyed off that
        # output path, so a retry under a freshly-numbered name silently
        # abandoned the previous attempt's progress (potentially hours of
        # decoded tiles) and restarted from zero, and left the run's recorded
        # h5_output_path pointing at an orphaned directory. That's the exact
        # opposite of what retrying a timed-out packaging job should do.
        # Reading the name back off the recorded path keeps it stable across
        # retries — the same approach start_feature_extraction_job already
        # takes, for the same reason. Full (non-subset) runs were unaffected
        # either way, since their name never had a suffix to renumber.
        # scope="tiled" already chose its own name above and must not be
        # overridden by whatever this run last recorded.
        if scope != "tiled":
            recorded_name = (
                Path(row["h5_output_path"]).parent.name if row["h5_output_path"] else None
            )
            if recorded_name and not (
                # One case where the recorded name must NOT be reused: a subset
                # run that previously packaged with scope="tiled" has the
                # *unsuffixed* full-dataset directory on record. Reading it back
                # for a plain scope="run" retry would write this run's 30-slide
                # manifest over the full-coverage .h5. Fall through to the
                # computed subset name, which is where a run-scoped package
                # belongs.
                row["is_subset"] and recorded_name == tile_dataset_name
            ):
                dataset_name = recorded_name
            else:
                dataset_name = _effective_h5_dataset_name(
                    tile_dataset_name, bool(row["is_subset"])
                )
        # Scoped to this one submission so _find_job_id_by_name can recover
        # it below — must match the job_name passed to submit_packaging_job
        # further down.
        job_name = f"hpl_h5_package_{submission_id}"

        if not row["h5_job_id"]:
            # No h5_job_id on record is ambiguous, not proof nothing was
            # attempted: a prior call here could have had sbatch succeed and
            # then the server crash/restart before _update_dataset_run()
            # right below it ran, leaving Postgres unaware of a job that's
            # actually live (or already finished) in Slurm. Check by this
            # submission's own job name before trusting the DB's silence —
            # without this, that crash window turns into a duplicate sbatch
            # submission every time it's hit.
            recovered_job_id = _find_job_id_by_name(job_name)
            if recovered_job_id:
                recovered_output_path = str(hpl_h5_output_path(HPL_DATASETS_ROOT, dataset_name))
                _update_dataset_run(
                    submission_id, h5_job_id=recovered_job_id, h5_output_path=recovered_output_path,
                )
                row["h5_job_id"] = recovered_job_id
                row["h5_output_path"] = recovered_output_path

        if row["h5_job_id"]:
            # Same reasoning as feature extraction's retry guard below: only
            # block a new attempt if the prior one actually succeeded or is
            # still genuinely running — otherwise a single failed packaging
            # attempt (e.g. the empty-CSV bug, or a TIMEOUT kill mid-write)
            # permanently locks the run out of ever packaging again. Note the
            # readiness check is Slurm state *plus* a real HDF5 read (see
            # _validate_h5), not file existence: packaging now stages to a
            # ".partial" sibling and renames on success, so a killed attempt
            # leaves nothing at this path — but a stale .h5 from an earlier
            # successful run would still be sitting there, and existence alone
            # would read that as "this attempt already completed."
            #
            # The comparison is against the output *this* submission would
            # write, not the one the row happens to remember. The row has a
            # single h5_job_id/h5_output_path slot shared by both scopes, so a
            # subset run that finished its own 30-slide .h5 was blocking a
            # scope="tiled" request for a completely different, non-existent
            # file. Only a recorded job pointing at the same target can say
            # anything about this one.
            target_output = hpl_h5_output_path(HPL_DATASETS_ROOT, dataset_name)
            recorded_output = Path(row["h5_output_path"]) if row["h5_output_path"] else None

            if recorded_output == target_output:
                prior_state = _get_slurm_job_state(row["h5_job_id"])
                if _job_output_ready(target_output, prior_state, validator=_validate_h5):
                    raise HTTPException(400, "Packaging has already completed for this run.")
                if prior_state in IN_FLIGHT_SLURM_STATES:
                    raise HTTPException(
                        400,
                        f"Packaging is already running for this run "
                        f"(Slurm state: {prior_state}).",
                    )
                # Otherwise the prior attempt failed/was cancelled/timed out (or
                # its state is unknown) — fall through and submit a fresh one.
            elif _job_output_ready(target_output, "", validator=_validate_h5):
                # A different target, but a complete and readable .h5 is already
                # sitting at it — from an earlier run, or an earlier scope. Slurm
                # state is irrelevant (no job on this row produced it), so "" is
                # passed deliberately: the file itself is the evidence. Refuse
                # rather than silently overwrite something valid.
                raise HTTPException(
                    400,
                    f"A complete .h5 already exists at {target_output}. Delete or move "
                    f"it first if you want to rebuild it.",
                )

        # With afterok, submitting while any tiling task has failed produces a
        # job whose dependency can never be satisfied. --kill-on-invalid-dep
        # makes Slurm kill it rather than queue it forever, but the user would
        # still just see packaging vanish with no explanation. Refuse up front
        # with something actionable instead. Deliberately conservative: only
        # states we positively observed as failures count, so an unreachable
        # sacct (None) or an unparsed state never blocks a legitimate submit.
        tiling_states = _get_slurm_array_state_counts(job_ids)
        if not allow_incomplete:
            failed_states = {
                state: n for state, n in (tiling_states or {}).items()
                if state != "COMPLETED" and state not in IN_FLIGHT_SLURM_STATES
            }
            if failed_states:
                raise HTTPException(
                    400,
                    {
                        "error": "tiling_incomplete",
                        "submission_id": submission_id,
                        "failed_task_states": failed_states,
                        "message": (
                            f"{sum(failed_states.values())} tiling task(s) did not complete "
                            f"successfully ({failed_states}). Packaging now would produce an "
                            f".h5 missing those slides. Resume the run to retry them, or "
                            f"resubmit with allow_incomplete=true to package without them."
                        ),
                    },
                )

        # Drop the dependency once tiling is terminal, because by then it can
        # only hurt. sbatch resolves --dependency against slurmctld, which
        # forgets a job MinJobAge seconds after it ends (default 300), whereas
        # the state check above reads sacct, whose retention is days. Between
        # those two windows sits the common case: tiling finished yesterday,
        # sacct still reports every task COMPLETED so nothing above objects,
        # and then sbatch rejects the whole submission with "Job dependency
        # problem" because slurmctld no longer recognises the IDs. Packaging
        # was unreachable for exactly the runs most ready to be packaged.
        #
        # This does not weaken the afterok guarantee. afterok exists to stop
        # packaging from starting while tiling is still going or after it
        # failed, and both of those are decided above from sacct state — the
        # dependency is a redundant second opinion here, and a stale one.
        #
        # Conservative on purpose: None means sacct was unreachable, so the
        # states are genuinely unknown and the dependency stays as the
        # backstop. An empty dict is different — per
        # _get_slurm_array_state_counts, that means sacct answered AND squeue
        # confirms nothing is live, i.e. the run aged out of accounting
        # retention entirely, which is the strongest evidence available that
        # it is long finished (and guarantees slurmctld has forgotten it too).
        if tiling_states is not None and not any(
            state in IN_FLIGHT_SLURM_STATES for state in tiling_states
        ):
            job_ids = []
        else:
            # Either sacct couldn't be reached (None — state genuinely
            # unknown) or it reports tasks still in flight. Neither is proof
            # the dependency is submittable, because sacct is not the
            # component that resolves it. Ask the controller, which is: any
            # ID it no longer holds makes the dependency unsatisfiable no
            # matter what sacct believes, while a genuinely running job is
            # always known to it, so filtering here can never drop a
            # dependency that was doing real work. If scontrol itself is
            # unusable (None) nothing is known for certain and the IDs stay
            # untouched — a rejected submit is a better failure than
            # packaging that quietly starts before tiling has finished.
            known_job_ids = _slurm_controller_known_jobs(job_ids)
            if known_job_ids is not None:
                job_ids = known_job_ids

        try:
            packaging_result = submit_packaging_job(
                manifest_path=manifest_path,
                tile_dir=Path(row["tile_dir"]),
                dataset_name=dataset_name,
                tile_dataset_name=tile_dataset_name,
                depends_on_job_ids=job_ids,
                allow_incomplete=allow_incomplete,
                resume=resume,
                partition=row["partition"],
                notify_email=row["notify_email"],
                job_name=job_name,
                # Explicit rather than left to submit_packaging_job's own
                # default (backend_dir.parent / "model_input") — that
                # default happens to match HPL_DATASETS_ROOT's own default
                # today, but only by coincidence; without this, overriding
                # HPL_DATASETS_ROOT would silently only affect single-slide
                # uploads (_run_postupload_pipeline passes it explicitly)
                # and not dataset-wide packaging, splitting the two onto
                # different output roots.
                output_root=HPL_DATASETS_ROOT,
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to submit packaging job: {e}")

        _update_dataset_run(
            submission_id,
            h5_job_id=packaging_result.get("h5_job_id"),
            h5_output_path=packaging_result.get("h5_output_path"),
        )
        _record_run_job(
            submission_id, "packaging", packaging_result.get("h5_job_id"),
            output_path=packaging_result.get("h5_output_path"),
            params={"scope": scope, "resume": resume, "allow_incomplete": allow_incomplete},
        )
        return {"submission_id": submission_id, **packaging_result}


def _count_lines(path: Path) -> int:
    """Line count without holding the file in memory.

    The completed-tiles checkpoint is one label per line and can reach
    millions of lines / hundreds of MB on a full run, so this reads in
    blocks rather than splitlines()-ing the lot.
    """
    total = 0
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                return total
            total += block.count(b"\n")


# How long a .partial can go untouched before it stops counting as evidence of
# a live writer. Generous on purpose: packaging alternates between long decode
# batches (a whole batch of tiles is decoded by the worker pool before anything
# is written) and bursts of writes, so short gaps are normal and a tight window
# would flap between "running" and "stalled" on an entirely healthy job.
_PACKAGING_ACTIVE_WINDOW_SECONDS = 15 * 60


def _packaging_write_activity(final_h5_path: Path) -> dict:
    """Whether something is currently writing this run's .h5, judged from disk.

    Exists because Slurm state is not always available (see
    _get_slurm_job_state) and is never available *promptly* — a job submitted
    seconds ago may have no accounting row at all. The .partial's mtime has
    neither problem: packaging touches it continuously while it runs, and the
    file only exists between the start of an attempt and the os.replace() that
    completes it.

    Costs one stat() on one file, which is what makes it safe to include in the
    10s status poll. Returns partial_* as None/False rather than raising when
    the file isn't there, since "no .partial" is a normal state (not started, or
    already finished and renamed).
    """
    partial_path = final_h5_path.with_name(final_h5_path.name + ".partial")
    try:
        stat = partial_path.stat()
    except OSError:
        return {
            "h5_partial_exists": False,
            "h5_partial_bytes": 0,
            "h5_partial_seconds_since_write": None,
            "h5_packaging_active": False,
        }
    seconds_since_write = max(0.0, time.time() - stat.st_mtime)
    return {
        "h5_partial_exists": True,
        "h5_partial_bytes": stat.st_size,
        "h5_partial_seconds_since_write": round(seconds_since_write, 1),
        "h5_packaging_active": seconds_since_write < _PACKAGING_ACTIVE_WINDOW_SECONDS,
    }


@app.get("/dataset-jobs/{submission_id}/packaging-progress")
def packaging_progress(
    submission_id: str,
    exact: bool = Query(
        True,
        description="Count the completed-tiles checkpoint exactly (hundreds of MB on a "
                    "large run). Pass false for live polling, which estimates tiles "
                    "written from the .partial's size instead — one stat() rather than "
                    "a full file read.",
    ),
):
    """How far an interrupted packaging attempt actually got.

    Deliberately a separate endpoint rather than more fields on
    /status: answering it means counting the lines of a checkpoint file
    that can be hundreds of MB, and /status is polled every 10s by every
    open browser tab. This is only called when someone actually opens the
    packaging step in the UI.

    "resumable" is the question the UI is really asking — whether clicking
    package again would continue the previous attempt or silently start
    from zero. That's true exactly when a .partial and its checkpoint are
    both still on disk; package_slides_to_h5 then validates the recorded
    run identity itself and falls back to a fresh run if anything moved.
    """
    row = _get_dataset_run_row(submission_id)
    output_path = row["h5_output_path"]
    if not row["h5_job_id"] or not output_path:
        return {"state": "not_started", "resumable": False}

    final_path = Path(output_path)
    partial_path = final_path.with_name(final_path.name + ".partial")
    ckpt = _checkpoint_paths(final_path)
    slurm_state = _get_slurm_job_state(row["h5_job_id"])

    activity = _packaging_write_activity(final_path)

    info = {
        "output_path": str(final_path),
        "slurm_state": slurm_state,
        "partial_exists": activity["h5_partial_exists"],
        "partial_bytes": activity["h5_partial_bytes"],
        "seconds_since_write": activity["h5_partial_seconds_since_write"],
        "writing_now": activity["h5_packaging_active"],
        # The finished article, so the UI can distinguish "no output yet" from
        # "output exists but hasn't validated".
        "final_exists": final_path.is_file(),
        "final_bytes": final_path.stat().st_size if final_path.is_file() else 0,
        "resumable": False,
        "tiles_done": None,
        "tiles_done_is_estimate": False,
        "tiles_total": None,
        "percent": None,
        "bytes_per_tile": None,
        "skipped_tiles": None,
    }

    if _job_output_ready(final_path, slurm_state, validator=_validate_h5):
        info["state"] = "complete"
        return info
    # Disk activity outranks a missing Slurm state. A job whose .partial was
    # touched moments ago is running, whatever sacct does or doesn't know — this
    # is what stops a freshly-started job from being reported as "interrupted".
    if slurm_state in IN_FLIGHT_SLURM_STATES or info["writing_now"]:
        info["state"] = "running"
    else:
        info["state"] = "interrupted"

    # tiles_total comes from the run config the attempt itself wrote, so it
    # reflects what that attempt was actually packaging rather than a count
    # recomputed now from possibly-changed tile directories.
    tile_size = None
    img_compression = None
    if ckpt["config"].is_file():
        try:
            config = json.loads(ckpt["config"].read_text())
            info["tiles_total"] = config.get("total_tiles")
            tile_size = config.get("tile_size")
            img_compression = config.get("img_compression")
        except (json.JSONDecodeError, OSError):
            pass

    # Every tile occupies exactly tile_size**2 * 3 bytes in the .h5 (uint8, one
    # tile per chunk) ONLY when the img dataset is uncompressed — that fixed
    # width is what makes the .partial's size a usable proxy for tiles
    # written, costing one stat() instead of a full checkpoint read. Once
    # img_compression is set (see make_hpl_hdf5.py's _IMG_COMPRESSION), each
    # tile's compressed chunk size varies with how much actual detail is in
    # that tile, so this proxy has no fixed divisor to use and is left unset
    # rather than reported as a number that quietly drifts from reality.
    if tile_size and not img_compression:
        info["bytes_per_tile"] = int(tile_size) ** 2 * 3

    if exact and ckpt["completed"].is_file():
        try:
            info["tiles_done"] = _count_lines(ckpt["completed"])
        except OSError:
            pass
    elif info["bytes_per_tile"] and info["partial_bytes"]:
        # Slightly low: HDF5's own metadata (b-tree nodes, the superblock)
        # shares the file, so dividing overstates nothing. Flagged as an
        # estimate so the UI never presents it as a tile-accurate figure.
        info["tiles_done"] = info["partial_bytes"] // info["bytes_per_tile"]
        info["tiles_done_is_estimate"] = True

    # Tiles the attempt gave up on (unreadable/corrupt JPEGs). Small file, and
    # worth surfacing: they count as done for resume purposes but never make it
    # into the .h5, so a run can legitimately finish short of tiles_total.
    if ckpt["skipped"].is_file():
        try:
            info["skipped_tiles"] = _count_lines(ckpt["skipped"])
        except OSError:
            pass

    if info["tiles_done"] is not None and info["tiles_total"]:
        info["percent"] = round(
            100.0 * min(info["tiles_done"], info["tiles_total"]) / info["tiles_total"], 1
        )

    # Resumability is about the checkpoint, so it needs a real count — an
    # estimate from file size says nothing about whether the checkpoint exists.
    # When polling cheaply, fall back to the checkpoint merely being non-empty.
    if info["tiles_done_is_estimate"]:
        try:
            checkpoint_has_content = (
                ckpt["completed"].is_file() and ckpt["completed"].stat().st_size > 0
            )
        except OSError:
            checkpoint_has_content = False
    else:
        checkpoint_has_content = bool(info["tiles_done"])
    info["resumable"] = bool(info["partial_exists"] and checkpoint_has_content)
    return info


class PackagingTestRequest(BaseModel):
    sample_size: Optional[int] = None
    slide_names: Optional[list[str]] = None
    random_seed: Optional[int] = None
    # Which pool the sample is drawn from. "run" keeps the original behaviour
    # (this run's own manifest); "tiled" draws from every slide in the raw
    # directory that has tiles on disk, whichever run produced them — mirroring
    # the same option on /package. Without this a test run could never exceed
    # its run's manifest, so a 30-slide subset run capped every test sample at
    # 30 however large a number the UI offered to accept.
    scope: str = "run"


@app.post("/dataset-jobs/{submission_id}/package-test")
def start_test_packaging_job(submission_id: str, req: PackagingTestRequest):
    """Package a chosen subset of slides into a separately-named test .h5 —
    sanity-check packaging (and a downstream feature-extraction checkpoint)
    against a sample before committing to a multi-hour run over the whole
    dataset. Deliberately NOT tracked on the run's own
    h5_job_id/h5_output_path — a test run succeeding or failing has no
    bearing on whether the real packaging run is allowed to proceed, and
    vice versa (the retry-guard above only ever looks at h5_job_id, which
    this never touches).

    scope="tiled" widens the pool past this run's manifest to everything
    tiled on disk. That is the difference between "test the pipeline on 3
    slides" and "test it on 3500 of the 14,000 I actually have", and only
    the former was previously possible from a subset run.
    """
    if not req.sample_size and not req.slide_names:
        raise HTTPException(400, "Provide sample_size or slide_names for a test run.")
    if req.scope not in ("run", "tiled"):
        raise HTTPException(400, f"scope must be 'run' or 'tiled', got '{req.scope}'.")

    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        if not row["job_id"]:
            # Not status == "submitted" specifically — a cancelled run's tiling
            # batches may well have already finished, and cancelling only flips
            # the status column, leaving the manifest/tiles on disk untouched.
            # job_id existing at all is what actually means tiling was submitted.
            raise HTTPException(400, "Tiling hasn't been submitted yet for this run.")

        manifest_path = Path(row["manifest_path"]) if row["manifest_path"] else None
        if not manifest_path:
            raise HTTPException(400, "This run never got far enough to have a manifest.")
        # Same reasoning as /package: scope="tiled" writes its own fresh
        # manifest from live disk coverage below and only needs this path's
        # parent directory, so the original file's existence is irrelevant to
        # it — only scope="run" actually opens it.
        if req.scope != "tiled" and not manifest_path.is_file():
            raise HTTPException(400, f"Manifest no longer exists on disk: {manifest_path}")

        job_ids = [j for j in row["job_id"].split(",") if j]
        # Same slurmctld-forgot-the-job-IDs problem the real /package endpoint
        # handles — see the long note there. afterany is no more resolvable
        # than afterok once the IDs have aged out of the controller, so a test
        # package against a finished run failed at sbatch for a dependency
        # that had nothing left to wait for.
        test_tiling_states = _get_slurm_array_state_counts(job_ids)
        if test_tiling_states is not None and not any(
            state in IN_FLIGHT_SLURM_STATES for state in test_tiling_states
        ):
            job_ids = []
        else:
            known_job_ids = _slurm_controller_known_jobs(job_ids)
            if known_job_ids is not None:
                job_ids = known_job_ids

        tile_dataset_name = _row_dataset_name(row)

        if req.scope == "tiled":
            # Draw from what is on disk instead of this run's fixed manifest.
            # Tiles already exist, so there is nothing to depend on — keeping
            # this run's tiling job IDs would only expose the submission to
            # slurmctld having forgotten them, same as /package's scope="tiled".
            coverage = _tiled_coverage(
                Path(row["raw_dir"]), Path(row["tile_dir"]), tile_dataset_name
            )
            if not coverage["slides_tiled"]:
                raise HTTPException(
                    400, f"No slide in {row['raw_dir']} has tiles on disk yet."
                )
            manifest_path = manifest_path.parent / (
                f"wsi_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_testpool_{tile_dataset_name}.txt"
            )
            write_manifest([Path(p) for p in coverage["tiled_paths"]], manifest_path)
            job_ids = []
            pool_size = coverage["slides_tiled"]
        else:
            pool_size = sum(
                1 for line in manifest_path.read_text().splitlines() if line.strip()
            )

        # Checked here, before the signature and the Slurm lookup, so an
        # over-large request is rejected against the pool the user actually
        # asked for rather than surfacing later from select_slides with no
        # mention of which pool it measured.
        if req.sample_size and req.sample_size > pool_size:
            pool_label = (
                "tiled-on-disk" if req.scope == "tiled" else "run's manifest"
            )
            widen_hint = (
                ""
                if req.scope == "tiled"
                else (
                    " Retry with scope='tiled' to draw from every slide tiled "
                    "on disk instead of just this run's."
                )
            )
            raise HTTPException(
                400,
                {
                    "error": "sample_larger_than_pool",
                    "submission_id": submission_id,
                    "scope": req.scope,
                    "pool_size": pool_size,
                    "requested": req.sample_size,
                    "message": (
                        f"Asked for {req.sample_size} slides but the "
                        f"{pool_label} pool holds {pool_size}.{widen_hint}"
                    ),
                },
            )

        # A random draw with no seed is a different set of slides every time, so
        # leaving the seed out of the signature made two genuinely different
        # attempts share one output path — and, because the Slurm lookup below
        # treats a matching name as "already done", made re-rolling a random
        # sample impossible: same N came back as "already completed" while never
        # having packaged those slides. Resolving a seed here keeps the
        # signature honest, and makes the draw reproducible, which it never was.
        # The cost is that an unseeded double-click submits two jobs instead of
        # being deduped; pass an explicit random_seed to get the old idempotency.
        resolved_seed = req.random_seed
        if resolved_seed is None and req.sample_size:
            resolved_seed = random.randrange(1_000_000_000)

        base_dataset_name = _effective_h5_dataset_name(
            tile_dataset_name,
            # scope="tiled" is drawn from full coverage rather than this run's
            # subset, so it doesn't inherit the run's "_subset_N" naming.
            False if req.scope == "tiled" else bool(row["is_subset"]),
        )
        # Signature-suffixed rather than a fixed "_test_sample" name: this
        # endpoint has no DB row to guard against, so the *filename itself*
        # is what has to keep two different test attempts (different
        # scope/sample_size/slide_names/seed) from clobbering each other's
        # output, and keeps a re-run of an explicitly-seeded attempt idempotent
        # (same signature -> same path -> caught by the Slurm lookup below)
        # rather than piling up duplicate jobs.
        signature = _attempt_signature(
            "package-test", submission_id, req.scope, req.sample_size,
            sorted(req.slide_names or []), resolved_seed,
        )
        test_dataset_name = f"{base_dataset_name}_test_sample_{signature}"
        job_name = f"hpl_h5_package_test_{signature}"

        existing_job_id = _find_job_id_by_name(job_name)
        if existing_job_id:
            existing_output = hpl_h5_output_path(HPL_DATASETS_ROOT, test_dataset_name)
            existing_state = _get_slurm_job_state(existing_job_id)
            if _job_output_ready(existing_output, existing_state, validator=_validate_h5):
                raise HTTPException(
                    400, "A test packaging run with these exact parameters has already completed."
                )
            if existing_state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    "A test packaging run with these exact parameters is already in progress "
                    f"(Slurm state: {existing_state}).",
                )
            # Otherwise that attempt failed/was cancelled/its state is
            # unknown — fall through and let this submit a fresh one.

        try:
            result = submit_packaging_job(
                manifest_path=manifest_path,
                tile_dir=Path(row["tile_dir"]),
                dataset_name=test_dataset_name,
                tile_dataset_name=tile_dataset_name,
                depends_on_job_ids=job_ids,
                partition=row["partition"],
                notify_email=row["notify_email"],
                sample_size=req.sample_size,
                slide_names=req.slide_names,
                random_seed=resolved_seed,
                job_name=job_name,
                # A test run packages a deliberately-chosen handful of slides,
                # so a partial dataset is the entire point — afterok would
                # make it hostage to every unrelated slide in the run having
                # tiled cleanly, which defeats the purpose of a quick check.
                allow_incomplete=True,
                # Same as the real /package endpoint — keep test packaging
                # on the same output root as everything else instead of
                # silently falling back to submit_packaging_job's own default.
                output_root=HPL_DATASETS_ROOT,
            )
        except ValueError as e:
            # select_slides() rejecting the request — asking for more slides
            # than this run's manifest holds, or naming slides that aren't in
            # it. That's the caller's input, not a server fault, and it used to
            # come back as a 500 the UI could only show as a generic failure.
            raise HTTPException(
                400,
                {
                    "error": "invalid_slide_selection",
                    "submission_id": submission_id,
                    "manifest_slides": row["total_slides"],
                    "message": str(e),
                },
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to submit test packaging job: {e}")

        # Recorded on the run so the job survives a page reload. Written to
        # test_h5_* rather than h5_job_id/h5_output_path on purpose: those gate
        # the packaging retry guard and stage 3's h5_ready, and a subset sample
        # must not satisfy either. This only makes the test job *visible* — it
        # still has no bearing on whether real packaging may proceed.
        #
        # Failure here is logged, not raised: the Slurm job is already submitted
        # by this point, and turning a bookkeeping error into a 500 would leave
        # the caller believing nothing was queued while the job ran anyway.
        try:
            _update_dataset_run(
                submission_id,
                test_h5_job_id=result.get("h5_job_id"),
                test_h5_output_path=result.get("h5_output_path"),
                test_h5_submitted_at=datetime.now(timezone.utc),
                test_h5_params=json.dumps({
                    "scope": req.scope,
                    "sample_size": req.sample_size,
                    "slide_names": req.slide_names,
                    "random_seed": resolved_seed,
                    "pool_size": pool_size,
                }),
            )
        except Exception as e:
            print(f"[warn] test packaging job {result.get('h5_job_id')} submitted "
                  f"but not recorded on {submission_id}: {e}")

        # The history row is what makes a *second* test packaging visible: the
        # test_h5_* columns above hold only the latest attempt, so without this
        # an earlier sample and its .h5 path disappear the moment another runs.
        _record_run_job(
            submission_id, "packaging_test", result.get("h5_job_id"),
            output_path=result.get("h5_output_path"),
            params={
                "scope": req.scope,
                "sample_size": req.sample_size,
                "slide_names": req.slide_names,
                "random_seed": resolved_seed,
                "pool_size": pool_size,
            },
        )

        # scope/pool_size/random_seed are echoed back so the UI can state what
        # was actually drawn and from where. random_seed especially: it's the
        # only record of which slides a random sample picked, and without it
        # a test .h5 worth investigating couldn't be reproduced.
        return {
            "submission_id": submission_id,
            "scope": req.scope,
            "pool_size": pool_size,
            "random_seed": resolved_seed,
            **result,
        }


@app.get("/dataset-jobs/{submission_id}/package-test-status")
def test_packaging_status(submission_id: str, job_id: str, output_path: str):
    """Status for one ad-hoc test packaging job. submission_id isn't
    actually looked up here — test runs aren't persisted to the DB at all
    (see start_test_packaging_job) — it's kept in the path purely for
    routing consistency with the other per-run endpoints. job_id and
    output_path are whatever start_test_packaging_job returned; the caller
    (the UI) is responsible for remembering them between polls.
    """
    state = _get_slurm_job_state(job_id)
    ready = _job_output_ready(Path(output_path), state, validator=_validate_h5)
    return {"job_id": job_id, "slurm_state": state, "ready": ready, "output_path": output_path}


class FeatureExtractionRequest(BaseModel):
    checkpoint: str
    model: str = "BarlowTwins_3"
    marker: str = "he"


@app.post("/dataset-jobs/{submission_id}/extract-features")
def start_feature_extraction_job(submission_id: str, req: FeatureExtractionRequest):
    """Manually start Stage 2 (running the packaged .h5 through Kai's frozen
    self-supervised encoder), once the user confirms the .h5 is ready and
    supplies a checkpoint path. Never auto-triggered — the checkpoint is a
    per-run input only the user knows, so this was always going to need a
    manual step, not just the "click to proceed" gating the other stages
    also now use.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        if not row["h5_job_id"] or not row["h5_output_path"]:
            raise HTTPException(400, "Packaging hasn't been started for this run yet.")

        h5_path = Path(row["h5_output_path"])
        # Read the dataset_name back out of the .h5 path itself (its parent dir
        # name) rather than recomputing via _effective_h5_dataset_name a second
        # time — that function numbers "_subset_N" by scanning the filesystem,
        # so calling it again here (after packaging already created that dir)
        # could silently pick a different, wrong number than packaging actually
        # used.
        dataset_name = h5_path.parent.name
        # Scoped to this one submission so _find_job_id_by_name can recover
        # it below — must match the job_name passed to
        # submit_feature_extraction_job further down.
        job_name = f"hpl_feature_extraction_{submission_id}"

        if not row["extraction_job_id"]:
            # See the matching comment in start_packaging_job — "no
            # extraction_job_id on record" doesn't prove nothing was
            # attempted, it's also what a crash between sbatch succeeding
            # and the _update_dataset_run() call right after it looks like.
            recovered_job_id = _find_job_id_by_name(job_name)
            if recovered_job_id:
                recovered_output_path = str(
                    expected_extraction_output_path(HPL_REPO_DIR, req.model, dataset_name, h5_path)
                )
                _update_dataset_run(
                    submission_id,
                    extraction_job_id=recovered_job_id,
                    extraction_output_path=recovered_output_path,
                )
                row["extraction_job_id"] = recovered_job_id
                row["extraction_output_path"] = recovered_output_path

        if row["extraction_job_id"]:
            # A previous attempt exists — only block starting another one if
            # that attempt actually succeeded or is still in flight. Blocking
            # unconditionally here would mean a single failed attempt (wrong
            # conda setup, bad checkpoint path, whatever) permanently locks out
            # ever retrying feature extraction for this run. File existence
            # alone isn't enough to mean "succeeded" either — same reasoning as
            # packaging above, an output file created early and then left
            # behind by a killed/timed-out attempt would otherwise look done.
            prior_output = Path(row["extraction_output_path"]) if row["extraction_output_path"] else None
            prior_state = _get_slurm_job_state(row["extraction_job_id"])
            # Validated, not just existence-checked: the encoder creates its
            # output file before encoding anything, so a killed attempt leaves
            # one behind that would otherwise read as a completed extraction
            # and permanently block this run at "already completed".
            # Bound to the input's tile count for the same reason the status
            # payload is: an output covering a fraction of the slides passes
            # every internal check, and refusing the retry with "already
            # completed" is precisely how a partial run becomes permanent.
            prior_expected = _extraction_expected_rows(row)
            if _job_output_ready(
                prior_output, prior_state,
                validator=lambda p: _validate_extraction_output(p, expected_rows=prior_expected),
            ):
                raise HTTPException(400, "Feature extraction has already completed for this run.")
            if prior_state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    f"Feature extraction is already running for this run (Slurm state: {prior_state}).",
                )
            # Otherwise the prior attempt failed/was cancelled/timed out (or its
            # state is unknown) — fall through and let this submit a fresh one.

        # Packaging must have genuinely COMPLETED *and* produced a readable
        # .h5 before extraction is allowed to start. File existence was the
        # entire check here, which was never sufficient — and is now actively
        # misleading in the opposite direction too, since packaging writes to
        # a ".partial" sibling and renames on success: a run still in flight
        # leaves nothing at this path, while a *stale* .h5 from an earlier
        # successful run does, and would have been accepted as this run's
        # output. Feature extraction is a long GPU job; discovering the input
        # was truncated hours in is the failure this prevents.
        h5_state = _get_slurm_job_state(row["h5_job_id"])
        if not _job_output_ready(h5_path, h5_state, validator=_validate_h5):
            if h5_state is None:
                raise HTTPException(
                    503,
                    "Couldn't reach Slurm to confirm packaging finished — try again shortly.",
                )
            if h5_state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    f"Packaging is still running (Slurm state: {h5_state}) — wait for it to "
                    "finish before extracting features.",
                )
            if not h5_path.is_file():
                raise HTTPException(
                    400,
                    f"Packaging has not produced its .h5 (Slurm state: "
                    f"{h5_state or 'no Slurm record'}): {h5_path}",
                )
            valid, reason = _validate_h5(h5_path)
            if not valid:
                raise HTTPException(
                    400,
                    f"The packaged .h5 is not usable ({reason}) — re-run packaging for this "
                    f"run before extracting features: {h5_path}",
                )
            raise HTTPException(
                400,
                f"Packaging did not complete successfully (Slurm state: "
                f"{h5_state or 'no Slurm record'}) — re-run packaging first.",
            )

        checkpoint = req.checkpoint.strip()
        if not checkpoint:
            raise HTTPException(400, "Checkpoint path is required.")

        try:
            result = submit_feature_extraction_job(
                real_hdf5_path=h5_path,
                checkpoint=checkpoint,
                dataset_name=dataset_name,
                model=req.model,
                marker=req.marker,
                notify_email=row["notify_email"],
                job_name=job_name,
            )
        except (NotADirectoryError, FileExistsError, FileNotFoundError) as e:
            # Misconfiguration and already-done are user-fixable states, not
            # server faults — 400 with the message the submitter composed.
            # FileNotFoundError covers a missing Singularity SIF / binary.
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to submit feature extraction job: {e}")

        _update_dataset_run(
            submission_id,
            extraction_job_id=result.get("extraction_job_id"),
            extraction_output_path=result.get("expected_output_path"),
            extraction_checkpoint=checkpoint,
        )
        _record_run_job(
            submission_id, "extraction", result.get("extraction_job_id"),
            output_path=result.get("expected_output_path"),
            params={"checkpoint": checkpoint, "model": req.model, "marker": req.marker},
        )
        return {"submission_id": submission_id, **result}


class FeatureExtractionTestRequest(BaseModel):
    h5_path: str
    checkpoint: str
    model: str = "BarlowTwins_3"
    marker: str = "he"


@app.post("/dataset-jobs/{submission_id}/extract-features-test")
def start_test_feature_extraction_job(submission_id: str, req: FeatureExtractionTestRequest):
    """Run feature extraction against an arbitrary .h5 — typically a
    test-sample .h5 from /package-test above — rather than this run's own
    tracked h5_output_path. Lets a checkpoint get validated against a
    small sample before running it over the full dataset's .h5, which can
    take hours. Deliberately NOT tracked on the run's own
    extraction_job_id, for the same reason as /package-test.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)  # 404s if the run doesn't exist

        h5_path = Path(req.h5_path)
        if not h5_path.is_file():
            raise HTTPException(400, f".h5 file not found: {h5_path}")
        # No Slurm job to check against here (the caller supplies an arbitrary
        # path), so validating the file itself is the only gate available —
        # and the one that matters, since the usual reason to point this at a
        # hand-picked .h5 is that a packaging run was interrupted.
        valid, reason = _validate_h5(h5_path)
        if not valid:
            raise HTTPException(400, f"Not a usable packaged .h5 ({reason}): {h5_path}")

        checkpoint = req.checkpoint.strip()
        if not checkpoint:
            raise HTTPException(400, "Checkpoint path is required.")

        dataset_name = h5_path.parent.name
        signature = _attempt_signature(
            "extract-test", submission_id, str(h5_path), checkpoint, req.model, req.marker,
        )
        job_name = f"hpl_feature_extraction_test_{signature}"

        existing_job_id = _find_job_id_by_name(job_name)
        if existing_job_id:
            existing_output = expected_extraction_output_path(HPL_REPO_DIR, req.model, dataset_name, h5_path)
            existing_state = _get_slurm_job_state(existing_job_id)
            # The test path knows its input directly rather than through the
            # run row, so bind the count from the .h5 the caller handed us.
            existing_expected = _packaged_h5_rows(h5_path)
            if _job_output_ready(
                existing_output, existing_state,
                validator=lambda p: _validate_extraction_output(p, expected_rows=existing_expected),
            ):
                raise HTTPException(
                    400, "A test feature-extraction run with these exact parameters has already completed."
                )
            if existing_state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    "A test feature-extraction run with these exact parameters is already in "
                    f"progress (Slurm state: {existing_state}).",
                )
            # Otherwise that attempt failed/was cancelled/its state is
            # unknown — fall through and let this submit a fresh one.

        try:
            result = submit_feature_extraction_job(
                real_hdf5_path=h5_path,
                checkpoint=checkpoint,
                dataset_name=dataset_name,
                model=req.model,
                marker=req.marker,
                notify_email=row["notify_email"],
                job_name=job_name,
            )
        except (NotADirectoryError, FileExistsError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to submit test feature extraction job: {e}")

        # Recorded even though this endpoint still writes nothing to
        # slurm_dataset_runs. That invariant is about *gating* — a test attempt
        # must not satisfy extraction_ready — and history does not gate
        # anything, so the two are not in tension. Before this, a test
        # extraction existed only in the caller's session state.
        _record_run_job(
            submission_id, "extraction_test", result.get("extraction_job_id"),
            output_path=result.get("expected_output_path"),
            params={
                "h5_path": str(h5_path),
                "checkpoint": checkpoint,
                "model": req.model,
                "marker": req.marker,
            },
        )
        return {"submission_id": submission_id, **result}


@app.get("/dataset-jobs/{submission_id}/extract-features-test-status")
def test_feature_extraction_status(submission_id: str, job_id: str, output_path: str):
    """Status for one ad-hoc test extraction job — same pattern as
    /package-test-status, nothing persisted server-side."""
    state = _get_slurm_job_state(job_id)
    out_path = Path(output_path)
    ready = _job_output_ready(out_path, state, validator=_validate_extraction_output)
    payload = {"job_id": job_id, "slurm_state": state, "ready": ready, "output_path": output_path}
    if not ready and out_path.is_file():
        # The file existing while the job is finished means an attempt died
        # partway. Surfacing the reason here is what stops the UI showing a
        # bare "not ready" for a job Slurm already called done.
        payload["extraction_invalid_reason"] = _validate_extraction_output(out_path)[1] or None
    return payload


class ClusterAssignmentRequest(BaseModel):
    # Optional so the UI can just say "go": the reference is a deployment-level
    # setting, not a per-run choice, and defaulting to the configured one keeps
    # the common case a single click. Overridable because comparing two
    # references is a real thing to want to do.
    reference: str | None = None
    k: int | None = None
    overwrite: bool = False


class ClusterAssignmentTestRequest(ClusterAssignmentRequest):
    # An arbitrary projections .h5, typically the one a test extraction wrote.
    projections_h5: str


def _assignment_output_path(projections_h5: Path, dataset_name: str) -> Path:
    """Where a run's assignment CSV goes: beside its projections file.

    Not in backend/ or a shared results dir — the CSV is only meaningful
    together with the embeddings it was computed from, and keeping the two
    adjacent is what makes a stale pair obvious instead of plausible.
    """
    return projections_h5.parent / f"{dataset_name}_hpc_assignments.csv"


@app.post("/dataset-jobs/{submission_id}/assign-clusters")
def start_cluster_assignment_job(submission_id: str, req: ClusterAssignmentRequest):
    """Stage 4: assign HPL cluster IDs to this run's embeddings by k-NN vote.

    Gated on extraction having produced a *valid* output rather than merely
    having run. Assigning clusters to a projections file that extraction left
    half-written would read zero rows as embeddings and return cluster IDs for
    them — confidently, since every tile gets a nearest neighbour however
    meaningless the vector.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        if not row["extraction_job_id"] or not row["extraction_output_path"]:
            raise HTTPException(400, "Feature extraction hasn't been started for this run yet.")

        projections = Path(row["extraction_output_path"])
        # Bound to the packaged input's tile count. Unbound, this accepts an
        # extraction that covered a fraction of the slides — and Stage 4 would
        # then write a complete, well-formed assignments CSV for that fraction,
        # which Stage 5 loads into the KB as if it were the whole dataset. The
        # per-slide aggregates come out of that silently wrong, with nothing
        # downstream able to tell.
        ok, reason = _validate_extraction_output(
            projections, expected_rows=_extraction_expected_rows(row)
        )
        if not ok:
            raise HTTPException(
                400,
                f"Feature extraction has not produced a usable output yet ({reason}). "
                f"Cluster assignment reads those embeddings, so it would produce IDs "
                f"for rows the encoder never wrote.",
            )

        if row.get("assignment_job_id") and not req.overwrite:
            state = _get_slurm_job_state(row["assignment_job_id"])
            if state in _SLURM_IN_FLIGHT:
                raise HTTPException(
                    400,
                    f"Cluster assignment is already running for this run "
                    f"(job {row['assignment_job_id']}, state {state}).",
                )

        dataset_name = projections.parent.name if projections.parent.name else submission_id
        out_csv = _assignment_output_path(projections, _row_dataset_name(row) or dataset_name)

        try:
            result = submit_cluster_assignment_job(
                projections_h5=projections,
                out_csv=out_csv,
                reference=Path(req.reference) if req.reference else None,
                k=req.k,
                notify_email=row["notify_email"],
                job_name=f"hpl_assign_{submission_id}",
                overwrite=True,  # decided above; the stage is cheap to redo
            )
        except (FileNotFoundError, FileExistsError, ValueError, KeyError) as e:
            # Same split as the other stages: a missing reference or an input
            # that isn't a projections file is the user's to fix, not a fault.
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to submit cluster assignment job: {e}")

        _update_dataset_run(
            submission_id,
            assignment_job_id=result.get("assignment_job_id"),
            assignment_output_path=result.get("out_csv"),
            assignment_reference=result.get("reference_path"),
        )
        _record_run_job(
            submission_id, "assignment", result.get("assignment_job_id"),
            output_path=result.get("out_csv"),
            params={
                "reference": result.get("reference_path"),
                "reference_rows": result.get("reference_rows"),
                "n_clusters": result.get("n_clusters"),
            },
        )
        return {"submission_id": submission_id, **result}


@app.post("/dataset-jobs/{submission_id}/assign-clusters-test")
def start_test_cluster_assignment_job(submission_id: str, req: ClusterAssignmentTestRequest):
    """Assign clusters for an arbitrary projections .h5 — typically the output
    of a test extraction — without touching this run's tracked Stage 4 state.

    Deliberately writes nothing to slurm_dataset_runs, for the same reason
    /extract-features-test does not: a test attempt must never satisfy
    assignment_ready and let the run look further along than it is.
    """
    projections = Path(req.projections_h5)
    ok, reason = _validate_extraction_output(projections)
    if not ok:
        raise HTTPException(400, f"{projections} is not a usable projections file ({reason}).")

    out_csv = projections.with_name(f"{projections.stem}_hpc_assignments.csv")
    try:
        result = submit_cluster_assignment_job(
            projections_h5=projections,
            out_csv=out_csv,
            reference=Path(req.reference) if req.reference else None,
            k=req.k,
            job_name=f"hpl_assign_test_{submission_id}",
            overwrite=True,
        )
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to submit test cluster assignment job: {e}")

    _record_run_job(
        submission_id, "assignment_test", result.get("assignment_job_id"),
        output_path=result.get("out_csv"),
        params={"projections_h5": str(projections), "reference": result.get("reference_path")},
    )
    return {"submission_id": submission_id, **result}


def _kb_load_source_csv(row: dict, override_path: str | None = None) -> Path:
    """This run's assignment CSV, or an explicit override path.

    The override exists for output from Stage 4's "Test on a sample .h5"
    mode: that mode deliberately never records an output path on
    slurm_dataset_runs (see start_test_cluster_assignment_job), so there is no
    tracked path here to fall back to for it. Given one explicitly, it is
    validated exactly the same way — Stage 5 does not get a looser bar just
    because the CSV did not come from this run's own tracking.

    Gated on the same validator the /status endpoint uses for
    assignment_ready — a CSV that exists but is a stub (interrupted run, half
    the columns) is not a state Stage 5 can load from, whatever the DB row's
    assignment_job_id says.
    """
    if override_path:
        csv_path = Path(override_path)
    else:
        path = row.get("assignment_output_path")
        if not path:
            raise HTTPException(400, "Cluster assignment hasn't been run for this dataset yet.")
        csv_path = Path(path)

    ok, reason = _validate_assignment_output(csv_path)
    if not ok:
        source = "This CSV" if override_path else "This run's assignment output"
        # Name the path. "no output file" without it is unactionable — the
        # commonest cause is a typed path that does not exist, and the message
        # was giving no way to tell that apart from a corrupt file.
        detail = (
            f"{source} isn't usable ({reason}):\n  {csv_path}\n\n"
            f"Stage 5 loads exactly what's there, so it refuses to read a partial "
            f"or missing CSV rather than loading whatever rows happen to be there."
        )
        if reason == "no output file":
            detail += (
                f"\n\nNothing exists at that path. Check it with "
                f"`ls -l {csv_path}`. If you meant the output of "
                f"migrate_tile_names.py, that file is only written when it is run "
                f"with --commit — a dry run reports what it would do and writes "
                f"nothing."
            )
        elif reason.startswith("missing column"):
            detail += (
                "\n\nIf the columns look like data values, this CSV has no header "
                "row. migrate_tile_names.py restores one, and doing so also "
                "recovers the first tile, which a headerless read silently drops."
            )
        raise HTTPException(400, detail)
    return csv_path


class KbLoadPreviewRequest(BaseModel):
    # min_margin previews compute_profiles()'s own exclusion (see
    # load_hpc_assignments.py) — leave-one-out validation put tiles below 0.1
    # vote_margin at 57% correct and 0.1-0.25 at 76%, against 92%+ once margin
    # clears 0.25, so this is how many of those the CSV actually holds before
    # anyone decides whether to exclude them.
    min_margin: float = 0.0
    # See _kb_load_source_csv — output from Stage 4's "Test on a sample .h5"
    # mode has no tracked path, so it has to be given explicitly.
    csv_path: str | None = None


@app.post("/dataset-jobs/{submission_id}/kb-load-preview")
def preview_kb_load(submission_id: str, req: KbLoadPreviewRequest = KbLoadPreviewRequest()):
    """Stage 5, dry-run half: what loading an assignment CSV into the
    Knowledge Bank would do, computed without changing anything.

    Same report load_hpc_assignments.py prints for --dry-run (its default
    posture), just returned as JSON instead of stdout — this endpoint calls
    the identical read_assignments()/inspect() pair the CLI does, so the two
    can never disagree about what a load would do.
    """
    row = _get_dataset_run_row(submission_id)
    csv_path = _kb_load_source_csv(row, override_path=req.csv_path)

    try:
        frame, cluster_column = _read_kb_assignments(csv_path)
    except SystemExit as e:
        raise HTTPException(400, str(e))

    report = _inspect_kb_load(_get_engine(), frame, cluster_column, min_margin=req.min_margin)
    match_rate = report["matched"] / report["rows"] if report["rows"] else 0.0
    report.update({
        "cluster_column": cluster_column,
        "match_rate": match_rate,
        "min_match_rate": _KB_MIN_MATCH_RATE,
        "would_refuse_low_match": match_rate < _KB_MIN_MATCH_RATE,
        # Tracked run state is meaningless for an explicit csv_path — it may
        # be a completely different run's test output, so showing this run's
        # own kb_load_done/at/rows next to it would claim a connection that
        # is not there.
        "already_loaded": bool(row.get("kb_load_done")) if not req.csv_path else None,
        "kb_load_at": (row["kb_load_at"].isoformat() if row.get("kb_load_at") else None)
                      if not req.csv_path else None,
        "kb_load_rows": row.get("kb_load_rows") if not req.csv_path else None,
        "kb_load_reference": row.get("kb_load_reference") if not req.csv_path else None,
    })
    return report


class KbLoadRequest(BaseModel):
    # Mirrors load_hpc_assignments.py's CLI flags. cancer_type stays optional
    # rather than guessed — hpl_profile_summary.cancer_type is left unset when
    # omitted, same as the CLI, rather than this endpoint inventing a value the
    # CLI never would.
    cancer_type: str | None = None
    allow_unknown_clusters: bool = False
    skip_profiles: bool = False
    # See KbLoadPreviewRequest.min_margin — this is the value that actually
    # gets applied to hpl_profile_proportion/summary, not just previewed.
    min_margin: float = 0.0
    # See _kb_load_source_csv. This still performs a real write to the KB —
    # unlike Stage 4's own test mode, there is no throwaway version of
    # "filling the knowledge bank". What it skips is recording kb_load_done
    # against this particular run, since the CSV may not be this run's own.
    csv_path: str | None = None


@app.post("/dataset-jobs/{submission_id}/kb-load")
def commit_kb_load(submission_id: str, req: KbLoadRequest):
    """Stage 5: write cluster assignments into tile_registry, plus the
    hpl_profile_proportion/summary aggregates the chatbot and HPC panels
    actually read.

    This is load_hpc_assignments.py's --commit path, called in-process rather
    than reimplemented — every guard the CLI enforces (95% match rate, unknown
    cluster IDs) applies here unchanged, because it runs the same function.
    That is deliberate: the one script that mutates the shared KB should have
    exactly one implementation of what makes a load safe to commit, not one
    for the terminal and a looser one for the UI.
    """
    row = _get_dataset_run_row(submission_id)
    csv_path = _kb_load_source_csv(row, override_path=req.csv_path)

    try:
        frame, cluster_column = _read_kb_assignments(csv_path)
    except SystemExit as e:
        raise HTTPException(400, str(e))

    eng = _get_engine()
    report = _inspect_kb_load(eng, frame, cluster_column, min_margin=req.min_margin)
    match_rate = report["matched"] / report["rows"] if report["rows"] else 0.0

    problems = []
    if match_rate < _KB_MIN_MATCH_RATE:
        problems.append(
            f"only {match_rate * 100:.1f}% of rows match tile_registry (need "
            f"{_KB_MIN_MATCH_RATE * 100:.0f}%). The usual cause is a slide-naming "
            f"difference between the .h5 and the registry, not missing tiles."
        )
    if report["unknown_clusters"] and not req.allow_unknown_clusters:
        problems.append(
            f"{len(report['unknown_clusters'])} cluster ID(s) have no hpc_dictionary "
            f"row: {report['unknown_clusters'][:10]}. Those tiles would show a cluster "
            f"with no pattern or malignancy annotation. Pass allow_unknown_clusters if "
            f"that is intended."
        )
    if problems:
        raise HTTPException(400, "Refusing to load: " + "; ".join(problems))

    profiles = None
    if not req.skip_profiles:
        profiles = _compute_kb_profiles(frame, cluster_column, req.cancer_type,
                                        min_margin=req.min_margin)

    updated = _write_kb_load(eng, frame, cluster_column, profiles=profiles)
    reference = report["reference"]

    if not req.csv_path:
        # Only this run's own tracked assignment marks the run's KB-load
        # state. An explicit csv_path may be a different run's test output
        # entirely (see _kb_load_source_csv), so recording it here would
        # claim progress this run never actually made.
        _update_dataset_run(
            submission_id,
            kb_load_done=True,
            kb_load_at=datetime.now(timezone.utc),
            kb_load_rows=updated,
            kb_load_reference=reference,
        )

    return {
        "submission_id": submission_id,
        "updated_rows": updated,
        "matched": report["matched"],
        "reference": reference,
        "profiles_written": profiles is not None,
        "min_margin": req.min_margin,
        "excluded_from_aggregates": report["excluded_from_aggregates"],
        "recorded_on_run": req.csv_path is None,
    }


@app.post("/dataset-jobs/{submission_id}/cancel")
def cancel_dataset_job(submission_id: str):
    """Cancel every Slurm job (all tiling batches + the packaging job, if
    any) associated with this submission via scancel, and mark it cancelled.
    scancel on a job that's already finished is a harmless no-op, so this
    doesn't need to know which of the jobs are still actually running.
    """
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM slurm_dataset_runs WHERE submission_id = :submission_id"),
            {"submission_id": submission_id},
        ).mappings().fetchone()

    if not row:
        raise HTTPException(404, f"No dataset job found with id {submission_id}")

    job_ids_to_cancel = [j for j in (row["job_id"] or "").split(",") if j]
    if row["h5_job_id"]:
        job_ids_to_cancel.append(row["h5_job_id"])
    if row["extraction_job_id"]:
        job_ids_to_cancel.append(row["extraction_job_id"])

    cancelled = []
    scancel_error = None
    if job_ids_to_cancel:
        try:
            subprocess.run(
                ["scancel", *job_ids_to_cancel],
                capture_output=True, text=True, timeout=15, check=True,
            )
            cancelled = job_ids_to_cancel
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            scancel_error = str(e)

    _update_dataset_run(submission_id, status="cancelled")

    return {
        "submission_id": submission_id,
        "cancelled_job_ids": cancelled,
        "scancel_error": scancel_error,
    }


@app.get("/dataset-jobs")
def list_dataset_jobs(
    with_state: bool = Query(
        False,
        description="Also resolve each run's live Slurm state (one sacct call for "
                    "the whole list). Off by default so the plain listing stays a "
                    "single DB query.",
    ),
    limit: int = Query(
        25, ge=1, le=200,
        description="How many recent runs to resolve state for. Only applies with "
                    "with_state=true.",
    ),
):
    """Past/active dataset job submissions, most recent first.

    with_state=true adds, per run:
      stage        which pipeline stage it has reached, from the row alone
      slurm_state  running / pending / complete / failed / no record / unknown

    Resolved for the whole list in ONE sacct call (see _slurm_states_by_job) —
    a per-run lookup would be one call each, and this endpoint is what draws the
    "Recent dataset jobs" list on every page render.
    """
    eng = _get_engine()
    df = pd.read_sql(
        "SELECT * FROM slurm_dataset_runs ORDER BY submitted_at DESC", eng
    )
    rows = json.loads(df.to_json(orient="records", date_format="iso"))
    if not with_state:
        return rows

    considered = rows[:limit]
    wanted: list[str] = []
    for row in considered:
        # The stage each run has reached decides which job ID says whether it is
        # busy. Reporting the tiling array's state for a run that finished
        # tiling hours ago and is now packaging would describe the wrong job.
        for field in ("extraction_job_id", "h5_job_id", "job_id"):
            value = row.get(field)
            if value:
                wanted.extend(j for j in str(value).split(",") if j)
                break

    states = _slurm_states_by_job(sorted(set(wanted)))

    for row in considered:
        stage, job_ids = None, []
        if row.get("extraction_job_id"):
            stage, job_ids = "extracting features", [row["extraction_job_id"]]
        elif row.get("h5_job_id"):
            stage, job_ids = "packaging", [row["h5_job_id"]]
        elif row.get("job_id"):
            stage = "tiling"
            job_ids = [j for j in str(row["job_id"]).split(",") if j]

        row["stage"] = stage or (row.get("status") or "queued")

        # A row status of error/cancelled is a decision already recorded about
        # the run and outranks whatever Slurm remembers about its jobs — a
        # cancelled run's batches may well read COMPLETED.
        if row.get("status") in ("error", "cancelled"):
            row["slurm_state"] = row["status"]
        elif not job_ids:
            row["slurm_state"] = row.get("status") or "queued"
        elif states is None:
            row["slurm_state"] = "unknown"
        else:
            seen: set[str] = set()
            for job_id in job_ids:
                for part in str(job_id).split(","):
                    if part:
                        seen |= states.get(part.split("_", 1)[0], set())
            # seen passed as-is: empty means sacct ran and had no rows for these
            # jobs ("no record" — aged out, or too fresh), which is a different
            # answer from sacct being unreachable ("unknown") handled above.
            row["slurm_state"] = _coarse_run_state(seen)

    return rows


def _packaging_scopes_by_run() -> dict[str, set[str]]:
    """Which packaging scopes each run has used, from the job history.

    Needed because "did this run package the whole dataset?" is not answerable
    from slurm_dataset_runs alone. A run flagged is_subset normally produces a
    `_subset_N` .h5, but the same run packaged with scope="tiled" packages
    every slide with tiles on disk and gets the plain, unsuffixed name — a full
    dataset .h5 produced by a subset run. Only the recorded params say which
    happened.

    One query for the whole table: it holds a handful of rows per run, and the
    alternative is a lookup per dataset on an endpoint the UI polls.
    """
    scopes: dict[str, set[str]] = {}
    try:
        eng = _get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT submission_id, params FROM slurm_dataset_run_jobs "
                    "WHERE stage = 'packaging'"
                )
            ).mappings().fetchall()
    except Exception as e:
        # Same degradation as _run_job_history: the table may not exist yet if
        # the code is deployed before migrate_dataset_run_jobs.sql is run. No
        # history means no scope overrides, which lands on is_subset alone —
        # conservative (a subset run's .h5 won't be claimed as the dataset's).
        print(f"[warn] could not read packaging scopes: {e}")
        return scopes

    for row in rows:
        params = row["params"]
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                continue
        if not isinstance(params, dict):
            continue
        scope = params.get("scope")
        if scope:
            scopes.setdefault(str(row["submission_id"]), set()).add(str(scope))
    return scopes


@app.get("/datasets")
def list_datasets(
    dataset_name: Optional[str] = Query(
        None,
        description="Return only this dataset. Required alongside raw_dir when "
                    "asking for coverage.",
    ),
    raw_dir: Optional[str] = Query(
        None,
        description="Return only the dataset tiled from this directory. Two runs "
                    "can share a dataset_name from different raw directories.",
    ),
    coverage: bool = Query(
        False,
        description="Also walk the filesystem for the authoritative count of how "
                    "many slides actually have tiles. Stats two files per slide "
                    "over cephfs, so this is a button, not a poll — and only "
                    "allowed for a single named dataset.",
    ),
):
    """Every dataset, with all of its runs rolled up into one pipeline state.

    This is the dataset-level counterpart to /dataset-jobs, which lists runs.
    The distinction matters because a resume does not continue a run, it starts
    a new one (see POST /dataset-jobs/{id}/resume) — so a dataset that took
    three resumes to tile is four rows there, none of which can say whether the
    dataset is finished. Here they are one entry with one answer and one
    next_action.

    Cost: one query for the runs, one for the packaging scopes, and a bounded
    Slurm lookup (see _listing_job_states) — squeue for everything, sacct only
    for recent jobs and only within a time budget. That is what makes it safe
    to poll. coverage=true is the exception and is deliberately restricted to a
    single dataset.
    """
    if coverage and not (dataset_name and raw_dir):
        raise HTTPException(
            400,
            "coverage=true needs both dataset_name and raw_dir — it walks the "
            "filesystem per slide, so it is not run across every dataset at once.",
        )

    eng = _get_engine()
    df = pd.read_sql(
        "SELECT * FROM slurm_dataset_runs ORDER BY submitted_at ASC", eng
    )
    rows = json.loads(df.to_json(orient="records", date_format="iso"))

    grouped = group_runs_by_dataset(rows)
    if dataset_name:
        grouped = [d for d in grouped if d["dataset_name"] == dataset_name]
    if raw_dir:
        wanted = raw_dir.rstrip("/")
        grouped = [d for d in grouped if d["raw_dir"].rstrip("/") == wanted]

    job_states, states_complete = _listing_job_states(grouped)

    scopes = _packaging_scopes_by_run()

    resolved = []
    for dataset in grouped:
        dataset_coverage = None
        if coverage:
            runs = dataset["runs"]
            # tile_dir is recorded per run and PROCESSED_TILES_DIR may have
            # moved since; the run's own value is what its tiles were written
            # under. Newest run wins, matching what a resume would use.
            tile_dir = Path(runs[-1]["tile_dir"]) if runs and runs[-1].get("tile_dir") else PROCESSED_TILES_DIR
            dataset_coverage = _tiling_readiness(
                Path(dataset["raw_dir"]), tile_dir, dataset["dataset_name"]
            )
        resolved.append(
            rollup_dataset(
                dataset["raw_dir"],
                dataset["dataset_name"],
                dataset["runs"],
                job_states=job_states,
                # One stat per artifact (not per slide), which is what keeps
                # this pollable. See _artifact_status for why existence at the
                # final path is trustworthy: packaging stages to `.partial`
                # and only os.replace()s on success.
                path_exists=lambda p: bool(p) and Path(p).is_file(),
                packaging_scopes=scopes,
                coverage=dataset_coverage,
            )
        )

    return {
        "datasets": resolved,
        "slurm_reachable": job_states is not None,
        # False means some jobs were answered by the live queue alone, so a
        # recent failure can be showing here as "no longer queued". Open the run
        # for the precise answer — /dataset-jobs/{id}/status asks accounting
        # about that one run and can afford to.
        "slurm_states_complete": states_complete,
    }


@app.get("/dataset-jobs/{submission_id}/status")
def dataset_job_status(submission_id: str):
    """Status for one dataset submission.

    While status is queued/discovering/error-with-no-job-yet, this just
    reflects the slurm_dataset_runs row — there's no Slurm job yet to ask
    sacct about. Once a job_id exists, this layers on live Slurm array state
    (via sacct) cross-referenced against each slide's actual
    tiling_summary.json, so a slide that "completed" with zero saved tiles
    shows up distinctly from one that genuinely succeeded. This is keyed off
    job_id existing, not off status == "submitted" — cancelling a run only
    flips its status column, it doesn't touch the manifest or the tiles
    already written to disk, so a cancelled run whose tiling batches had
    already finished should still report tiling_complete accurately (the UI
    uses this to offer test packaging/extraction against a cancelled run's
    already-tiled slides).
    """
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM slurm_dataset_runs WHERE submission_id = :submission_id"),
            {"submission_id": submission_id},
        ).mappings().fetchone()

    if not row:
        raise HTTPException(404, f"No dataset job found with id {submission_id}")
    row = dict(row)

    if not row["job_id"]:
        # No job_id on record doesn't prove tiling was never submitted — it's
        # also what a crash between submit_dataset_array's sbatch call(s) and
        # _run_dataset_submission's own follow-up DB write looks like (see
        # that function's _persist_plan). Recover by this submission's own
        # job name before assuming the row's stuck status column (e.g. still
        # "discovering") reflects reality.
        recovered = _find_job_ids_by_name_prefix(f"wsi_mask_tile_{submission_id}")
        if recovered:
            recovered_job_id = ",".join(recovered)
            _update_dataset_run(submission_id, status="submitted", job_id=recovered_job_id)
            row["job_id"] = recovered_job_id
            if row["status"] not in ("cancelled", "error"):
                row["status"] = "submitted"

    base = {
        "submission_id": submission_id,
        "status": row["status"],
        "error": row["error"],
        "raw_dir": row["raw_dir"],
        "job_id": row["job_id"],
        "total_slides": row["total_slides"],
        # Whether this run's manifest is itself a subset of the raw directory
        # (submitted with sample_size / slide_names) rather than every slide in
        # it. Exposed because the packaging step's "Full dataset" option means
        # "every slide in *this run's manifest*", which for a subset run is not
        # the full dataset at all — without this the UI had no way to say so,
        # and a run created as a 30-slide sample looked identical to one over
        # the whole directory right up until the .h5 came out short.
        "is_subset": bool(row["is_subset"]),
        # Needed by the UI to offer "run the whole directory" as a *new* run
        # that reuses this one's tile output folder — the already-tiled slides
        # are only skipped when the dataset name matches (the skip check in
        # submit_mask_tile_slurm.py is scoped to tile_dir/<dataset_name>/).
        "dataset_name": _row_dataset_name(row),
        "partition": row["partition"],
        "notify_email": row["notify_email"],
        # What this run tiled with, so a derived submission (resume, or the
        # packaging step's full-directory run) can reuse it instead of asking
        # the user to retype settings it has no way to verify. None means the
        # run predates the tiling_params column — the UI must say so rather
        # than presenting the current defaults as this run's settings.
        "tiling_params": _row_tiling_params(row),
        "h5_job_id": row["h5_job_id"],
        "h5_output_path": row["h5_output_path"],
        "extraction_job_id": row["extraction_job_id"],
        "extraction_output_path": row["extraction_output_path"],
        "extraction_checkpoint": row["extraction_checkpoint"],
    }

    if row["h5_job_id"]:
        # Three independent conditions, none sufficient alone: the file is
        # present (packaging stages to a ".partial" and only renames on
        # success, so this now genuinely means "a run finished"), the Slurm
        # job reached COMPLETED specifically (not merely a terminal state —
        # FAILED/CANCELLED/TIMEOUT are terminal and mean it did not work), and
        # the .h5 actually opens and reads back (see _validate_h5 — a file can
        # be complete by both of the first two measures and still be
        # unreadable after a bad disk or an interrupted copy).
        h5_path = Path(row["h5_output_path"]) if row["h5_output_path"] else None
        h5_state = _get_slurm_job_state(row["h5_job_id"])
        base["h5_ready"] = _job_output_ready(h5_path, h5_state, validator=_validate_h5)
        base["h5_slurm_state"] = h5_state
        # Surfaced so the UI can say *why* a COMPLETED packaging job still
        # isn't usable, rather than showing a silent "not ready" forever.
        if not base["h5_ready"] and h5_path and h5_path.is_file():
            base["h5_invalid_reason"] = _validate_h5(h5_path)[1] or None
        # Advisory, not a blocker: the run proceeds normally, but the KB load
        # at the end will need migrate_tile_names.py first.
        if base["h5_ready"] and h5_path:
            base["h5_legacy_tile_names"] = _h5_has_legacy_tile_names(h5_path)
        # Slurm-independent evidence that packaging is alive: the .partial is
        # being written to right now. This is the only thing that answers "did
        # my job actually start?" when sacct/squeue can't be reached — without
        # it, h5_slurm_state comes back None and a job that had been running
        # happily for an hour was reported as "interrupted (no Slurm record)".
        # One stat() on one file, so it's cheap enough for the 10s poll.
        if not base["h5_ready"] and h5_path:
            base.update(_packaging_write_activity(h5_path))

    # Test packaging, reported the same way as the real thing so the UI can
    # show a Slurm state ticking and then a finished .h5, rather than the
    # nothing it had once a reload discarded its session_state.
    #
    # row.get() rather than row[...]: this reads columns added by
    # migrate_dataset_runs_test_packaging.sql, and the code may well be deployed
    # before the migration is run. A missing column should degrade to "no test
    # job recorded", not 500 every status poll for every run.
    test_job_id = row.get("test_h5_job_id")
    if test_job_id:
        test_path = Path(row["test_h5_output_path"]) if row.get("test_h5_output_path") else None
        test_state = _get_slurm_job_state(test_job_id)
        base["test_h5_job_id"] = test_job_id
        base["test_h5_output_path"] = row.get("test_h5_output_path")
        base["test_h5_slurm_state"] = test_state
        # Same three-condition test as real packaging (present, COMPLETED,
        # actually readable) — a test .h5 that cannot be opened is no more
        # usable for a checkpoint trial than a real one.
        base["test_h5_ready"] = _job_output_ready(
            test_path, test_state, validator=_validate_h5
        )
        base["test_h5_params"] = _row_test_packaging_params(row)
        if not base["test_h5_ready"] and test_path and test_path.is_file():
            base["test_h5_invalid_reason"] = _validate_h5(test_path)[1] or None
        if not base["test_h5_ready"] and test_path:
            # Prefixed, so a test job's write activity can't be mistaken for the
            # real packaging job's in the same payload.
            activity = _packaging_write_activity(test_path)
            base.update({f"test_{k}": v for k, v in activity.items()})

    if row["extraction_job_id"]:
        ext_path = Path(row["extraction_output_path"]) if row["extraction_output_path"] else None
        ext_state = _get_slurm_job_state(row["extraction_job_id"])
        # Bind the input's tile count into the validator. Without it this only
        # checks the file is internally consistent, and a run that encoded a
        # fraction of the slides — an array job where most tasks died, a
        # sharded run merged from the shards that happened to finish — writes a
        # perfectly self-consistent file and reads as "features ready". That
        # is the whole failure mode this pipeline is written against: right
        # shape, right dtype, no missing values, a third of the data.
        expected = _extraction_expected_rows(row)
        base["extraction_ready"] = _job_output_ready(
            ext_path, ext_state,
            validator=lambda p: _validate_extraction_output(p, expected_rows=expected),
        )
        base["extraction_slurm_state"] = ext_state
        base["extraction_expected_rows"] = expected
        if not base["extraction_ready"] and ext_path and ext_path.is_file():
            # Mirrors h5_invalid_reason / test_h5_invalid_reason above: an
            # output that exists but doesn't validate is the single most
            # confusing state to show without a reason attached.
            base["extraction_invalid_reason"] = _validate_extraction_output(
                ext_path, expected_rows=expected
            )[1] or None

    if row.get("assignment_job_id"):
        asg_path = Path(row["assignment_output_path"]) if row.get("assignment_output_path") else None
        asg_state = _get_slurm_job_state(row["assignment_job_id"])
        base["assignment_ready"] = _job_output_ready(
            asg_path, asg_state, validator=_validate_assignment_output
        )
        base["assignment_slurm_state"] = asg_state
        base["assignment_output_path"] = row.get("assignment_output_path")
        base["assignment_reference"] = row.get("assignment_reference")
        if not base["assignment_ready"] and asg_path and asg_path.is_file():
            base["assignment_invalid_reason"] = _validate_assignment_output(asg_path)[1] or None

    # Stage 5. No job_id/slurm_state pair here — the load runs in-process and
    # either commits in one transaction or doesn't, so kb_load_done is the
    # single fact worth tracking (see migrate_dataset_runs_kb_load.sql).
    base["kb_load_done"] = bool(row.get("kb_load_done"))
    base["kb_load_at"] = row["kb_load_at"].isoformat() if row.get("kb_load_at") else None
    base["kb_load_rows"] = row.get("kb_load_rows")
    base["kb_load_reference"] = row.get("kb_load_reference")

    if not row["job_id"] or not row["manifest_path"]:
        return base

    manifest_path = Path(row["manifest_path"])
    tile_dir = Path(row["tile_dir"]) / _row_dataset_name(row)

    slide_paths: list[Path] = []
    if manifest_path.exists():
        slide_paths = [
            Path(line.strip())
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        ]

    # job_id is comma-joined when a large dataset was split across multiple
    # Slurm array batches — one combined sacct call across all of them
    # (task index 0 of batch 1 and task index 0 of batch 2 are different
    # slides, so per-index detail can't be merged, only summed counts per
    # state — which is all this endpoint actually needs).
    job_ids = [jid.strip() for jid in row["job_id"].split(",") if jid.strip()]
    raw_state_counts = _get_slurm_array_state_counts(job_ids)
    # None means sacct itself failed (missing/timed out) — genuinely
    # unknown, don't guess. {} means sacct ran fine and just has no rows
    # for these job IDs, which for an old run means they've aged out of
    # its accounting-DB retention window, not that they're still pending.
    sacct_unreachable = raw_state_counts is None
    slurm_state_counts = raw_state_counts or {}

    # Whether the "Start packaging" button should appear: every tiling batch
    # has reported a terminal sacct state (nothing still pending/running).
    tiling_complete = bool(slurm_state_counts) and not (
        set(slurm_state_counts) & IN_FLIGHT_SLURM_STATES
    )

    # Reading every slide's _tiling_summary.json is the expensive part of
    # this endpoint (up to one file per slide, over a network filesystem) —
    # the UI only ever displays succeeded/zero_tile/not_attempted once
    # tiling_complete is true, so skip it entirely while still tiling
    # instead of redoing it on every 10s auto-refresh for no visible
    # benefit. Once complete, cache the result forever (see
    # _tiling_breakdown_cache above) instead of re-scanning on every future
    # poll too — together these were the reason status checks kept timing
    # out (first at 30s, then even at 60s) for a 14,000+ slide run.
    #
    # Also run this scan as a fallback when slurm_state_counts came back
    # truly empty — meaning both sacct AND squeue (see
    # _slurm_jobs_live_states) confirmed nothing, not just sacct having a
    # rough moment or a fresh submission it hasn't indexed yet (squeue
    # would have caught that live). Not when sacct itself was unreachable
    # — that's still ambiguous, and doing a full disk scan on every 10s
    # poll of an actively-tiling 14,000-slide run whenever sacct has a
    # rough moment would reintroduce exactly the timeout problem above.
    # A real "nothing, confirmed by both" response means this job predates
    # sacct's retention window — every slide having its completion marker
    # on disk is what actually proves tiling finished in that case.
    succeeded, zero_tile, not_attempted = [], [], []
    need_breakdown = tiling_complete or (not slurm_state_counts and not sacct_unreachable)
    if need_breakdown:
        cached = _tiling_breakdown_cache.get(submission_id)
        if cached is not None:
            succeeded, zero_tile, not_attempted = cached
        else:
            for slide_path in slide_paths:
                slide_id = slide_id_from_raw_path(slide_path)
                summary_path = tile_dir / slide_id / f"{slide_id}_tiling_summary.json"
                if not summary_path.exists():
                    not_attempted.append(slide_id)
                    continue
                try:
                    summary = json.loads(summary_path.read_text())
                except (json.JSONDecodeError, OSError):
                    not_attempted.append(slide_id)
                    continue
                if summary.get("saved_tiles", 0) > 0:
                    succeeded.append(slide_id)
                else:
                    zero_tile.append(slide_id)

        if not tiling_complete and not slurm_state_counts and not sacct_unreachable and slide_paths:
            # Not conditioned on not_attempted being zero — a slide that
            # crashed without ever writing a summary file, or was one of a
            # handful the array genuinely never got to before being
            # cancelled, is still "tiling done" in the sense that matters
            # here: sacct having zero rows already proves this job isn't
            # live or pending anymore (see the comment above), so nothing
            # further is ever going to attempt these slides on its own.
            # Surfacing them as a count the user can see and choose to
            # ignore (the "Ignore N incomplete slide(s)" button below) is
            # correct; silently refusing to ever show a packaging option
            # at all just because a couple of slides never produced output
            # is not.
            tiling_complete = True

        if tiling_complete:
            _tiling_breakdown_cache[submission_id] = (succeeded, zero_tile, not_attempted)

    base.update({
        "tiling_complete": tiling_complete,
        # Distinguishes "tiling is genuinely still running" from "we could not
        # ask Slurm". Both used to arrive at the UI as tiling_complete=False,
        # which it rendered as "waiting on tiling" — a positive claim the
        # server had no evidence for, and one that left packaging blocked with
        # no way forward for as long as sacct stayed unreachable.
        "slurm_unreachable": sacct_unreachable,
        "slurm_state_counts": slurm_state_counts,
        "attempted": len(succeeded) + len(zero_tile),
        "succeeded": len(succeeded),
        "zero_tile_slides": zero_tile,
        "not_yet_attempted": len(not_attempted),
    })
    return base


@app.get("/slide/{slide_id}/info")
def slide_info(slide_id: str):
    slide = _open_slide(slide_id)
    dims = slide.level_dimensions
    return {
        "slide_id": slide_id.upper(),
        "level_count": slide.level_count,
        "level_dimensions": [{"width": w, "height": h} for w, h in dims],
        "mpp_x": slide.properties.get("openslide.mpp-x"),
        "mpp_y": slide.properties.get("openslide.mpp-y"),
        "vendor": slide.properties.get("openslide.vendor"),
        "objective_power": slide.properties.get("openslide.objective-power"),
        "tile_size_5x": TILE_SIZE_5X,
        "scale_5x_to_native": SCALE,
        "tile_size_native": TILE_SIZE_NATIVE,
    }



@app.get("/dzi/{slide_id}.dzi")
def dzi_metadata(slide_id: str):
    dz = _get_deepzoom(slide_id)
    dzi_xml = dz.get_dzi("jpeg")

    return StreamingResponse(
        io.BytesIO(dzi_xml.encode("utf-8")),
        media_type="application/xml",
    )


@app.get("/dzi/{slide_id}_files/{level}/{col}_{row}.jpeg")
def dzi_tile(
    slide_id: str,
    level: int,
    col: int,
    row: int,
    quality: int = Query(90, ge=10, le=100),
):
    dz = _get_deepzoom(slide_id)

    try:
        tile = dz.get_tile(level, (col, row)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            404,
            f"Could not read DZI tile level={level}, col={col}, row={row}: {e}",
        )

    return _jpeg_response(_img_to_jpeg_bytes(tile, quality=quality))

@app.get("/slide/{slide_id}/thumbnail")
def slide_thumbnail(
    slide_id: str,
    max_width: int = Query(3000, ge=100, le=8000),
    quality: int = Query(85, ge=10, le=100),
):
    slide_id = slide_id.strip().upper()
    cached = cache.get(slide_id, kind="thumbnail", level=None, x=None, y=None,
                       width=max_width, height=None)
    if cached:
        return _jpeg_response(_img_to_jpeg_bytes(cached, quality))

    slide = _open_slide(slide_id)
    w0, h0 = slide.level_dimensions[0]
    thumb_height = int(h0 * (max_width / w0))
    thumb = slide.get_thumbnail((max_width, thumb_height)).convert("RGB")
    cache.put(thumb, slide_id, quality=quality, kind="thumbnail", level=None,
              x=None, y=None, width=max_width, height=None)
    return _jpeg_response(_img_to_jpeg_bytes(thumb, quality))


@app.get("/slide/{slide_id}/tile")
def slide_tile(
    slide_id: str,
    level: int = Query(0, ge=0),
    x: int = Query(..., ge=0),
    y: int = Query(..., ge=0),
    w: int = Query(256, ge=64, le=2048),
    h: int = Query(256, ge=64, le=2048),
    quality: int = Query(85, ge=10, le=100),
):
    """Read a tile at (x, y) in *level* coordinates, return JPEG."""
    slide_id = slide_id.strip().upper()
    cached = cache.get(slide_id, kind="tile", level=level, x=x, y=y, width=w, height=h)
    if cached:
        return _jpeg_response(_img_to_jpeg_bytes(cached, quality))

    slide = _open_slide(slide_id)
    if level >= slide.level_count:
        raise HTTPException(400, f"Level {level} out of range (max {slide.level_count - 1})")

    ds = slide.level_downsamples[level]
    origin_x = int(x * ds)
    origin_y = int(y * ds)
    region = slide.read_region((origin_x, origin_y), level, (w, h)).convert("RGB")
    cache.put(region, slide_id, quality=quality, kind="tile", level=level,
              x=x, y=y, width=w, height=h)
    return _jpeg_response(_img_to_jpeg_bytes(region, quality))


@app.get("/slide/{slide_id}/region")
def slide_region(
    slide_id: str,
    x: int = Query(..., description="Native-coordinate X"),
    y: int = Query(..., description="Native-coordinate Y"),
    w: int = Query(TILE_SIZE_NATIVE),
    h: int = Query(TILE_SIZE_NATIVE),
    level: int = Query(0, ge=0),
    quality: int = Query(85),
):
    """Read an arbitrary region in native (level-0) coordinates."""
    slide_id = slide_id.strip().upper()
    cached = cache.get(slide_id, kind="region", level=level, x=x, y=y, width=w, height=h)
    if cached:
        return _jpeg_response(_img_to_jpeg_bytes(cached, quality))

    slide = _open_slide(slide_id)
    ds = slide.level_downsamples[level]
    read_w = int(w / ds)
    read_h = int(h / ds)
    region = slide.read_region((x, y), level, (read_w, read_h)).convert("RGB")
    cache.put(region, slide_id, quality=quality, kind="region", level=level,
              x=x, y=y, width=w, height=h)
    return _jpeg_response(_img_to_jpeg_bytes(region, quality))


@app.get("/slide/{slide_id}/tiles_meta")
def slide_tiles_meta(slide_id: str):
    """Return tile coordinates + HPC labels + heatmap probs for a slide (JSON)."""
    slide_id = slide_id.strip().upper()
    eng = _get_engine()
    q = text("""
        SELECT
            tc.slide_tile, tc.slides, tc.tiles,
            tc.col, tc.row,
            tc.x_5x, tc.y_5x, tc.x_native, tc.y_native,
            tr.hpc_id, tr.image_index AS h5_index,
            hd.inflammation, hd.necrosis, hd.malignant
        FROM tile_coordinates tc
        LEFT JOIN tile_registry tr
          ON UPPER(tr.slide_tile) = UPPER(tc.slide_tile)
        LEFT JOIN hpc_dictionary hd
          ON hd.hpc_id = tr.hpc_id
        WHERE UPPER(tc.slides) = :slide_id
    """)
    df = pd.read_sql(q, eng, params={"slide_id": slide_id})
    df.columns = df.columns.astype(str).str.strip()

    if "slide_tile" in df.columns:
        df["slide_tile"] = df["slide_tile"].astype(str).str.strip().str.upper()

    # Merge heatmap probs
    if _heatmap_probs is not None and "slide_tile" in df.columns:
        df = df.merge(_heatmap_probs, on="slide_tile", how="left")

    # Replace NaN with None for JSON
    df = df.where(df.notna(), None)
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/slide/{slide_id}/adjacency")
def slide_adjacency(slide_id: str):
    slide_id = slide_id.strip().upper()
    eng = _get_engine()
    q = text("""
        SELECT tc.slide_tile, tc.x_native, tc.y_native, tr.hpc_id
        FROM tile_coordinates tc
        LEFT JOIN tile_registry tr ON UPPER(tr.slide_tile) = UPPER(tc.slide_tile)
        WHERE UPPER(tc.slides) = :slide_id
    """)
    df = pd.read_sql(q, eng, params={"slide_id": slide_id})
    if df.empty:
        return {"pair_edge_counts": {}, "tile_neighbor_pairs": {}}
    pair_counts, tile_pairs = _compute_adjacency(df)
    return {"pair_edge_counts": pair_counts, "tile_neighbor_pairs": tile_pairs}


@app.get("/hpc/{hpc_id}/info")
def hpc_info(hpc_id: int):
    eng = _get_engine()
    with eng.connect() as conn:
        base = conn.execute(
            text("SELECT * FROM hpc_dictionary WHERE hpc_id = :h LIMIT 1"), {"h": str(hpc_id)}
        ).fetchone()
        if not base:
            raise HTTPException(404, f"HPC {hpc_id} not found")
        result = dict(base._mapping)

        mal = conn.execute(
            text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :h LIMIT 1"), {"h": str(hpc_id)}
        ).fetchone()
        non = conn.execute(
            text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :h LIMIT 1"), {"h": str(hpc_id)}
        ).fetchone()

    result["malignant_details"] = dict(mal._mapping) if mal else None
    result["non_malignant_details"] = dict(non._mapping) if non else None
    return result


@app.get("/hpc/{hpc_id}/survival")
def hpc_survival(hpc_id: int):
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM hpc_survival_analysis WHERE hpc_id = :h LIMIT 1"),
            {"h": str(hpc_id)}
        ).fetchone()
    if not row:
        raise HTTPException(404, f"No survival data for HPC {hpc_id}")
    return dict(row._mapping)


@app.get("/tile_image/{slide_tile}")
def tile_image_by_key(slide_tile: str, quality: int = Query(85)):
    """Return the H5-backed tile image for a slide_tile key."""
    slide_tile = slide_tile.strip().upper()
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT image_index FROM tile_registry WHERE UPPER(slide_tile) = :st LIMIT 1"),
            {"st": slide_tile}
        ).fetchone()
    if not row or row.image_index is None:
        raise HTTPException(404, f"No H5 index for {slide_tile}")

    idx = int(row.image_index)
    f = _get_h5()
    ds = f["train_img"]
    if idx < 0 or idx >= ds.shape[0]:
        raise HTTPException(400, f"Index {idx} out of range")

    arr = _to_uint8(ds[idx])
    from PIL import Image as PILImage
    if arr.ndim == 2:
        img = PILImage.fromarray(arr)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        img = PILImage.fromarray(arr if arr.shape[-1] != 1 else arr[:, :, 0])
    else:
        raise HTTPException(500, f"Unexpected tile shape {arr.shape}")
    return _jpeg_response(_img_to_jpeg_bytes(img, quality))


class QueryRequest(BaseModel):
    query: str
    slide_id: Optional[str] = None


@app.post("/query")
def handle_query(req: QueryRequest):
    """Full NL query pipeline — returns structured JSON answer.

    This mirrors the fetch_answer_from_db + handle_* logic from app_v21,
    but returns JSON instead of Streamlit markdown.
    For a first version we keep the DB query logic server-side and
    return the raw structured text that the Streamlit client can display.
    """
    from query_planner import build_query_plan
    plan = build_query_plan(req.query)

    result = {
        "plan": plan,
        "slide_id": req.slide_id,
        "structured_answer": None,
    }

    # For now, return the plan — the Streamlit client still runs
    # fetch_answer_from_db locally for the full NL pipeline.
    # In Phase 2 you move that logic here too.
    return result


# ---------------------------------------------------------------------------
# Run with:  uvicorn tile_server_v2_:app --host 0.0.0.0 --port 8000 --workers 2
# For local dev with autoreload (single worker only — Uvicorn doesn't
# support reload + multiple workers):  UVICORN_RELOAD=true python tile_server_v2_.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # "tile_server_v2_:app" (this file's own module name), not "tile_server:app"
    # — that used to point at a different, older file (tile_server.py) with
    # none of the upload/dataset-job/Slurm endpoints. Under reload or
    # multiple workers, Uvicorn re-imports the app from this string in a
    # subprocess, so that typo meant reload/multi-worker runs were silently
    # serving stale code, not this file.
    reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    uvicorn.run(
        "tile_server_v2_:app",
        host="0.0.0.0",
        port=8000,
        workers=1 if reload else 2,
        reload=reload,
    )
