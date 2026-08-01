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
import io
import os
import re
import time
import json
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
from datetime import datetime
from tile_cache import TileCache
from tile_mask import run_tissue_detection
from auto_tile_from_mask import tile_slide_from_mask
from slide_naming import slide_id_from_raw_path
from submit_mask_tile_slurm import submit_array as submit_dataset_array
from submit_mask_tile_slurm import submit_packaging_job
from make_hpl_hdf5 import package_slides_to_h5, hpl_h5_output_path
from find_missing_slides import find_missing_slides_detailed
from submit_feature_extraction import (
    submit_feature_extraction_job,
    expected_extraction_output_path,
    HPL_REPO_DIR,
)

# sacct states that mean "still queued or actively running" — anything else
# (COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, ...) is
# terminal. Used to decide when a "start next stage" button should appear.
# COMPLETING is included even though the job's steps have finished — Slurm's
# epilogue (and any of the job's own writes still flushing to a network
# filesystem) may not be done yet, so a retry-guard or tiling_complete check
# that treated COMPLETING as terminal could let a second packaging/extraction
# job start writing the same output file while the first is still finishing.
IN_FLIGHT_SLURM_STATES = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING"}


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


_SACCT_ARRAY_TASK_RE = re.compile(r"^\d+_(\d+)\|(\S+)$")

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
            return []
        print(f"[{','.join(job_ids)}] squeue exited {result.returncode}: "
              f"{(result.stderr or '').strip()[:200]}")
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


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
        match = _SACCT_ARRAY_TASK_RE.match(line.strip())
        if match:
            state = match.group(2)
            counts[state] = counts.get(state, 0) + 1
    if counts:
        return counts

    live_states = _slurm_jobs_live_states(job_ids)
    if live_states is None:
        return None
    for state in live_states:
        counts[state] = counts.get(state, 0) + 1
    return counts


def _get_slurm_job_state(job_id: str) -> str | None:
    """Single (non-array) job's current Slurm state via sacct, e.g. for the
    h5-packaging job.

    Returns None if neither sacct nor squeue could be reached — genuinely
    unknown, callers should not guess. Returns "" only once sacct has no
    record of this job AND squeue also confirms it isn't currently
    queued/running — for an old-enough job that combination means it's
    aged out of Slurm's accounting-DB retention window (long finished),
    not that it never existed. sacct alone returning nothing isn't enough:
    a job submitted moments ago can legitimately have no sacct rows yet
    since slurmdbd syncs on its own schedule, well after squeue would
    already show it live — without the squeue cross-check, a freshly
    retried packaging/extraction job would look "finished long ago" on
    every single poll right after submission, offering another retry
    before the previous one even had a chance to start.
    """
    result = _run_slurm(
        ["sacct", "-j", job_id, "--format=JobID,State", "--parsable2", "--noheader"],
        timeout=15,
    )
    if result is None:
        return None

    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 2 and parts[0] == job_id:
            return parts[1]

    live_states = _slurm_jobs_live_states([job_id])
    if live_states is None:
        return None
    if live_states:
        return live_states[0]
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
    if slurm_state != "COMPLETED" and not (
        slurm_state == "" and output_path.stat().st_size > 0
    ):
        return False
    if validator is not None:
        ok, reason = validator(output_path)
        if not ok:
            print(f"[{output_path}] output rejected as not ready: {reason}")
            return False
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


class DatasetJobRequest(BaseModel):
    dataset_path: str
    max_concurrent: int = 10
    min_tissue: float = 30.0
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
            min_tissue=req.min_tissue,
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


def _start_dataset_submission(
    raw_dir: Path, req: DatasetJobRequest, background_tasks: BackgroundTasks
) -> dict:
    """Create a new slurm_dataset_runs row and kick off the background
    pipeline for it. Shared by POST /dataset-jobs (a fresh submission) and
    the /resume endpoint (a follow-up submission for whatever a previous
    run's slides are still missing) — both are "start a submission for this
    raw_dir with this req," just with req.slide_names populated differently.
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
    req = req.model_copy(update={"dataset_name": dataset_name})

    eng = _get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO slurm_dataset_runs
                    (submission_id, raw_dir, mask_dir, tile_dir, status,
                     is_subset, partition, notify_email, dataset_name)
                VALUES
                    (:submission_id, :raw_dir, :mask_dir, :tile_dir, 'queued',
                     :is_subset, :partition, :notify_email, :dataset_name)
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
            },
        )

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

    resume_req = DatasetJobRequest(
        dataset_path=row["raw_dir"],
        slide_names=missing_slide_ids,
        partition=row["partition"],
        notify_email=row["notify_email"],
        # Re-run into the same folder tiles already live in, not whatever
        # raw_dir.name would resolve to by default — matters if the
        # original submission was given a custom dataset_name.
        dataset_name=dataset_name,
    )
    result = _start_dataset_submission(Path(row["raw_dir"]), resume_req, background_tasks)
    result.update({
        "resumed": True,
        "resumed_from_submission_id": submission_id,
        "missing_slide_count": len(missing_raw_paths),
        "never_attempted_count": len(breakdown["never_attempted"]),
        "corrupt_metadata_count": len(breakdown["corrupt"]),
        "zero_tile_count": len(breakdown["zero_tile"]),
        "total_in_original_manifest": total,
    })
    return result


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


@app.post("/dataset-jobs/{submission_id}/package")
def start_packaging_job(
    submission_id: str,
    allow_incomplete: bool = Query(
        False,
        description="Package even though some slides failed tiling. Switches the Slurm "
                    "dependency from afterok to afterany and accepts a dataset with holes.",
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
        if row["h5_output_path"]:
            dataset_name = Path(row["h5_output_path"]).parent.name
        else:
            dataset_name = _effective_h5_dataset_name(tile_dataset_name, bool(row["is_subset"]))
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
            prior_output = Path(row["h5_output_path"]) if row["h5_output_path"] else None
            prior_state = _get_slurm_job_state(row["h5_job_id"])
            if _job_output_ready(prior_output, prior_state, validator=_validate_h5):
                raise HTTPException(400, "Packaging has already completed for this run.")
            if prior_state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    f"Packaging is already running for this run (Slurm state: {prior_state}).",
                )
            # Otherwise the prior attempt failed/was cancelled/timed out (or its
            # state is unknown) — fall through and let this submit a fresh one.

        manifest_path = Path(row["manifest_path"]) if row["manifest_path"] else None
        if not manifest_path or not manifest_path.is_file():
            raise HTTPException(400, f"Manifest no longer exists on disk: {manifest_path}")

        job_ids = [j for j in row["job_id"].split(",") if j]

        # With afterok, submitting while any tiling task has failed produces a
        # job whose dependency can never be satisfied. --kill-on-invalid-dep
        # makes Slurm kill it rather than queue it forever, but the user would
        # still just see packaging vanish with no explanation. Refuse up front
        # with something actionable instead. Deliberately conservative: only
        # states we positively observed as failures count, so an unreachable
        # sacct (None) or an unparsed state never blocks a legitimate submit.
        if not allow_incomplete:
            tiling_states = _get_slurm_array_state_counts(job_ids) or {}
            failed_states = {
                state: n for state, n in tiling_states.items()
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

        try:
            packaging_result = submit_packaging_job(
                manifest_path=manifest_path,
                tile_dir=Path(row["tile_dir"]),
                dataset_name=dataset_name,
                tile_dataset_name=tile_dataset_name,
                depends_on_job_ids=job_ids,
                allow_incomplete=allow_incomplete,
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
        return {"submission_id": submission_id, **packaging_result}


class PackagingTestRequest(BaseModel):
    sample_size: Optional[int] = None
    slide_names: Optional[list[str]] = None
    random_seed: Optional[int] = None


@app.post("/dataset-jobs/{submission_id}/package-test")
def start_test_packaging_job(submission_id: str, req: PackagingTestRequest):
    """Package only a subset of this run's slides into a separately-named
    test .h5 — sanity-check packaging (and a downstream feature-extraction
    checkpoint) against a handful of slides before committing to a
    multi-hour run over the whole dataset. Deliberately NOT tracked on the
    run's own h5_job_id/h5_output_path — a test run succeeding or failing
    has no bearing on whether the real packaging run is allowed to
    proceed, and vice versa (the retry-guard above only ever looks at
    h5_job_id, which this never touches).
    """
    if not req.sample_size and not req.slide_names:
        raise HTTPException(400, "Provide sample_size or slide_names for a test run.")

    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        if not row["job_id"]:
            # Not status == "submitted" specifically — a cancelled run's tiling
            # batches may well have already finished, and cancelling only flips
            # the status column, leaving the manifest/tiles on disk untouched.
            # job_id existing at all is what actually means tiling was submitted.
            raise HTTPException(400, "Tiling hasn't been submitted yet for this run.")

        manifest_path = Path(row["manifest_path"]) if row["manifest_path"] else None
        if not manifest_path or not manifest_path.is_file():
            raise HTTPException(400, f"Manifest no longer exists on disk: {manifest_path}")

        job_ids = [j for j in row["job_id"].split(",") if j]
        tile_dataset_name = _row_dataset_name(row)
        base_dataset_name = _effective_h5_dataset_name(tile_dataset_name, bool(row["is_subset"]))
        # Signature-suffixed rather than a fixed "_test_sample" name: this
        # endpoint has no DB row to guard against, so the *filename itself*
        # is what has to keep two different test attempts (different
        # sample_size/slide_names/random_seed) from clobbering each other's
        # output, and keeps a re-run of the exact same attempt idempotent
        # (same signature -> same path -> caught by the Slurm lookup below)
        # rather than piling up duplicate jobs.
        signature = _attempt_signature(
            "package-test", submission_id, req.sample_size,
            sorted(req.slide_names or []), req.random_seed,
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
                random_seed=req.random_seed,
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
        except Exception as e:
            raise HTTPException(500, f"Failed to submit test packaging job: {e}")

        return {"submission_id": submission_id, **result}


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
            if _job_output_ready(prior_output, prior_state):
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
        except NotADirectoryError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to submit feature extraction job: {e}")

        _update_dataset_run(
            submission_id,
            extraction_job_id=result.get("extraction_job_id"),
            extraction_output_path=result.get("expected_output_path"),
            extraction_checkpoint=checkpoint,
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
            if _job_output_ready(existing_output, existing_state):
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
        except NotADirectoryError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to submit test feature extraction job: {e}")

        return {"submission_id": submission_id, **result}


@app.get("/dataset-jobs/{submission_id}/extract-features-test-status")
def test_feature_extraction_status(submission_id: str, job_id: str, output_path: str):
    """Status for one ad-hoc test extraction job — same pattern as
    /package-test-status, nothing persisted server-side."""
    state = _get_slurm_job_state(job_id)
    ready = _job_output_ready(Path(output_path), state)
    return {"job_id": job_id, "slurm_state": state, "ready": ready, "output_path": output_path}


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
def list_dataset_jobs():
    """Past/active dataset job submissions, most recent first."""
    eng = _get_engine()
    df = pd.read_sql(
        "SELECT * FROM slurm_dataset_runs ORDER BY submitted_at DESC", eng
    )
    return json.loads(df.to_json(orient="records", date_format="iso"))


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

    if row["extraction_job_id"]:
        ext_path = Path(row["extraction_output_path"]) if row["extraction_output_path"] else None
        ext_state = _get_slurm_job_state(row["extraction_job_id"])
        base["extraction_ready"] = _job_output_ready(ext_path, ext_state)
        base["extraction_slurm_state"] = ext_state

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
