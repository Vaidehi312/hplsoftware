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
GET  /dataset-jobs/{job_id}/status   → live Slurm + per-slide tiling status for a dataset job
POST /query                          → full NL query → structured answer
GET  /tile_image/{slide_tile}        → H5-backed tile image by slide_tile key
"""

import io
import os
import re
import time
import json
import subprocess
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

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
UPLOAD_THUMB_DIR = UPLOAD_ROOT / "thumbnails"
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
        mask_result = run_tissue_detection(
            slide_path=raw_path,
            output_dir=str(TISSUE_MASK_DIR),
        )

        _set_processing_status(slide_id, "tiling")
        tile_slide_from_mask(
            slide_path=raw_path,
            mask_path=mask_result["mask_path"],
            output_dir=str(PROCESSED_TILES_DIR),
            min_tissue_percent=MIN_TISSUE_PERCENT,
        )

        _set_processing_status(slide_id, "done")
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


def _get_slurm_array_states(job_id: str) -> dict[int, str]:
    """array task index -> Slurm state, via sacct. Empty dict if sacct is
    unavailable or the job has aged out of accounting history — callers
    should treat that as "unknown", not "nothing is running".
    """
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,State", "--parsable2", "--noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[{job_id}] sacct unavailable: {e}")
        return {}

    states: dict[int, str] = {}
    for line in result.stdout.splitlines():
        match = _SACCT_ARRAY_TASK_RE.match(line.strip())
        if match:
            states[int(match.group(1))] = match.group(2)
    return states


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

    internal_id = str(uuid.uuid4())
    safe_filename = original_filename.replace(" ", "_")

    UPLOAD_RAW_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_THUMB_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    save_path = UPLOAD_RAW_DIR / f"{safe_user_slide_id}_{internal_id}_{safe_filename}"

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
    thumbnail_path = None
    metadata_path = None
    status = "uploaded"
    try:
        slide = openslide.OpenSlide(str(save_path))

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
        
        w0, h0 = slide.level_dimensions[0]
        max_width = 1500
        thumb_height = int(h0 * (max_width / w0))
        thumb = slide.get_thumbnail((max_width, thumb_height)).convert("RGB")
        thumbnail_path = UPLOAD_THUMB_DIR / f"{safe_user_slide_id}.jpg"
        thumb.save(thumbnail_path, format="JPEG", quality=85)

        metadata_path = UPLOAD_METADATA_DIR / f"{safe_user_slide_id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(slide_info_payload, f, indent=2)


        slide.close()

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
        "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
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
    dataset_name: str
    max_concurrent: int = 10
    min_tissue: float = 30.0


@app.get("/dataset-roots")
def list_dataset_roots():
    """Datasets available to submit — populates the UI's picker."""
    return {"root": str(LONG_TERM_SCRATCH), "datasets": _list_dataset_roots()}


@app.post("/dataset-jobs")
def create_dataset_job(req: DatasetJobRequest):
    """Submit a dataset-wide masking+tiling Slurm array job.

    dataset_name must be one of /dataset-roots's own results — re-checked
    here server-side rather than trusting whatever the client sent, since
    this triggers a real sbatch submission against the user's allocation.
    """
    allowed = _list_dataset_roots()
    if req.dataset_name not in allowed:
        raise HTTPException(
            400,
            f"'{req.dataset_name}' is not an allowed dataset root. Allowed: {allowed}",
        )

    raw_dir = LONG_TERM_SCRATCH / req.dataset_name

    try:
        result = submit_dataset_array(
            raw_dir=raw_dir,
            mask_dir=TISSUE_MASK_DIR,
            tile_dir=PROCESSED_TILES_DIR,
            max_concurrent=req.max_concurrent,
            min_tissue=req.min_tissue,
        )
    except (NotADirectoryError, FileNotFoundError, RuntimeError) as e:
        raise HTTPException(400, str(e))

    if result.get("job_id"):
        eng = _get_engine()
        with eng.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO slurm_dataset_runs
                        (job_id, raw_dir, manifest_path, mask_dir, tile_dir, total_slides)
                    VALUES
                        (:job_id, :raw_dir, :manifest_path, :mask_dir, :tile_dir, :total_slides)
                """),
                {
                    "job_id": result["job_id"],
                    "raw_dir": result["raw_dir"],
                    "manifest_path": result["manifest_path"],
                    "mask_dir": result["mask_dir"],
                    "tile_dir": result["tile_dir"],
                    "total_slides": result["slides_found"],
                },
            )

    return result


@app.get("/dataset-jobs")
def list_dataset_jobs():
    """Past/active dataset job submissions, most recent first."""
    eng = _get_engine()
    df = pd.read_sql(
        "SELECT * FROM slurm_dataset_runs ORDER BY submitted_at DESC", eng
    )
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/dataset-jobs/{job_id}/status")
def dataset_job_status(job_id: str):
    """Live progress for one dataset job: Slurm state counts (via sacct)
    cross-referenced against each slide's actual tiling_summary.json, so a
    slide that "completed" with zero saved tiles shows up distinctly from
    one that genuinely succeeded.
    """
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM slurm_dataset_runs WHERE job_id = :job_id"),
            {"job_id": job_id},
        ).mappings().fetchone()

    if not row:
        raise HTTPException(404, f"No dataset job found with id {job_id}")

    manifest_path = Path(row["manifest_path"])
    tile_dir = Path(row["tile_dir"])

    slide_paths: list[Path] = []
    if manifest_path.exists():
        slide_paths = [
            Path(line.strip())
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        ]

    slurm_states = _get_slurm_array_states(job_id)
    slurm_state_counts: dict[str, int] = {}
    for state in slurm_states.values():
        slurm_state_counts[state] = slurm_state_counts.get(state, 0) + 1

    succeeded, zero_tile, not_attempted = [], [], []
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

    return {
        "job_id": job_id,
        "total_slides": row["total_slides"],
        "slurm_state_counts": slurm_state_counts,
        "attempted": len(succeeded) + len(zero_tile),
        "succeeded": len(succeeded),
        "zero_tile_slides": zero_tile,
        "not_yet_attempted": len(not_attempted),
    }


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
# Run with:  uvicorn tile_server:app --host 0.0.0.0 --port 8000 --workers 2
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tile_server:app", host="0.0.0.0", port=8000, workers=2, reload=True)
