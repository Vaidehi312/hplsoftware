"""
FastAPI Tile Server — runs on HPCC near the .svs files and PostgreSQL.

Endpoints
---------
GET  /health                         → liveness check
GET  /slides                         → list all slide IDs
GET  /slide/{slide_id}/info          → slide metadata (dimensions, levels, mpp)
GET  /slide/{slide_id}/thumbnail     → whole-slide JPEG preview
GET  /slide/{slide_id}/tile          → single tile at (level, x, y, w, h)
GET  /slide/{slide_id}/region        → arbitrary region in native coords
GET  /slide/{slide_id}/tiles_meta    → tile_coordinates + tile_registry + hpc join
GET  /slide/{slide_id}/adjacency     → precomputed adjacency pairs
GET  /hpc/{hpc_id}/info              → HPC dictionary + malignant/non-malignant details
GET  /hpc/{hpc_id}/survival          → survival analysis row
POST /query                          → full NL query → structured answer
GET  /tile_image/{slide_tile}        → H5-backed tile image by slide_tile key
"""

import io
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import openslide
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from tile_cache import TileCache

# ---------------------------------------------------------------------------
# Configuration — edit these to match your HPCC environment
# ---------------------------------------------------------------------------
DB_USER = os.getenv("DB_USER", "vpandya")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hpl_kb")

WSI_ROOT = os.getenv("WSI_ROOT", "/hpc-home/home/users/vpandya/long-term-scratch/tcga_wsi")
H5_PATH = os.getenv("H5_PATH", "/hpc-home/home/users/vpandya/long-term-scratch/Vaidehi/TCGA/hdf5_TCGA_LUAD_5x_he_train_tiles.h5")

CACHE_DIR = os.getenv("TILE_CACHE_DIR", "/tmp/hpc_tile_cache")
_dataset_config_map: dict[str, dict] = {}
_wsi_meta_map: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Globals initialised at startup
# ---------------------------------------------------------------------------
engine = None
cache: TileCache = None
_h5_handle = None
_wsi_handles: dict[str, openslide.OpenSlide] = {}
_wsi_map: dict[str, str] = {}  # slide_id → hpc_path on HPCC
_heatmap_probs: pd.DataFrame | None = None


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
    


def _load_wsi_metadata():
    global _wsi_meta_map
    eng = _get_engine()
    df = pd.read_sql("""
        SELECT slide_id, dataset_id, mpp_x, mpp_y
        FROM wsi_metadata
    """, eng)
    df["slide_id"] = df["slide_id"].astype(str).str.strip().str.upper()
    df["dataset_id"] = df["dataset_id"].astype(str).str.strip().str.upper()
    _wsi_meta_map = df.set_index("slide_id")[["dataset_id", "mpp_x", "mpp_y"]].to_dict(orient="index")


def _load_dataset_config():
    global _dataset_config_map
    eng = _get_engine()
    df = pd.read_sql("""
        SELECT dataset_id, target_mpp, tile_size_5x_px
        FROM dataset_config
    """, eng)
    df["dataset_id"] = df["dataset_id"].astype(str).str.strip().str.upper()
    _dataset_config_map = df.set_index("dataset_id")[["target_mpp", "tile_size_5x_px"]].to_dict(orient="index")


def _get_tile_size_native_for_slide(slide_id: str) -> int:
    slide_id = slide_id.strip().upper()

    meta = _wsi_meta_map.get(slide_id)
    if not meta:
        raise HTTPException(404, f"No metadata found for slide {slide_id}")

    cfg = _get_dataset_config_for_slide(slide_id)

    mpp_x = meta.get("mpp_x")
    if mpp_x is None or pd.isna(mpp_x):
        raise HTTPException(500, f"No mpp_x found for slide {slide_id}")

    target_mpp = cfg.get("target_mpp")
    tile_size_5x_px = cfg.get("tile_size_5x_px")

    if target_mpp is None or tile_size_5x_px is None:
        raise HTTPException(500, f"Incomplete dataset config for slide {slide_id}")

    return int(round(float(tile_size_5x_px) * (float(target_mpp) / float(mpp_x))))

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


# ---------------------------------------------------------------------------
# Grid / adjacency helpers (moved from app_v21)
# ---------------------------------------------------------------------------

def _grid_xy_from_df(df_slide: pd.DataFrame):
    df = df_slide.copy()
    df["x_native"] = pd.to_numeric(df["x_native"], errors="coerce")
    df["y_native"] = pd.to_numeric(df["y_native"], errors="coerce")
    df = df.dropna(subset=["x_native", "y_native"])

    xs = sorted(df["x_native"].unique())
    ys = sorted(df["y_native"].unique())

    x_to_gx = {x: i for i, x in enumerate(xs)}
    y_to_gy = {y: i for i, y in enumerate(ys)}

    df["gx"] = df["x_native"].map(x_to_gx)
    df["gy"] = df["y_native"].map(y_to_gy)
    return df


def _neighbors_8(gx, gy):
    return [
        (gx - 1, gy - 1), (gx, gy - 1), (gx + 1, gy - 1),
        (gx - 1, gy),                     (gx + 1, gy),
        (gx - 1, gy + 1), (gx, gy + 1), (gx + 1, gy + 1),
    ]
    
    
def _get_dataset_config_for_slide(slide_id: str) -> dict:
    slide_id = slide_id.strip().upper()

    meta = _wsi_meta_map.get(slide_id)
    if not meta:
        raise HTTPException(404, f"No metadata found for slide {slide_id}")

    dataset_id = meta.get("dataset_id")
    if not dataset_id:
        raise HTTPException(500, f"No dataset_id found for slide {slide_id}")

    cfg = _dataset_config_map.get(str(dataset_id).strip().upper())
    if not cfg:
        raise HTTPException(500, f"No dataset_config found for dataset_id {dataset_id}")

    return cfg



def _compute_adjacency(df_slide: pd.DataFrame):
    df2 = df_slide.copy()
    df2["hpc_id"] = pd.to_numeric(df2["hpc_id"], errors="coerce")
    df2 = df2.dropna(subset=["hpc_id", "x_native", "y_native"])
    df2["hpc_id"] = df2["hpc_id"].astype(int)
    id_col = "slide_tile" if "slide_tile" in df2.columns else "tiles"

    df2 = _grid_xy_from_df(df2)

    pos_to_row = {}
    for _, r in df2.iterrows():
        pos_to_row[(int(r["gx"]), int(r["gy"]))] = r

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

    serialisable = {}
    for pair_key, sets in tile_has_neighbor_pair.items():
        k = f"{pair_key[0]}_{pair_key[1]}"
        serialisable[k] = {
            "a_touch": sorted(sets["a_touch"]),
            "b_touch": sorted(sets["b_touch"]),
        }

    pair_counts = {f"{a}_{b}": cnt for (a, b), cnt in pair_edge_counts.items()}
    return pair_counts, serialisable

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_wsi_map()
    _load_wsi_metadata()
    _load_dataset_config()
    _load_heatmap_probs()
    yield
    global _h5_handle
    if _h5_handle:
        _h5_handle.close()
    for s in _wsi_handles.values():
        s.close()

app = FastAPI(title="HPC Tile Server", lifespan=lifespan)
cache = TileCache(cache_dir=CACHE_DIR)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "slides_loaded": len(_wsi_map)}


@app.get("/slides")
def list_slides():
    return {"slides": sorted(_wsi_map.keys())}


@app.get("/slide/{slide_id}/info")
def slide_info(slide_id: str):
    slide_id = slide_id.strip().upper()
    slide = _open_slide(slide_id)
    dims = slide.level_dimensions
    meta = _wsi_meta_map.get(slide_id, {})
    cfg = _get_dataset_config_for_slide(slide_id)

    tile_size_native = _get_tile_size_native_for_slide(slide_id)

    return {
        "slide_id": slide_id,
        "level_count": slide.level_count,
        "level_dimensions": [{"width": w, "height": h} for w, h in dims],
        "mpp_x": meta.get("mpp_x") or slide.properties.get("openslide.mpp-x"),
        "mpp_y": meta.get("mpp_y") or slide.properties.get("openslide.mpp-y"),
        "dataset_id": meta.get("dataset_id"),
        "target_mpp": cfg.get("target_mpp"),
        "tile_size_5x_px": cfg.get("tile_size_5x_px"),
        "tile_size_native": tile_size_native,
        "vendor": slide.properties.get("openslide.vendor"),
        "objective_power": slide.properties.get("openslide.objective-power"),
    }
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
    w: int = Query(..., description="Native width"),
    h: int = Query(..., description="Native height"),
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


# @app.get("/tile_image/{slide_tile}")
# def tile_image_by_key(slide_tile: str, quality: int = Query(85)):
#     """Return the H5-backed tile image for a slide_tile key."""
#     slide_tile = slide_tile.strip().upper()
#     eng = _get_engine()
#     with eng.connect() as conn:
#         row = conn.execute(
#             text("SELECT image_index FROM tile_registry WHERE UPPER(slide_tile) = :st LIMIT 1"),
#             {"st": slide_tile}
#         ).fetchone()
#     if not row or row.image_index is None:
#         raise HTTPException(404, f"No H5 index for {slide_tile}")

#     idx = int(row.image_index)
#     f = _get_h5()
#     ds = f["train_img"]
#     if idx < 0 or idx >= ds.shape[0]:
#         raise HTTPException(400, f"Index {idx} out of range")

#     arr = _to_uint8(ds[idx])
#     from PIL import Image as PILImage
#     if arr.ndim == 2:
#         img = PILImage.fromarray(arr)
#     elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
#         img = PILImage.fromarray(arr if arr.shape[-1] != 1 else arr[:, :, 0])
#     else:
#         raise HTTPException(500, f"Unexpected tile shape {arr.shape}")
#     return _jpeg_response(_img_to_jpeg_bytes(img, quality))




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
    ds = f["img"]

    if idx < 0 or idx >= ds.shape[0]:
        raise HTTPException(400, f"Index {idx} out of range")

    raw = ds[idx]
    print(f"[tile_image] slide_tile={slide_tile} idx={idx}")
    print(f"[tile_image] raw shape={raw.shape}, dtype={raw.dtype}")

    arr = _to_uint8(raw)
    print(f"[tile_image] after _to_uint8 shape={arr.shape}, dtype={arr.dtype}")

    from PIL import Image as PILImage

    if arr.ndim == 2:
        img = PILImage.fromarray(arr)

    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
            print(f"[tile_image] transposed to shape={arr.shape}")

        if arr.shape[-1] == 1:
            img = PILImage.fromarray(arr[:, :, 0])
        elif arr.shape[-1] in (3, 4):
            img = PILImage.fromarray(arr)
        else:
            raise HTTPException(500, f"Unexpected channel dimension for tile shape {arr.shape}")

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
