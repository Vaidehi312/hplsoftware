import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from textblob import TextBlob
import re
import numpy as np
import io
import h5py
import openslide
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_image_zoom import image_zoom
import colorsys
import hashlib
from query_planner import build_query_plan
from plan_query import save_query_plan_to_file
from llm_explainer import explain
from collections import defaultdict
import os
import time
import json

# #region agent log
_DEBUG_LOG_PATH = "/Users/vaidehipandya/Desktop/Work_cursor/.cursor/debug-f5a3eb.log"
def _dlog(hypothesis_id, location, message, data=None):
    try:
        payload = {"sessionId": "f5a3eb", "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data or {}, "timestamp": int(time.time() * 1000)}
        open(_DEBUG_LOG_PATH, "a").write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion



st.set_page_config(page_title="HPC Chatbot", page_icon="💬", layout="wide")
st.title("🧠 HPC Chatbot with Image Upload")



DB_USER = "vpandya"
DB_PASS = ""
DB_HOST = "127.0.0.1"
DB_PORT = "5433"
DB_NAME = "hpl_kb"

st.write("DB_HOST =", DB_HOST)
st.write("DB_PORT =", DB_PORT)
st.write("DB_USER =", DB_USER)
st.write("DB_NAME =", DB_NAME)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True
)


HPC_WSI_ROOT = "/hpc-home/home/users/vpandya/long-term-scratch/tcga_wsi"
LOCAL_WSI_MOUNT = "/Users/vaidehipandya/mnt/tcga_wsi"

def resolve_wsi_path(p: str) -> str:
    # #region agent log
    _dlog("B", "app_v21.py:resolve_wsi_path:entry", "resolve_wsi_path entry", {"raw_path_preview": (p or "")[:80]})
    t0 = time.time()
    # #endregion
    p = (p or "").strip()

    # avoid urls
    if p.lower().startswith("sftp://"):
        raise ValueError("WSI path is an SFTP url. Store the HPC filesystem path instead.")

    # map HPC path to local mount path
    mapped = p.startswith(HPC_WSI_ROOT + "/")
    if mapped:
        p = p.replace(HPC_WSI_ROOT, LOCAL_WSI_MOUNT, 1)
    # #region agent log
    _dlog("B", "app_v21.py:resolve_wsi_path:exit", "resolve_wsi_path exit", {"mapped": mapped, "duration_ms": round((time.time() - t0) * 1000, 2), "resolved_preview": p[:80]})
    # #endregion
    return p
# TEST CONNECTION
with engine.connect() as conn:
    st.success(conn.execute(text("select 1")).scalar())

# ------------------------------
# Shared constants / paths
# ------------------------------
TILE_SIZE_5X = 224
SCALE = 1.8 / 0.252
TILE_SIZE_NATIVE = int(TILE_SIZE_5X * SCALE)



h5_path = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"

# Heatmap probability table (wide) – loaded once and merged per-slide in load_tile_coords_for_slide()
HEATMAP_PROB_CSV = "/Users/vaidehipandya/Desktop/Work/Heatmap/tile_hpc_probabilities_wide.csv"

@st.cache_data(show_spinner=False)
def load_heatmap_probabilities_wide() -> pd.DataFrame:
    """Load the wide probability table with columns like p_hpc_31.

    This is merged into per-slide tile coords inside `load_tile_coords_for_slide()`.
    """
    # #region agent log
    t0 = time.time()
    # #endregion
    df_probs = pd.read_csv(HEATMAP_PROB_CSV)
    df_probs.columns = df_probs.columns.astype(str).str.strip()

    if "slide_tile" not in df_probs.columns:
        raise KeyError("Heatmap probability CSV must include a 'slide_tile' column")

    df_probs["slide_tile"] = df_probs["slide_tile"].astype(str).str.strip().str.upper()
    keep_cols = ["slide_tile"] + [c for c in df_probs.columns if c.startswith("p_hpc_")]
    # #region agent log
    _dlog("D", "app_v21.py:load_heatmap_probabilities_wide:exit", "load_heatmap_probabilities_wide done", {"duration_ms": round((time.time() - t0) * 1000, 2), "row_count": len(df_probs)})
    # #endregion
    return df_probs[keep_cols].copy()

# Global cache for per-slide merge
try:
    df_probs = load_heatmap_probabilities_wide()
except Exception as e:
    df_probs = None
    st.warning(f"Heatmap probability table not loaded: {e}")


# ------------------------------
# Adjacency helpers (used by compute_adjacency_and_cooccurrence)
# ------------------------------

def _grid_xy(x_native, y_native):
    gx = int(float(x_native) // float(TILE_SIZE_NATIVE))
    gy = int(float(y_native) // float(TILE_SIZE_NATIVE))
    return gx, gy


def _neighbors_8(gx, gy):
    return [
        (gx - 1, gy - 1), (gx, gy - 1), (gx + 1, gy - 1),
        (gx - 1, gy),                 (gx + 1, gy),
        (gx - 1, gy + 1), (gx, gy + 1), (gx + 1, gy + 1),
    ]




def normalize_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # Fix merge suffixes for slide_tile too
    if "slide_tile" not in df.columns:
        if "slide_tile_x" in df.columns:
            df = df.rename(columns={"slide_tile_x": "slide_tile"})
        elif "slide_tile_y" in df.columns:
            df = df.rename(columns={"slide_tile_y": "slide_tile"})

    # Normalize column names if merges created suffixes
    if "slides" not in df.columns:
        if "slides_x" in df.columns:
            df = df.rename(columns={"slides_x": "slides"})
        elif "slides_y" in df.columns:
            df = df.rename(columns={"slides_y": "slides"})

    if "tiles" not in df.columns:
        if "tiles_x" in df.columns:
            df = df.rename(columns={"tiles_x": "tiles"})
        elif "tiles_y" in df.columns:
            df = df.rename(columns={"tiles_y": "tiles"})

    # Standardize slide_tile first if present
    if "slide_tile" in df.columns:
        df["slide_tile"] = df["slide_tile"].astype(str).str.strip().str.upper()

        # Optional: reconstruct slides/tiles if missing
        if "slides" not in df.columns:
            df["slides"] = df["slide_tile"].str.split("_", n=1).str[0]
        if "tiles" not in df.columns:
            df["tiles"] = df["slide_tile"].str.split("_", n=1).str[1]

    # Normalize values
    if "slides" in df.columns:
        df["slides"] = df["slides"].astype(str).str.strip().str.upper()

    if "tiles" in df.columns:
        df["tiles"] = df["tiles"].astype(str).str.strip()

    # Ensure slide_tile exists if not present
    if "slide_tile" not in df.columns and ("slides" in df.columns) and ("tiles" in df.columns):
        df["slide_tile"] = (df["slides"] + "_" + df["tiles"]).astype(str).str.strip().str.upper()

    return df


def normalize_tile_coords_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # Resolve slide_tile first (merge suffixes)
    if "slide_tile" not in df.columns:
        if "slide_tile_x" in df.columns:
            df = df.rename(columns={"slide_tile_x": "slide_tile"})
        elif "slide_tile_y" in df.columns:
            df = df.rename(columns={"slide_tile_y": "slide_tile"})

    # Resolve slides
    if "slides" not in df.columns:
        if "slides_x" in df.columns:
            df = df.rename(columns={"slides_x": "slides"})
        elif "slides_y" in df.columns:
            df = df.rename(columns={"slides_y": "slides"})

    # Resolve tiles
    if "tiles" not in df.columns:
        if "tiles_x" in df.columns:
            df = df.rename(columns={"tiles_x": "tiles"})
        elif "tiles_y" in df.columns:
            df = df.rename(columns={"tiles_y": "tiles"})

    # Resolve x_native, y_native
    if "x_native" not in df.columns:
        if "x_native_x" in df.columns:
            df = df.rename(columns={"x_native_x": "x_native"})
        elif "x_native_y" in df.columns:
            df = df.rename(columns={"x_native_y": "x_native"})

    if "y_native" not in df.columns:
        if "y_native_x" in df.columns:
            df = df.rename(columns={"y_native_x": "y_native"})
        elif "y_native_y" in df.columns:
            df = df.rename(columns={"y_native_y": "y_native"})

    # Standardize slide_tile if present
    if "slide_tile" in df.columns:
        df["slide_tile"] = df["slide_tile"].astype(str).str.strip().str.upper()

        # Optional backfill
        if "slides" not in df.columns:
            df["slides"] = df["slide_tile"].str.split("_", n=1).str[0]
        if "tiles" not in df.columns:
            df["tiles"] = df["slide_tile"].str.split("_", n=1).str[1]

    # Standardize values
    if "slides" in df.columns:
        df["slides"] = df["slides"].astype(str).str.strip().str.upper()
    if "tiles" in df.columns:
        df["tiles"] = df["tiles"].astype(str).str.strip()

    # Ensure slide_tile exists
    if "slide_tile" not in df.columns and ("slides" in df.columns) and ("tiles" in df.columns):
        df["slide_tile"] = (df["slides"] + "_" + df["tiles"]).astype(str).str.strip().str.upper()

    return df


def detect_adjacency_intent(q: str):
    """
    Detect queries like:
      Show me tiles where HPC 21 and HPC 27 co occur
      HPC 21 besides HPC 27
      co-occurrence of HPC 31 and HPC 27
    Returns (a, b) or None
    """
    q0 = (q or "").lower()

    keywords = [
        "beside", "besides", "next to", "adjacent", "touching", "near", "around",
        "cooccur", "co-occur", "co occur",
        "cooccurrence", "co-occurrence", "co occurrence",
        "cooccurence"
    ]

    if not any(k in q0 for k in keywords):
        return None

    hpcs = re.findall(r"hpc\s*([0-9]+)", q0)
    if len(hpcs) >= 2:
        a, b = int(hpcs[0]), int(hpcs[1])
        if a != b:
            return a, b

    return None

def detect_single_hpc_adjacency_intent(q: str):
    """
    Detect queries like:
    - 'Which tiles are beside HPC 31'
    - 'Show neighbors of HPC 21'
    Returns: hpc_id or None
    """
    q0 = (q or "").lower()

    adjacency_words = [
        "beside", "besides", "next to", "adjacent", "touching", "near", "around", "neighbors"
    ]

    if not any(w in q0 for w in adjacency_words):
        return None

    hpcs = re.findall(r"hpc\s*([0-9]+)", q0)
    if len(hpcs) == 1:
        return int(hpcs[0])

    return None


def parse_hpc_id(q: str):
    if not q:
        return None
    m = re.search(r"hpc\s*([0-9]+)", q.lower())
    return int(m.group(1)) if m else None

def parse_inflammation(q: str):
    if not q:
        return None
    q0 = (q or "").lower()
    if "marked" in q0:
        return "marked"
    if "mild" in q0 or "moderate" in q0:
        return "mild-moderate"
    if "non" in q0 and "sparse" in q0:
        return "none-sparse"
    return None

def parse_necrosis(q: str):
    if not q:
        return None
    q0 = (q or "").lower()
    if "universal" in q0:
        return "universal"
    if "some" in q0:
        return "some"
    if "none" in q0:
        return "none"
    return None

def parse_highlight_mode(q: str):
    q0 = (q or "").lower()
    if "heatmap" in q0:
        return "Heatmap"
    if "inflammation" in q0:
        return "Inflammation"
    if "necrosis" in q0:
        return "Necrosis"
    if "malignant" in q0:
        return "Malignant"
    if "adjacent" in q0 or "beside" in q0 or "cooccur" in q0:
        return "Adjacency"
    if "hpc" in q0 or "cluster" in q0:
        return "HPC clusters"
    return None

def apply_chat_query_to_wsi_state(prompt: str, slide_id: str):
    det = detect_entity_patterns(prompt)
    if det and det.get("slide"):
        slide_id = str(det["slide"]).strip().upper()
        st.session_state.active_slide = slide_id

    if not prompt:
        return

    slide_id = str(slide_id).strip().upper()

    # reset per-query filters
    st.session_state.query_inflammation = None
    st.session_state.query_necrosis = None

    # clear adjacency unless this prompt is adjacency
    st.session_state.adj_hpc_a = None
    st.session_state.adj_hpc_b = None
    st.session_state.adj_tile_sets = None

    df = load_tile_coords_for_slide(slide_id).copy()

    if df.empty:
        st.warning(f"No tile coords in DB for slide {slide_id}")
        return

    if "slides" not in df.columns:
        raise KeyError(f"'slides' missing in per-slide df. Columns are: {df.columns.tolist()}")

    if "slide_tile" not in df.columns:
        raise KeyError(f"'slide_tile' missing in per-slide df. Columns are: {df.columns.tolist()}")


    pair = detect_adjacency_intent(prompt)
    single = detect_single_hpc_adjacency_intent(prompt)

    # adjacency needs these cols
    if (pair or single is not None):
        needed_cols = {"slide_tile", "x_native", "y_native", "hpc_id"}
        missing = needed_cols - set(df.columns)
        if missing:
            st.warning(f"Cannot compute adjacency, missing columns: {missing}")
            return

    if pair:
        a, b = pair
        st.session_state.highlight_mode = "Adjacency"
        st.session_state.adj_hpc_a = a
        st.session_state.adj_hpc_b = b

        pair_edge_counts, tile_has_neighbor_pair = compute_adjacency_cached_for_slide(slide_id)

        p = (a, b) if a < b else (b, a)
        st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(p, {"a_touch": set(), "b_touch": set()})
        return

    if single is not None:
        st.session_state.highlight_mode = "Adjacency"

        pair_edge_counts, tile_has_neighbor_pair = compute_adjacency_cached_for_slide(slide_id)

        candidates = [((a, b), cnt) for (a, b), cnt in pair_edge_counts.items() if a == single or b == single]

        if candidates:
            (a_sel, b_sel), _ = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
            st.session_state.adj_hpc_a = a_sel
            st.session_state.adj_hpc_b = b_sel

            p = (a_sel, b_sel) if a_sel < b_sel else (b_sel, a_sel)
            st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(p, {"a_touch": set(), "b_touch": set()})
        else:
            st.session_state.adj_tile_sets = {"a_touch": set(), "b_touch": set()}
        return

    # normal parsing
    hpc_id = parse_hpc_id(prompt)
    infl = parse_inflammation(prompt)
    nec = parse_necrosis(prompt)
    mode = parse_highlight_mode(prompt)

    st.session_state.query_inflammation = infl
    st.session_state.query_necrosis = nec

    q0 = (prompt or "").lower()
    if ("heatmap" in q0) and (hpc_id is not None):
        st.session_state.highlight_mode = "Heatmap"
        st.session_state.heat_hpc = int(hpc_id)
        st.session_state.setdefault("heat_min", 0.0)
        st.session_state.setdefault("heat_alpha", 0.6)
        st.session_state.selected_hpc = None
        return

    if hpc_id is not None:
        st.session_state.selected_hpc = hpc_id
        st.session_state.highlight_mode = mode or "HPC clusters"
    elif mode:
        st.session_state.highlight_mode = mode



# -------------------------

def render_hpc_annotation(hpc_id):
    try:
        h = int(hpc_id)
    except (TypeError, ValueError):
        st.warning(f"Invalid HPC ID: {hpc_id}")
        return

    with engine.connect() as conn:
        dict_row = conn.execute(
            text("SELECT * FROM hpc_dictionary WHERE hpc_id = :h LIMIT 1"),
            {"h": h}
        ).fetchone()

        mal_row = conn.execute(
            text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :h LIMIT 1"),
            {"h": h}
        ).fetchone()

        non_row = conn.execute(
            text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :h LIMIT 1"),
            {"h": h}
        ).fetchone()

    st.markdown(f"### 🧠 HPC {h} Annotation")

    if dict_row:
        st.markdown("#### HPC Dictionary Entry")
        for k, v in dict(dict_row._mapping).items():
            st.write(f"- **{k}**: {v}")
    else:
        st.info("No dictionary entry available for this HPC.")

    if mal_row:
        st.markdown("#### Malignant Epithelium Details")
        for k, v in dict(mal_row._mapping).items():
            st.write(f"- **{k}**: {v}")
    elif non_row:
        st.markdown("#### Non Malignant Epithelium Details")
        for k, v in dict(non_row._mapping).items():
            st.write(f"- **{k}**: {v}")
    else:
        st.info("No epithelial phenotype annotations found.")



@st.cache_data(show_spinner=False)
def load_wsi_registry():
    # #region agent log
    t0 = time.time()
    # #endregion
    q = "SELECT slide_id, hpc_path FROM wsi_registry"
    df = pd.read_sql(q, engine)
    df["slide_id"] = df["slide_id"].astype(str).str.strip().str.upper()
    # #region agent log
    _dlog("C", "app_v21.py:load_wsi_registry:exit", "load_wsi_registry done", {"duration_ms": round((time.time() - t0) * 1000, 2), "row_count": len(df)})
    # #endregion
    return dict(zip(df["slide_id"], df["hpc_path"]))

WSI_MAP = load_wsi_registry()
@st.cache_data(show_spinner=False)

def load_tile_coords_for_slide(slide_id: str) -> pd.DataFrame:
    # #region agent log
    t0 = time.time()
    # #endregion
    slide_id = (slide_id or "").strip().upper()
    if not slide_id:
        return pd.DataFrame()

    q = """
        SELECT
            tc.slide_tile,
            tc.slides,
            tc.tiles,
            tc.col,
            tc.row,
            tc.x_5x,
            tc.y_5x,
            tc.x_native,
            tc.y_native,
            tr.hpc_id,
            tr.image_index AS h5_index,
            hd.inflammation,
            hd.necrosis,
            hd.malignant
        FROM tile_coordinates tc
        LEFT JOIN tile_registry tr
          ON UPPER(tr.slide_tile) = UPPER(tc.slide_tile)
        LEFT JOIN hpc_dictionary hd
          ON hd.hpc_id = tr.hpc_id
        WHERE UPPER(tc.slides) = :slide_id
    """

    df = pd.read_sql(text(q), engine, params={"slide_id": slide_id})
    df = normalize_tile_coords_columns(df)

    # Optional: merge heatmap probabilities if df_probs exists later
    if "df_probs" in globals() and isinstance(globals().get("df_probs"), pd.DataFrame):
        probs = globals()["df_probs"].copy()
        if "slide_tile" in probs.columns:
            probs["slide_tile"] = probs["slide_tile"].astype(str).str.strip().str.upper()
            df["slide_tile"] = df["slide_tile"].astype(str).str.strip().str.upper()
            keep_cols = ["slide_tile"] + [c for c in probs.columns if c.startswith("p_hpc_")]
            probs = probs[keep_cols]
            df = df.merge(probs, on="slide_tile", how="left", validate="m:1")

    # #region agent log
    _dlog("C", "app_v21.py:load_tile_coords_for_slide:exit", "load_tile_coords_for_slide done", {"duration_ms": round((time.time() - t0) * 1000, 2), "slide_id": slide_id, "row_count": len(df)})
    # #endregion
    return df





def compute_adjacency_and_cooccurrence(df_slide):
    df2 = df_slide.copy()

    df2["hpc_id"] = pd.to_numeric(df2["hpc_id"], errors="coerce")
    df2 = df2.dropna(subset=["hpc_id", "x_native", "y_native"])
    df2["hpc_id"] = df2["hpc_id"].astype(int)

    # prefer slide_tile if present
    id_col = "slide_tile" if "slide_tile" in df2.columns else "tiles"

    pos_to_row = {}
    for _, r in df2.iterrows():
        gx, gy = _grid_xy(r["x_native"], r["y_native"])

        # avoid silent overwrites
        if (gx, gy) in pos_to_row:
            continue

        pos_to_row[(gx, gy)] = r

    pair_edge_counts = {}
    tile_has_neighbor_pair = {}
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

    return pair_edge_counts, tile_has_neighbor_pair

@st.cache_data(show_spinner=False)
def compute_adjacency_cached_for_slide(slide_id: str):
    df = load_tile_coords_for_slide(slide_id)
    df_min = df[["slide_tile", "x_native", "y_native", "hpc_id"]].copy()
    return compute_adjacency_and_cooccurrence(df_min)


def load_tile_from_h5(index):
    if not h5_path:
        raise ValueError("h5_path is not set")

    with h5py.File(h5_path, "r") as f:
        tile = f["train_img"][index]

    tile = np.squeeze(tile).astype(np.uint8)
    return Image.fromarray(tile)


def load_tile_registry():
    try:
        query = """
            SELECT slides, tiles, slide_tile, hpc_id, image_index
            FROM tile_registry
        """
        df = pd.read_sql(query, engine)

        # normalize keys + slide_tile casing to match the rest of the pipeline
        df = normalize_key_cols(df)

        # enforce stable dtypes
        if "hpc_id" in df.columns:
            df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")
        if "image_index" in df.columns:
            df["image_index"] = pd.to_numeric(df["image_index"], errors="coerce").astype("Int64")

        return df
    except Exception as e:
        st.error(f"Error loading tile registry: {e}")
        return None
    
    
def get_min_adj_df(df):
     # Adjacency identity should be globally unique; use slide_tile not tiles.
    needed = ["slide_tile", "x_native", "y_native", "hpc_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"get_min_adj_df missing columns: {missing}. Available: {df.columns.tolist()}")
    return df[needed].copy()

@st.cache_data(show_spinner=False)
def load_hpc_dictionary():
    try:
        query = """
            SELECT hpc_id, inflammation, necrosis, malignant
            FROM hpc_dictionary
        """
      
        df = pd.read_sql(query, engine)

        # normalize columns + types
        df.columns = df.columns.astype(str).str.strip()
        if "hpc_id" in df.columns:
            df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")

        # (optional) standardize label text
        for c in ["inflammation", "necrosis"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        return df    
    except Exception as e:
        st.error(f"Error loading hpc dictionary: {e}")
        return None
    



# ==================================================
# Load DB tables
# ==================================================
tile_registry = load_tile_registry()
hpc_dict = load_hpc_dictionary()

# def color_for_hpc(hpc_id):
#     # Default color for missing HPC
#     if hpc_id is None:
#         return (255, 0, 0)  # red as RGB

#     h = int(hashlib.md5(str(hpc_id).encode()).hexdigest(), 16)
#     hue = (h % 360) / 360.0

#     r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
#     return int(r * 255), int(g * 255), int(b * 255)


def color_for_hpc(hpc_id):
    if hpc_id is None or pd.isna(hpc_id):
        return (160, 160, 160)  # or keep red if you prefer

    h = int(hashlib.md5(str(int(hpc_id)).encode()).hexdigest(), 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def color_for_inflammation(label):
    if label is None or pd.isna(label) or str(label).strip() == "":
        return (160, 160, 160)  # missing label

    s = str(label).strip().lower()

    mapping = {
        "none-sparse": (80, 200, 120),
        "mild-moderate": (255, 200, 0),
        "marked": (255, 80, 80),
    }
    return mapping.get(s, (160, 160, 160))


def color_for_necrosis(label):
    if label is None or pd.isna(label) or str(label).strip() == "":
        return (160, 160, 160)  # Missing / unknown

    s = str(label).strip().lower()

    if s == "none":
        return (60, 200, 120)     # Green
    if s == "some":
        return (255, 165, 0)      # Orange
    if s == "universal":
        return (200, 40, 40)      # Red

    return (160, 160, 160)


def color_for_malignant(flag):
    # Missing or unknown
    if flag is None or pd.isna(flag):
        return (160, 160, 160)

    if isinstance(flag, (int, np.integer)):
        flag = bool(flag)
    # Postgres bool comes into pandas as True/False.
    # If it comes as string, normalize.
    if isinstance(flag, str):
        s = flag.strip().lower()
        if s in ["true", "t", "1", "yes", "y"]:
            flag = True
        elif s in ["false", "f", "0", "no", "n"]:
            flag = False
        else:
            return (160, 160, 160)

    if bool(flag) is True:
        return (255, 80, 80)     # Malignant
    else:
        return (80, 200, 120)    # Non-malignant

def color_for_adjacency_group(group_name):
    if group_name == "a_touch":
        return (80, 200, 120)
    if group_name == "b_touch":
        return (255, 165, 0)
    return (160, 160, 160)

def rgb_to_css(rgb):
    r, g, b = map(int, rgb)
    return f"rgb({r}, {g}, {b})"



def render_tile_preview(slide_id, limit: int = 30):
    slide_id = (slide_id or "").strip().upper()
    if not slide_id:
        return

    # Helper: robust display normalization (avoid washed-out tiles)
    def _to_uint8(tile_arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(tile_arr)
        arr = np.squeeze(arr)

        # If already uint8, keep as-is
        if arr.dtype == np.uint8:
            return arr

        # Convert to float for scaling
        x = arr.astype(np.float32)

        # If values look like [0, 1], scale directly
        mx = float(np.nanmax(x)) if x.size else 0.0
        mn = float(np.nanmin(x)) if x.size else 0.0
        if 0.0 <= mn and mx <= 1.0:
            x = x * 255.0
            return np.clip(x, 0, 255).astype(np.uint8)

        # Percentile scaling (more stable than per-tile min/max)
        lo = float(np.nanpercentile(x, 1))
        hi = float(np.nanpercentile(x, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # Fallback to simple clip if degenerate
            return np.clip(x, 0, 255).astype(np.uint8)

        x = (x - lo) / (hi - lo) * 255.0
        return np.clip(x, 0, 255).astype(np.uint8)

    with engine.connect() as conn:
        tile_rows = conn.execute(
            text(
                """
                SELECT tiles, image_index
                FROM tile_registry
                WHERE slides ILIKE :slide_id
                ORDER BY image_index ASC
                LIMIT :lim
                """
            ),
            {"slide_id": slide_id, "lim": int(limit)}
        ).fetchall()

    if not tile_rows:
        return

    with st.expander(f"📂 Preview Tiles ({len(tile_rows)} tiles)", expanded=False):
        try:
            with h5py.File(h5_path, "r") as f:
                dataset = f["train_img"]
                n = int(dataset.shape[0])

                cols = st.columns(3)
                for i, row in enumerate(tile_rows):
                    # row.image_index may be numpy/int/None
                    if row.image_index is None:                    
                        continue

                    try:
                        idx = int(row.image_index)
                    except (TypeError, ValueError):
                        continue

                    if idx < 0 or idx >= n:
                        continue

                    tile = dataset[idx]
                    tile_u8 = _to_uint8(tile)

                    # Handle grayscale vs RGB
                    if tile_u8.ndim == 2:
                        img = Image.fromarray(tile_u8)
                    elif tile_u8.ndim == 3 and tile_u8.shape[-1] in (1, 3, 4):
                        if tile_u8.shape[-1] == 1:
                            img = Image.fromarray(tile_u8[:, :, 0])
                        else:
                            img = Image.fromarray(tile_u8)
                    else:
                        # unexpected shape
                        continue
                    
                    cols[i % 3].image(
                        img,
                        caption=f"{row.tiles} (index {idx})",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Error loading tiles: {e}")

def render_tile_info(tile_row, heat_hpc: int | None = None):
    st.subheader("Tile info")

    # Core fields you already have in df
    info = {
        "slide_tile": tile_row.get("slide_tile"),
        "tiles": tile_row.get("tiles"),
        "hpc_id": tile_row.get("hpc_id"),
        "inflammation": tile_row.get("inflammation"),
        "necrosis": tile_row.get("necrosis"),
        "malignant": tile_row.get("malignant"),
        
    }

    # Clean display
    info_df = pd.DataFrame(
        [{"Field": k, "Value": (None if pd.isna(v) else v)} for k, v in info.items()]
    )
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    # If in heatmap mode, also show the probability for the selected heatmap HPC
    if heat_hpc is not None:
        col = f"p_hpc_{int(heat_hpc)}"
        if col in tile_row.index:
            p = tile_row.get(col)
            if p is not None and not pd.isna(p):
                st.metric(f"Heatmap probability for HPC {heat_hpc}", f"{float(p):.4f}")

def show_wsi(slide_id):

    slide_id = (slide_id or "").strip().upper()


    # Pull tile coordinates for THIS slide only (from DB)
    coords_df = load_tile_coords_for_slide(slide_id)

    if coords_df is None or coords_df.empty:
        st.warning(f"No tile coordinates found in DB for slide {slide_id}")
        return

    # --- Session State Initialization ---
    if "selected_tile" not in st.session_state:
        st.session_state.selected_tile = None
    
    if "adj_hpc_a" not in st.session_state:
        st.session_state.adj_hpc_a = None

    if "adj_hpc_b" not in st.session_state:
        st.session_state.adj_hpc_b = None

    if "adj_tile_sets" not in st.session_state:
        st.session_state.adj_tile_sets = None

    if slide_id not in WSI_MAP:
        st.info("No WSI available.")
        return
    # --- Query filter state ---
    if "query_inflammation" not in st.session_state:
        st.session_state.query_inflammation = None

    if "query_necrosis" not in st.session_state:
        st.session_state.query_necrosis = None

    wsi_path_raw = WSI_MAP[slide_id]
    wsi_path = resolve_wsi_path(wsi_path_raw)
    # #region agent log
    t_stat = time.time()
    # #endregion

    # ------------------------------
    # Build df ONCE, early
    # - coords_df already comes from Postgres via `load_tile_coords_for_slide()`.
    # - If you loaded/merged probability columns (p_hpc_*) in `load_tile_coords_for_slide`,
    #   they will already be present here.
    # ------------------------------
    df = coords_df.copy()

    # Keep only the columns we need + any probability columns (p_hpc_*) that happen to exist.
    base_cols = [
        "tiles", "slides", "x_native", "y_native",
        "h5_index", "hpc_id",
        "inflammation", "necrosis", "malignant",
        "slide_tile",
    ]
    prob_cols = [c for c in df.columns if str(c).startswith("p_hpc_")]
    cols_to_keep = [c for c in base_cols if c in df.columns] + prob_cols
    df = df[cols_to_keep].copy()
    


    st.write("WSI path:", wsi_path)
    st.write("WSI path (raw):", wsi_path_raw)
    st.write("WSI path (resolved):", wsi_path)
    st.write("Exists:", os.path.exists(wsi_path))
    st.write("Is file:", os.path.isfile(wsi_path))
    st.write(
        "Size (MB):",
        round(os.path.getsize(wsi_path) / (1024 * 1024), 2) if os.path.exists(wsi_path) else None
    )
    st.write("Extension:", os.path.splitext(wsi_path)[1].lower())
    # #region agent log
    _dlog("B", "app_v21.py:show_wsi:stat", "exists/isfile/getsize on mount", {"duration_ms": round((time.time() - t_stat) * 1000, 2)})
    # #endregion

    if df.empty:
        st.warning(f"No tile coordinates found for slide {slide_id}")
        return

    try:
        if not os.path.isfile(wsi_path):
            st.error(f"WSI file not found via mount: {wsi_path}")
            return

        # #region agent log
        t_open = time.time()
        # #endregion
        slide = openslide.OpenSlide(wsi_path)
        # #region agent log
        _dlog("A", "app_v21.py:show_wsi:OpenSlide", "OpenSlide open", {"duration_ms": round((time.time() - t_open) * 1000, 2)})
        t_thumb = time.time()
        # #endregion
        # 1. Pick display level (~3000 px width)
        # ------------------------------
        # Load a fast thumbnail instead of large region
        # ------------------------------

        thumb_width = 3000

        w0, h0 = slide.level_dimensions[0]
        thumb_height = int(h0 * (thumb_width / w0))

        base_region = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
        # #region agent log
        _dlog("A", "app_v21.py:show_wsi:get_thumbnail", "get_thumbnail", {"duration_ms": round((time.time() - t_thumb) * 1000, 2)})
        # #endregion

        # compute downsample so existing overlay math keeps working
        downsample = w0 / base_region.size[0]

        level_dims = base_region.size

        # ------------------------------
        # 3. GRID OVERLAY + CLICK HIGHLIGHT
        # ------------------------------
        show_grid = st.checkbox("Show tile grid overlays", value=True, key="grid-toggle")

        if "highlight_mode" not in st.session_state:
            st.session_state.highlight_mode = "HPC clusters"

        st.radio(
            "Highlight mode",
            options=["HPC clusters", "Inflammation", "Necrosis", "Malignant", "Adjacency", "Heatmap"],
            key="highlight_mode",
            horizontal=True
        )



        if st.session_state.highlight_mode == "Heatmap":
            prob_cols = [c for c in df.columns if c.startswith("p_hpc_")]
            if not prob_cols:
                st.warning("No p_hpc_* probability columns found for this slide. Did you merge the wide table?")
            else:
                # Extract HPC ids from column names p_hpc_31 -> 31
                hpc_ids_heat = sorted([int(c.split("_")[-1]) for c in prob_cols])
                st.selectbox(
                    "Heatmap HPC",
                    options=hpc_ids_heat,
                    index=0,
                    key="heat_hpc"
                )
                st.slider(
                    "Heat intensity (visual strength)",
                    0.1, 1.0, 0.6, 0.05,
                    key="heat_alpha"
                )

                st.info("Heatmap auto-scales per HPC using log-normalized probabilities.")

                with st.expander("Heatmap stats for selected HPC", expanded=False):
                    h = int(st.session_state.get("heat_hpc"))
                    col = f"p_hpc_{h}"

                    if col in df.columns:
                        p_series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                        st.write("Min prob:", float(p_series.min()))
                        st.write("Max prob:", float(p_series.max()))
                        st.write("Mean prob:", float(p_series.mean()))
                        st.write("Non-zero tiles:", int((p_series > 0).sum()))
                    else:
                        st.write("No probability column found for this HPC.")


                # Debug: confirm probabilities exist and threshold is not filtering everything
                with st.expander("Heatmap debug", expanded=False):
                    h = int(st.session_state.get("heat_hpc"))
                    col = f"p_hpc_{h}"
                    st.write("Selected heat_hpc:", h)
                    st.write("Column exists:", col in df.columns)

                    if col in df.columns:
                        s = pd.to_numeric(df[col], errors="coerce")
                        st.write("Non null probs:", int(s.notna().sum()), "out of", len(s))
                        st.write("Max prob:", float(s.max()) if s.notna().any() else None)
                        st.write("Min prob:", float(s.min()) if s.notna().any() else None)
                        pmin = float(st.session_state.get("heat_min", 0.0))
                        st.write("Tiles above threshold:", int((s >= pmin).sum()))



        if st.session_state.highlight_mode == "Inflammation":
            st.markdown(
                """
                <div style="display:flex; gap:18px; align-items:center; flex-wrap:wrap; padding:6px 0;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px; height:14px; background:#50C878; border:1px solid #000; display:inline-block;"></span>
                    <span>None-sparse</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px; height:14px; background:#FFC800; border:1px solid #000; display:inline-block;"></span>
                    <span>Mild-moderate</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px; height:14px; background:#FF5050; border:1px solid #000; display:inline-block;"></span>
                    <span>Marked</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px; height:14px; background:#A0A0A0; border:1px solid #000; display:inline-block;"></span>
                    <span>Missing</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
            
        if st.session_state.highlight_mode == "Necrosis":
            st.markdown(
            """
            <div style="display:flex; gap:18px; align-items:center; padding:6px 0;">
                <div>
                    <span style="width:14px;height:14px;background:#3CC878;border:1px solid #000;display:inline-block;"></span>
                    None
                </div>
                <div>
                    <span style="width:14px;height:14px;background:#FFA500;border:1px solid #000;display:inline-block;"></span>
                    Some
                </div>
                <div>
                    <span style="width:14px;height:14px;background:#C82828;border:1px solid #000;display:inline-block;"></span>
                    Universal
                </div>
                <div>
                    <span style="width:14px;height:14px;background:#A0A0A0;border:1px solid #000;display:inline-block;"></span>
                    Missing
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.highlight_mode == "Malignant":
            st.markdown(
            """
            <div style="display:flex; gap:18px; align-items:center; padding:6px 0;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px;height:14px;background:#FF5050;border:1px solid #000;display:inline-block;"></span>
                    Malignant
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px;height:14px;background:#50C878;border:1px solid #000;display:inline-block;"></span>
                    Non-malignant
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px;height:14px;background:#A0A0A0;border:1px solid #000;display:inline-block;"></span>
                    Missing
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.session_state.highlight_mode == "Adjacency":
            a = st.session_state.get("adj_hpc_a", None)
            b = st.session_state.get("adj_hpc_b", None)

            st.markdown(
            f"""
            <div style="display:flex; gap:18px; align-items:center; padding:6px 0;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px;height:14px;background:#50C878;border:1px solid #000;display:inline-block;"></span>
                    HPC {a} touching HPC {b}
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px;height:14px;background:#FFA500;border:1px solid #000;display:inline-block;"></span>
                    HPC {b} touching HPC {a}
                </div>
            </div>
            """,
            unsafe_allow_html=True
            )
        df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")

        highlight_mode = st.session_state.get("highlight_mode", "HPC clusters")
        selected_hpc = st.session_state.get("selected_hpc", None)

        # Only restrict by selected_hpc in HPC clusters mode.
        # Heatmap should draw across all tiles, not just the selected cluster.
        if selected_hpc is not None and highlight_mode == "HPC clusters":
            grid_df = df[df["hpc_id"] == selected_hpc]

        else:
            grid_df = df

        base_rgba = base_region.convert("RGBA")


        # Layer for outlines / non-heatmap drawing
        overlay = base_rgba.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Layer ONLY for heatmap
        heat_layer = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))

        heat_draw = ImageDraw.Draw(heat_layer, "RGBA")


       

# HPC FILTER PANEL (COLORED)
        if "selected_hpc" not in st.session_state:
            st.session_state.selected_hpc = None

        df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")
        hpc_list = sorted(df["hpc_id"].dropna().astype(int).unique().tolist())


        if st.session_state.highlight_mode == "Adjacency":
            
            pair_edge_counts, tile_has_neighbor_pair = compute_adjacency_cached_for_slide(slide_id)
            with st.expander("Adjacency and cooccurrence controls", expanded=False):

                st.subheader("Cooccurrence on this slide")

                top_pairs = sorted(pair_edge_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]

                if top_pairs:
                    options = [f"HPC {a} ↔ HPC {b} ({cnt} edges)" for (a, b), cnt in top_pairs]
                    selected_pair = st.selectbox(
                        "Pick a top cooccurring pair",
                        options,
                        key="adj_top_pair_select"
                    )

                    idx = options.index(selected_pair)
                    (a_sel, b_sel), _ = top_pairs[idx]

                    if st.button("Highlight selected top pair", key="adj_top_pair_btn"):
                        p = (a_sel, b_sel) if a_sel < b_sel else (b_sel, a_sel)
                        st.session_state.adj_hpc_a = a_sel
                        st.session_state.adj_hpc_b = b_sel
                        st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(
                            p, {"a_touch": set(), "b_touch": set()}
                        )
                        st.rerun()
                else:
                    st.info("No adjacent cross HPC pairs found on this slide.")

                # st.divider()
                # st.subheader("Show where HPC A is beside HPC B")

                # if len(hpc_list) < 2:
                #     st.info("Not enough distinct HPCs on this slide to compute adjacency.")
                # else:
                #     col1, col2, col3 = st.columns([1, 1, 1])

                #     with col1:
                #         a_choice = st.selectbox("HPC A", options=hpc_list, index=0, key="adj_a_choice")
                #     with col2:
                #         b_choice = st.selectbox("HPC B", options=hpc_list, index=1, key="adj_b_choice")
                #     with col3:
                #         apply_btn = st.button("Apply adjacency highlight", key="adj_apply_btn")

                #     if apply_btn and int(a_choice) != int(b_choice):
                #         a = int(a_choice)
                #         b = int(b_choice)
                #         p = (a, b) if a < b else (b, a)

                #         st.session_state.adj_hpc_a = a
                #         st.session_state.adj_hpc_b = b
                #         st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(
                #             p, {"a_touch": set(), "b_touch": set()}
                #         )

                #         st.rerun()

        with st.expander(" Highlight tiles by HPC cluster", expanded=False):

            # Option to reset
            if st.button("Show all HPCs"):
                st.session_state.selected_hpc = None
                st.rerun()

            cols = st.columns(4)

            for i, hpc_id in enumerate(hpc_list):
                color = color_for_hpc(hpc_id)
                css_color = rgb_to_css(color)

                with cols[i % 4]:
                    clicked = st.markdown(
                        f"""
                        <div style="
                            display:flex;
                            align-items:center;
                            cursor:pointer;
                            padding:6px;
                            border-radius:6px;
                            background-color:{'#222' if st.session_state.selected_hpc == hpc_id else 'transparent'};
                        ">
                            <div style="
                                width:14px;
                                height:14px;
                                background-color:{css_color};
                                border:1px solid #000;
                                margin-right:6px;
                            "></div>
                            <span>HPC {hpc_id}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Invisible button overlay for click handling
                    if st.button(f"HPC_{hpc_id}", key=f"hpc_btn_{hpc_id}"):
                        st.session_state.selected_hpc = hpc_id
                        st.rerun()

        selected_hpc = st.session_state.get("selected_hpc")
        if selected_hpc is not None:
            with st.expander("🧠 HPC Biological Interpretation", expanded=True):
                render_hpc_annotation(selected_hpc)   

        infl_f = st.session_state.get("query_inflammation", None)
        nec_f = st.session_state.get("query_necrosis", None)

        filtered_df = df.copy()
        if st.session_state.get("selected_hpc") is not None:
            filtered_df = filtered_df[filtered_df["hpc_id"] == st.session_state.selected_hpc]
        if infl_f is not None:
            filtered_df = filtered_df[filtered_df["inflammation"].astype(str).str.lower().str.strip() == infl_f]
        if nec_f is not None:
            filtered_df = filtered_df[filtered_df["necrosis"].astype(str).str.lower().str.strip() == nec_f]

        st.caption(f"Matched tiles: {len(filtered_df)}")
        # --------------------------------------------------
        # HEATMAP (draw regardless of show_grid)
        # --------------------------------------------------
        highlight_mode = st.session_state.get("highlight_mode", "HPC clusters")

        if highlight_mode == "Heatmap":
            h = int(st.session_state.get("heat_hpc"))
            col = f"p_hpc_{h}"

            if col not in df.columns:
                st.warning(f"No probability column {col} found")
            else:
                # Get numeric vector of probabilities
                p_series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

                # Auto-scale range for this HPC
                pmax = float(p_series.max())
                pmin = float(p_series.min())

                # Avoid division by zero
                if pmax == pmin:
                    pmax = pmin + 1e-9

                # Parameters
                alpha_max = float(st.session_state.get("heat_alpha", 0.6))
                alpha_max = max(0.2, min(alpha_max, 0.95))
                alpha_min = 0.10

                # Log scaling strength
                use_log = True
                epsilon = 1e-6

                # Precompute arrays once (faster than iterrows)
                xs = (df["x_native"].astype(float) / float(downsample)).astype(int).to_numpy()
                ys = (df["y_native"].astype(float) / float(downsample)).astype(int).to_numpy()
                ps = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)

                valid = np.isfinite(ps)
                if not valid.any():
                    st.warning("No valid probabilities available for heatmap on this slide.")
                else:
                    p_valid = ps[valid]
                    pmin = float(np.nanmin(p_valid))
                    pmax = float(np.nanmax(p_valid))
                    if pmax <= pmin:
                        pmax = pmin + 1e-9

                    if use_log:
                        denom = (np.log10(pmax + epsilon) - np.log10(pmin + epsilon))
                        if denom == 0:
                            denom = 1e-9
                        t_all = (np.log10(ps + epsilon) - np.log10(pmin + epsilon)) / denom
                    else:
                        t_all = (ps - pmin) / (pmax - pmin)

                    t_all = np.clip(t_all, 0.0, 1.0)

                    # Alpha mapping
                    a_all = alpha_min + (alpha_max - alpha_min) * t_all
                    alpha_all = (255.0 * a_all).astype(np.uint8)

                    ts = int(float(TILE_SIZE_NATIVE) / float(downsample))

                    # Draw only valid tiles
                    for x, y, t, alpha in zip(xs[valid], ys[valid], t_all[valid], alpha_all[valid]):
                        # low: cyan (0,255,255), high: yellow (255,255,0)
                        R = int(255 * float(t))
                        G = 255
                        B = int(255 * (1 - float(t)))
                        heat_draw.rectangle(
                            [int(x), int(y), int(x) + ts, int(y) + ts],
                            fill=(R, G, B, int(alpha)),
                            outline=(255, 255, 255, 40),
                        )

        if show_grid:

            highlight_mode = st.session_state.get("highlight_mode", "HPC clusters")
            selected_hpc = st.session_state.get("selected_hpc", None)

            
            # If an HPC is selected, restrict drawing only for modes where it makes sense
            # Heatmap should always draw across all tiles
            if selected_hpc is not None and highlight_mode not in ["Adjacency", "Heatmap"]:
                grid_df = df[df["hpc_id"] == selected_hpc]
            else:
                grid_df = df
            if highlight_mode != "Heatmap":
                for _, r in grid_df.iterrows():
                    x = int(r["x_native"] / downsample)
                    y = int(r["y_native"] / downsample)
                    ts = int(TILE_SIZE_NATIVE / downsample)

                    # Optional per tile filters
                    if infl_f is not None:
                        val = str(r.get("inflammation", "")).strip().lower()
                        if val != infl_f:
                            continue

                    if nec_f is not None:
                        val = str(r.get("necrosis", "")).strip().lower()
                        if val != nec_f:
                            continue

                # Pick color depending on mode
                    # Pick color depending on mode
                    if highlight_mode == "Inflammation":
                        color = color_for_inflammation(r.get("inflammation", None))

                    # interpret slider as MAX opacity, but cap it to stay transparent
                    elif highlight_mode == "Necrosis":
                        color = color_for_necrosis(r.get("necrosis", None))

                    elif highlight_mode == "Malignant":
                        color = color_for_malignant(r.get("malignant", None))

                    elif highlight_mode == "Adjacency":
                        adj = st.session_state.get("adj_tile_sets", None)
                        if adj is None:
                            continue

                        # IMPORTANT: adjacency sets are built using slide_tile (globally unique)
                        tkey = str(r.get("slide_tile"))
                        if (tkey not in adj.get("a_touch", set())) and (tkey not in adj.get("b_touch", set())):                        
                            continue

                    # normalize probability above threshold
                        if tkey in adj.get("a_touch", set()):
                            color = color_for_adjacency_group("a_touch")
                        else:
                            color = color_for_adjacency_group("b_touch")

                    # alpha varies with probability, fixes binary mask

                    else:
                        # Default mode: HPC clusters
                        color = color_for_hpc(r.get("hpc_id", None))

                    # Draw outline for all non heatmap modes
                    draw.rectangle(
                        [x, y, x + ts, y + ts],
                        outline=color,
                        width=5
                    )


        # Draw highlight on selected tile
        sel = st.session_state.get("selected_tile")
        if sel is not None and ("x_native" in sel) and ("y_native" in sel):
            sx = int(sel["x_native"] / downsample)
            sy = int(sel["y_native"] / downsample)
            ts = int(TILE_SIZE_NATIVE / downsample)

            # Glow + strong green box
            draw.rectangle(
                [sx - 3, sy - 3, sx + ts + 3, sy + ts + 3],
                outline="yellow",
                width=15
            )
            draw.rectangle(
                [sx, sy, sx + ts, sy + ts],
                outline="lime",
                width=15
            )

        # If heatmap mode, blend heatmap layer on top
        if st.session_state.highlight_mode == "Heatmap":
            overlay = Image.alpha_composite(overlay, heat_layer)
        
        # Ensure a stable 8-bit RGB array for Streamlit components
        overlay_np = np.asarray(overlay.convert("RGB"), dtype=np.uint8)
        # ------------------------------
        # 4. VIEWER MODE SELECTION
        # ------------------------------
        st.subheader("Slide Viewer")

        mode = st.radio(
            "Choose viewer mode:",
            ["Zoom", "Click"],
            horizontal=True,
            key="viewer-mode"
        )
        
        # ------------------------------
        # 5. ZOOM MODE
        # ------------------------------
    
        if mode == "Zoom":
            
            # Ensure the component gets a standard numpy array
            overlay_np = np.asarray(overlay_np, dtype=np.uint8)            
            image_zoom(
                overlay_np,
                zoom_factor=2,
                keep_aspect_ratio=True
            )
            return
        # ------------------------------
        # 6. CLICK MODE
        # ------------------------------

        st.write("Click a tile to view it.")

        click = streamlit_image_coordinates(overlay_np, key="wsi-click-coords")

        if not click:
            st.info("Click anywhere on the slide to select a tile.")
            return

        cx, cy = click["x"], click["y"]

        # Convert click → native coordinates
        native_x = cx * downsample
        native_y = cy * downsample

        # ------------------------------
        # 7. Identify clicked tile
        # ------------------------------
        tile_row = None
        tol = TILE_SIZE_NATIVE * 0.1  # 10 percent tolerance

        for _, r in df.iterrows():
            x0, y0 = r["x_native"], r["y_native"]
            x1, y1 = x0 + TILE_SIZE_NATIVE, y0 + TILE_SIZE_NATIVE

            if (x0 - tol) <= native_x <= (x1 + tol) and (y0 - tol) <= native_y <= (y1 + tol):
                tile_row = r
                break

        if tile_row is None:
            st.warning("Clicked area does not match any tile.")
            return

        # ------------------------------
        # 8. Save selected tile + force rerun
        # ------------------------------
        new_selected = {
            "slide_tile": str(tile_row.get("slide_tile", "")),
            "x_native": float(tile_row["x_native"]),
            "y_native": float(tile_row["y_native"]),
        }

        prev = st.session_state.get("selected_tile")

        same_tile = (
            isinstance(prev, dict)
            and prev.get("slide_tile") == new_selected["slide_tile"]
            and float(prev.get("x_native", -1)) == new_selected["x_native"]
            and float(prev.get("y_native", -1)) == new_selected["y_native"]
        )

        if not same_tile:
            st.session_state.selected_tile = new_selected
            st.rerun()

        # ------------------------------
        # 9. Display selected tile image
        # ------------------------------
        h5_idx = tile_row.get("h5_index", None)

        # if duplicate columns made it a Series, take the first non-null value
        if isinstance(h5_idx, pd.Series):
            h5_idx = h5_idx.dropna()
            h5_idx = h5_idx.iloc[0] if len(h5_idx) else None

        if h5_idx is None or pd.isna(h5_idx):
            st.warning("Selected tile has no h5_index mapping. Check your tile_registry merge/rename.")
            return

        idx = int(h5_idx)
        tile_name = tile_row["tiles"]

        st.success(f"Tile selected: {tile_name}")

        tile_img = load_tile_from_h5(idx)
        st.image(tile_img, caption=f"{tile_name} (index {idx})")


        heat_hpc = st.session_state.get("heat_hpc") if st.session_state.highlight_mode == "Heatmap" else None
        render_tile_info(tile_row, heat_hpc=heat_hpc)
        # ------------------------------
        # 10. Show HPC annotation for this tile
        # ------------------------------
        hpc_id = tile_row.get("hpc_id", None)
        if hpc_id is not None and not pd.isna(hpc_id):
            render_hpc_annotation(int(hpc_id))
        
    except Exception as e:
        st.exception(e)



def detect_entity_patterns(query):
    q = query.strip().lower()

    def after(keyword):
        m = re.search(rf"{keyword}\s*[:=]?\s*([^\s,;]+)", q, re.I)
        return m.group(1).strip() if m else None

    tile_match = after("tile")
    slide_match = after("slide")
    sample_match = after("sample")
    hpc_match = after("hpc")

    if slide_match is None:
        m = re.search(r"\b(TCGA-[A-Z0-9\-]+-?DX\d+)\b", query, re.I)
        if m:
            slide_match = m.group(1)

    if not tile_match:
        m = re.search(
            r"\b(tile[_\-]?\d+|[A-Za-z0-9_\-]+\.jpe?g|[A-Za-z0-9_\-]+\.png|[A-Za-z0-9_\-]+\.tif)\b",
            q
        )
        tile_match = m.group(0) if m else None

    if not hpc_match:
        m = re.search(r"\bhpc[-_\s]*([0-9]+)\b", q, re.I)
        hpc_match = m.group(1) if m else None

    return {
        "tile": tile_match,
        "slide": slide_match,
        "sample": sample_match,
        "hpc": [hpc_match] if hpc_match else None
    }

def fetch_answer_from_db(query: str):
    q = (query or "").lower()

    polarity = classify_malignancy_polarity(q)
    is_malignant_question = polarity in ("malignant", "non")
    is_negative = polarity == "non"

    with engine.connect() as conn:
        if any(kw in q for kw in ["how many", "count", "total hpcs", "number of hpcs"]):
            return handle_analytics(conn, q, "count_hpcs")

        if any(kw in q for kw in ["portion", "coverage", "covered by malignant", "percent malignant"]):
            return handle_analytics(conn, q, "slide_malignant_coverage")

        if any(kw in q for kw in ["survival", "cox", "hazard ratio", "p-value", "significant", "regression"]):
            return handle_survival(conn, q)

        detected = detect_entity_patterns(query)

        intents = {
            "malignant": is_malignant_question,
            "negative": is_negative,
            "tile": detected.get("tile"),
            "slide": detected.get("slide"),
            "sample": detected.get("sample"),
            "hpc": detected.get("hpc"),
        }

        handlers = {
            "tile": handle_tile,
            "slide": handle_slide,
            "sample": handle_sample,
            "hpc": handle_hpc,
        }

        output_parts = []
        for entity, func in handlers.items():
            match = intents.get(entity)
            if match:
                part = func(conn, match, intents)
                if part:
                    output_parts.append(part)

        if not output_parts:
            return "Please specify a valid tile, slide, sample, or HPC ID."

        return "\n\n---\n\n".join(output_parts)


def classify_malignancy_polarity(q: str):
    """
    Return: 'non' | 'malignant' | None
    Detects negated malignant epithelium first to avoid substring traps.
    """
    q = q.lower()

    # normalize hyphen/underscore variants to spaces
    q_norm = re.sub(r'[_\-]+', ' ', q)

    # --- check NEGATIVE FIRST ---
    neg_patterns = [
        r'\bnon\s*malignant\b',
        r'\bnot\s+malignant\b',
        r"\bdoes\s+not\s+(have|contain|show)\s+malignant\b",
        r"\bwithout\s+malignant\b",
        r"\black\s+of\s+malignant\b",         # if you’ve seen “lack of”
        r"\bno\s+malignant\b",
        r"\bnon\s*malignant\s*epithelium\b",
        r"\bnot\s+malignant\s*epithelium\b",
        r"\bdoes\s+not\s+have\s+malignant\s*epithelium\b",
        r"\bwithout\s+malignant\s*epithelium\b",
        r"\bno\s+malignant\s*epithelium\b",
    ]
    if any(re.search(p, q_norm) for p in neg_patterns):
        return 'non'

    # --- then POSITIVE ---
    pos_patterns = [
        r'\bmalignant\b',
        r'\bmalignant\s*epithelium\b',
    ]
    if any(re.search(p, q_norm) for p in pos_patterns):
        return 'malignant'

    return None


def handle_tile(conn, match, intents):
    """Resolve a tile by name/path, optionally attach malignant vs non-malignant KB details,
    and preview the corresponding H5 image.

    Notes:
    - Prefer exact match first, then fall back to ILIKE.
    - Use slide_tile if the user provided it (globally unique) otherwise tiles.
    - Treat hpc_id as int for all HPC table lookups.
    """

    tile_name = match if isinstance(match, str) else match.group(1)
    tile_name = (tile_name or "").strip()

    if not tile_name:
        return "Please specify a tile name."

    # If the user passed a full slide_tile like TCGA-..._tile_123, prefer that
    provided_slide_tile = None
    if "_" in tile_name and tile_name.upper().startswith("TCGA-"):
        provided_slide_tile = tile_name.strip().upper()

    # 1) Resolve the row from tile_registry
    if provided_slide_tile:
        row = conn.execute(
            text(
                """
                SELECT *
                FROM tile_registry
                WHERE slide_tile = :st
                LIMIT 1
                """
            ),
            {"st": provided_slide_tile},
        ).fetchone()
    else:
        # Try exact match first (fast + unambiguous)
        row = conn.execute(
            text(
                """
                SELECT *
                FROM tile_registry
                WHERE tiles = :t
                   OR h5_source_path = :t
                LIMIT 1
                """
            ),
            {"t": tile_name},
        ).fetchone()

        # Fallback to partial match
        if not row:
            row = conn.execute(
                text(
                    """
                    SELECT *
                    FROM tile_registry
                    WHERE tiles ILIKE :t
                       OR h5_source_path ILIKE :t
                    ORDER BY image_index ASC
                    LIMIT 1
                    """
                ),
                {"t": f"%{tile_name}%"},
            ).fetchone()

    if not row:
        return f"No tile found matching '{tile_name}'."
    info = dict(row._mapping)
    out = [f"### Tile `{tile_name}` Summary"]
    out += [f"- **{k}**: {v}" for k, v in info.items()]

    # robust index conversion
    idx = info.get("image_index")
    try:
        idx = int(idx) if idx is not None else None
    except (TypeError, ValueError):
        idx = None

    # robust hpc_id conversion    
    hpc_id = info.get("hpc_id")
    try:
        hpc_id_int = int(hpc_id) if hpc_id is not None else None
    except (TypeError, ValueError):
        hpc_id_int = None

    # 2) Optional: attach malignant vs non-malignant details
    # intents["malignant"] means user explicitly asked about malignant vs non-malignant
    if intents.get("malignant") and hpc_id_int is not None:
        if intents.get("negative"):

            # User explicitly asked for NON-malignant
            kb_row = conn.execute(
                text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                {"hpc_id": hpc_id_int},
            ).fetchone()
            out.append(f"Tile belongs to **non-malignant epithelium** (HPC {hpc_id_int})")
        else:
            
            # Determine malignant flag, then pick details table
            malignant_flag = conn.execute(
                text("SELECT malignant FROM hpc_dictionary WHERE hpc_id = :hpc_id LIMIT 1"),
                {"hpc_id": hpc_id_int},
            ).scalar()

            # Normalize malignant_flag to bool (covers numpy/int/string)
            if isinstance(malignant_flag, (int, np.integer)):
                malignant_flag = bool(malignant_flag)
            elif isinstance(malignant_flag, str):
                s = malignant_flag.strip().lower()
                if s in ["true", "t", "1", "yes", "y"]:
                    malignant_flag = True
                elif s in ["false", "f", "0", "no", "n"]:
                    malignant_flag = False
                else:
                    malignant_flag = None

            if malignant_flag is True:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                    {"hpc_id": hpc_id_int},
                ).fetchone()
                out.append(f"Tile belongs to **malignant epithelium** (HPC {hpc_id_int})")
            else:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                    {"hpc_id": hpc_id_int},
                ).fetchone()
                out.append(f"Tile belongs to **non-malignant epithelium** (HPC {hpc_id_int})")


        if kb_row:
            out.append("**Epithelium details from KB:**")
            for k, v in dict(kb_row._mapping).items():
                out.append(f"- **{k}**: {v}")
            out.append("")

    # 3) Display image if index available
    try:
        def _to_uint8(tile_arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(tile_arr)
            arr = np.squeeze(arr)

            if arr.dtype == np.uint8:
                return arr

            x = arr.astype(np.float32)
            mn = float(np.nanmin(x)) if x.size else 0.0
            mx = float(np.nanmax(x)) if x.size else 0.0

            # If already 0..1, scale directly
            if 0.0 <= mn and mx <= 1.0:
                x = x * 255.0
                return np.clip(x, 0, 255).astype(np.uint8)

            # Percentile scaling for stability (avoids washed-out tiles)
            lo = float(np.nanpercentile(x, 1))
            hi = float(np.nanpercentile(x, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return np.clip(x, 0, 255).astype(np.uint8)

            x = (x - lo) / (hi - lo) * 255.0
            return np.clip(x, 0, 255).astype(np.uint8)

        with h5py.File(h5_path, "r") as f:
            dataset = f["train_img"]
            total = int(dataset.shape[0])

            if idx is None:
                st.warning("No valid image index found for this tile.")
            elif idx < 0 or idx >= total:
                st.warning(f"Invalid index {idx}, skipping image.")
            else:
                tile = dataset[idx]
                tile_u8 = _to_uint8(tile)

                if tile_u8.ndim == 2:
                    img = Image.fromarray(tile_u8)
                elif tile_u8.ndim == 3 and tile_u8.shape[-1] in (1, 3, 4):
                    if tile_u8.shape[-1] == 1:
                        img = Image.fromarray(tile_u8[:, :, 0])
                    else:
                        img = Image.fromarray(tile_u8)
                else:
                    st.warning(f"Unexpected tile shape {tile_u8.shape}")
                    return "\n".join(out)

                st.image(img, caption=f"{tile_name} (index {idx})", use_container_width=True)

    except Exception as e:
        st.error(f"Could not load image from H5 file: {e}")

    return "\n".join(out)
  

def handle_slide(conn, match, intents):
    # Normalize slide ID
    slide_id = match if isinstance(match, str) else match.group(0)
    slide_id = re.sub(r"^slide\s+", "", str(slide_id), flags=re.I).strip().upper()
    out = [f"### 🧫 Slide `{slide_id}` Summary\n"]

    # -------- 1) Malignancy question (malignant vs non-malignant) --------
    if intents.get("malignant"):
        want_non = bool(intents.get("negative"))

        # If user asked for "non-malignant", return non-malignant HPCs; else malignant HPCs.
        malignant_clause = "hd.malignant = FALSE" if want_non else "hd.malignant = TRUE"

        hpc_ids = conn.execute(
            text(
                f"""
                SELECT DISTINCT hp.hpc_id
                FROM hpl_profile_proportion hp
                JOIN hpc_dictionary hd ON hp.hpc_id = hd.hpc_id
                WHERE UPPER(hp.slides) = :slide_id
                  AND {malignant_clause}
                ORDER BY hp.hpc_id
                """
            ),
            {"slide_id": slide_id},
        ).scalars().all()

        if hpc_ids:
            label = "Non-malignant" if want_non else "Malignant"
            out.append(f"⚠️ {label} HPCs: {', '.join(map(str, hpc_ids))}")
        else:
            out.append("No matching HPCs detected for this slide.")

        st.session_state.active_slide = slide_id
        return "\n".join(out) + f"\n\nThe slide viewer is now updated for **{slide_id}**."

    # -------- 2) Slide summary --------
    # Use exact slide id to avoid accidental partial matches.
    summary = conn.execute(
        text(
            """
            SELECT *
            FROM hpl_profile_summary
            WHERE UPPER(slides) = :slide_id
            LIMIT 1
            """
        ),
        {"slide_id": slide_id},
    ).fetchone()

    if summary:
        out.append("**Slide Summary:**")
        for k, v in dict(summary._mapping).items():
            out.append(f"- **{k}**: {v}")
        out.append("")

    else:
        out.append("No slide summary found.")
        out.append("")
    # -------- 3) Top HPCs --------
    proportions = conn.execute(
        text(
            """
            SELECT hpc_id, proportion, samples
            FROM hpl_profile_proportion
            WHERE UPPER(slides) = :slide_id
            ORDER BY proportion DESC
            LIMIT 5
            """
        ),
        {"slide_id": slide_id},
    ).fetchall()

    if proportions:
        out.append("**Top HPCs:**")
        for row in proportions:
            d = dict(row._mapping)
            # proportion might be None or non-float; guard formatting
            prop = d.get("proportion", None)
            try:
                prop_txt = f"{float(prop):.5f}" if prop is not None else "NA"
            except (TypeError, ValueError):
                prop_txt = "NA"
            out.append(
            f"- **HPC {d.get('hpc_id')}** → proportion: {prop_txt}, sample: {d.get('samples')}"
            )
        out.append("")

    

    # -------- 4) Update Slide Viewer --------
    st.session_state.active_slide = slide_id
    return "\n".join(out) + f"\n\nThe slide viewer is now updated for **{slide_id}**."

def handle_sample(conn, match, intents):
    sample_id = match if isinstance(match, str) else match.group(0)
    sample_id = (sample_id or "").strip().upper()

    out = [f"### 🧬 Sample `{sample_id}` Summary"]

    if not sample_id:
        return "Please specify a valid sample ID."

    # 1) Exact match first (recommended)
    summary = conn.execute(
        text("""
            SELECT *
            FROM hpl_profile_summary
            WHERE UPPER(samples) = :s
            LIMIT 1
        """),
        {"s": sample_id}
    ).fetchone()

    # 2) Fallback to partial match only if exact not found
    if not summary:
        summary = conn.execute(
            text("""
                SELECT *
                FROM hpl_profile_summary
                WHERE samples ILIKE :s
                LIMIT 1
            """),
            {"s": f"%{sample_id}%"}
        ).fetchone()

    if summary:
        out += [f"- **{k}**: {v}" for k, v in dict(summary._mapping).items()]
    else:
        out.append("No sample summary found.")

    return "\n".join(out)


def handle_hpc(conn, ids, intents):
    """
    Handle HPC queries.

    Key behaviors:
    - Treat hpc_id as int for all HPC-table queries.
    - If the user asked malignant vs non-malignant, respect intents['negative'].
    - Avoid ILIKE on numeric IDs (can cause partial matches and type issues).
    - Keep the generic "scan other tables" logic for non-malignancy lookups.
    - Preview a few example tiles with robust uint8 conversion (avoid washed-out tiles).
    """

    insp = inspect(engine)
    out = []

    # Helper: robust display normalization (avoid washed-out tiles)
    def _to_uint8(tile_arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(tile_arr)
        arr = np.squeeze(arr)

        if arr.dtype == np.uint8:
            return arr

        x = arr.astype(np.float32)

        if x.size:
            mn = float(np.nanmin(x))
            mx = float(np.nanmax(x))
        else:
            mn, mx = 0.0, 0.0

        if 0.0 <= mn and mx <= 1.0:
            x = x * 255.0
            return np.clip(x, 0, 255).astype(np.uint8)

        lo = float(np.nanpercentile(x, 1)) if x.size else 0.0
        hi = float(np.nanpercentile(x, 99)) if x.size else 255.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.clip(x, 0, 255).astype(np.uint8)

        x = (x - lo) / (hi - lo) * 255.0
        return np.clip(x, 0, 255).astype(np.uint8)

    for hpc_id in ids:
        # Normalize HPC id to int early
        try:
            hpc_id_int = int(hpc_id)
        except (TypeError, ValueError):
            out.append(f"## HPC {hpc_id} Summary\n\n⚠️ Invalid HPC ID: {hpc_id}")
            continue

        block = [f"## HPC {hpc_id_int} Summary\n"]

        # 1) Dictionary Info
        base = conn.execute(
            text("SELECT * FROM hpc_dictionary WHERE hpc_id = :hpc_id LIMIT 1"),
            {"hpc_id": hpc_id_int},
        ).fetchone()

        if not base:
            block.append(f"No entry found for HPC {hpc_id_int}.\n")
            out.append("\n".join(block))
            continue

        d = dict(base._mapping)

        # 2) If user asked about malignancy only
        if intents.get("malignant"):
            want_non = bool(intents.get("negative"))

            malignant_flag = d.get("malignant")
            if isinstance(malignant_flag, (int, np.integer)):
                malignant_flag = bool(malignant_flag)
            elif isinstance(malignant_flag, str):
                s = malignant_flag.strip().lower()
                if s in ["true", "t", "1", "yes", "y"]:
                    malignant_flag = True
                elif s in ["false", "f", "0", "no", "n"]:
                    malignant_flag = False
                else:
                    malignant_flag = None

            if want_non:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                    {"hpc_id": hpc_id_int},
                ).fetchone()
                block.append("**Non-malignant epithelium details:**")
            else:
                if malignant_flag is True:
                    kb_row = conn.execute(
                        text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                        {"hpc_id": hpc_id_int},
                    ).fetchone()
                    block.append("**Malignant epithelium details:**")
                else:
                    kb_row = conn.execute(
                        text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id LIMIT 1"),
                        {"hpc_id": hpc_id_int},
                    ).fetchone()
                    block.append("**Non-malignant epithelium details:**")

            if kb_row:
                for k, v in dict(kb_row._mapping).items():
                    block.append(f"- **{k}**: {v}")
                block.append("")
            else:
                block.append("No epithelial phenotype details found for this HPC.")

            out.append("\n".join(block))
            continue

        # 3) Normal query (not malignancy question)
        block.append("**Dictionary Details:**")
        for k, v in d.items():
            block.append(f"- **{k}**: {v}")
        block.append("")

        # Scan other tables
        tables = insp.get_table_names()
        for table in tables:
            if table in ["hpc_dictionary", "h_latent_vectors"]:
                continue

            cols = [c["name"] for c in insp.get_columns(table)]
            id_col = "hpc_id" if "hpc_id" in cols else ("dominant_hpc" if "dominant_hpc" in cols else None)
            if not id_col:
                continue

            if id_col in ("hpc_id", "dominant_hpc"):
                where_sql = f"{id_col} = :hpc_id"
                params = {"hpc_id": hpc_id_int}
            else:
                where_sql = f"TRIM({id_col}::text) ILIKE TRIM(:h)"
                params = {"h": f"%{str(hpc_id_int)}%"}

            rows = conn.execute(
                text(f"SELECT * FROM {table} WHERE {where_sql} LIMIT 5"),
                params,
            ).fetchall()

            if not rows:
                continue

            block.append(f"**{table.replace('_', ' ').title()}:**")
            for r in rows:
                for k, v in dict(r._mapping).items():
                    block.append(f"- **{k}**: {v}")
                block.append("")

        # 4) Display linked tile images (this must be OUTSIDE the table loop)
        tile_rows = conn.execute(
            text(
                """
                SELECT tiles, image_index
                FROM tile_registry
                WHERE hpc_id = :hpc_id
                ORDER BY image_index ASC
                LIMIT 6
                """
            ),
            {"hpc_id": hpc_id_int},
        ).fetchall()

        if tile_rows:
            block.append("")
            st.subheader(f"Example Tiles for HPC {hpc_id_int}")

            try:
                
                with h5py.File(h5_path, "r") as f:
                    dataset = f["train_img"]
                    total = int(dataset.shape[0])

                    cols_ui = st.columns(3)
                    for i, row in enumerate(tile_rows):
                        tile_name = row.tiles

                        try:
                            idx = int(row.image_index) if row.image_index is not None else None
                        except (TypeError, ValueError):
                            idx = None
                        if idx is None or idx < 0 or idx >= total:
                            continue

                        tile = dataset[idx]
                        tile_u8 = _to_uint8(tile)

                        if tile_u8.ndim == 2:
                            img = Image.fromarray(tile_u8)
                        elif tile_u8.ndim == 3 and tile_u8.shape[-1] in (1, 3, 4):
                            img = Image.fromarray(tile_u8[:, :, 0]) if tile_u8.shape[-1] == 1 else Image.fromarray(tile_u8)
                        else:
                            continue

                        cols_ui[i % 3].image(img, caption=f"{tile_name} (index {idx})", use_container_width=True)

            except Exception as e:
                st.error(f"Error loading tiles: {e}")

        out.append("\n".join(block))

    return "\n\n---\n\n".join(out)
def handle_analytics(conn, query, intent_type):
    
    """Analytics handlers.
        intent_type:
      - "count_hpcs": count malignant/non-malignant/unknown HPCs (unknown = NULL)
      - "slide_malignant_coverage": coverage on a given slide based on proportions

    Notes:
      - Keep NULL malignant separate as "unknown".
      - Coverage percentages are based on SUM(hp.proportion) for the slide.
    """

    q = (query or "")
    polarity = classify_malignancy_polarity(q)

    # ------------------------------------------------------------
    # Count HPCs across dictionary
    # ------------------------------------------------------------

    if intent_type == "count_hpcs":
    # Treat NULL malignant as unknown, do not count it in malignant or non-malignant
        result = conn.execute(
            text(
                """
                SELECT
                    SUM(CASE WHEN malignant IS TRUE  THEN 1 ELSE 0 END) AS malignant_count,
                    SUM(CASE WHEN malignant IS FALSE THEN 1 ELSE 0 END) AS non_malignant_count,
                    SUM(CASE WHEN malignant IS NULL  THEN 1 ELSE 0 END) AS unknown_count,
                    COUNT(*) AS total_hpcs
                FROM hpc_dictionary;
                """
            )
        ).fetchone()

        total = int(result.total_hpcs or 0)
        malignant = int(result.malignant_count or 0)
        non_malignant = int(result.non_malignant_count or 0)
        unknown = int(getattr(result, "unknown_count", 0) or 0)
       
        malignant_percent = round(100 * malignant / total, 2) if total else 0
        non_malignant_percent = round(100 * non_malignant / total, 2) if total else 0
        unknown_percent = round(100 * unknown / total, 2) if total else 0

   
        if polarity == "malignant":
            return f"There are **{malignant} malignant HPCs** out of {total} total ({malignant_percent}%)."
        if polarity == "non":
            return f"There are **{non_malignant} non-malignant HPCs** out of {total} total ({non_malignant_percent}%)."

        return (
            f"Total HPCs: **{total}**\n"
            f"- Malignant: {malignant} ({malignant_percent}%)\n"
            f"- Non-malignant: {non_malignant} ({non_malignant_percent}%)\n"
            f"- Unknown: {unknown} ({unknown_percent}%)"
        )
    


    # ------------------------------------------------------------
    # Slide coverage
    # ------------------------------------------------------------
    if intent_type == "slide_malignant_coverage":
        # Accept TCGA-55-7574-01Z-00-DX1 and similar IDs
        # - allows optional dash before DX
        m = re.search(r"\b(TCGA-[A-Z0-9\-]+-?DX\d+)\b", q, re.I)
        if not m:
            return "Please specify a valid slide ID."


        slide_id = m.group(1).strip().upper()

        # Compute malignant and non-malignant coverage separately
        result = conn.execute(
            text(
                """
                SELECT
                    ROUND(
                        CAST(
                            100 * SUM(CASE WHEN hd.malignant IS TRUE  THEN hp.proportion ELSE 0 END)
                            / NULLIF(SUM(hp.proportion), 0)
                            AS numeric
                        ),
                        2
                    ) AS malignant_percent,
                    ROUND(
                        CAST(
                            100 * SUM(CASE WHEN hd.malignant IS FALSE THEN hp.proportion ELSE 0 END)
                            / NULLIF(SUM(hp.proportion), 0)
                            AS numeric
                        ),
                        2
                    ) AS non_malignant_percent
                FROM hpl_profile_proportion hp
                JOIN hpc_dictionary hd ON hp.hpc_id = hd.hpc_id
                WHERE UPPER(hp.slides) = :slide_id;
                """
            ),
            {"slide_id": slide_id},
        ).fetchone()

        # If the slide exists but sums to 0, both percents may be NULL

        if not result:
            return f"No data found for slide `{slide_id}`."

    # -------------------------------------------------------------------
        mal = result.malignant_percent
        non = result.non_malignant_percent

        # If user asked malignant or did not specify polarity, default to malignant
        if polarity in ("malignant", None):
            if mal is None:
                return f"No malignant coverage data found for slide `{slide_id}`."
            return f"Malignant epithelium covers **{mal}%** of slide `{slide_id}`."

        # User asked non-malignant
        if non is None:
            return f"No non-malignant coverage data found for slide `{slide_id}`."
        return f"Non-malignant covers **{non}%** of slide `{slide_id}`."
    
    return "Sorry, I couldn’t identify what kind of analytics you want."



def handle_survival(conn, query: str):
    """Return survival analysis for an HPC.

    Supported queries:
    - "survival hpc 12"
    - "survival tile tile_123.jpeg"
    - If no HPC/tile specified, returns top 10 most significant HPCs.

    Notes:
    - Prefer exact numeric match on hpc_id.
    - If tile is provided, map tile -> hpc_id using slide_tile when available.
    - Robust formatting when columns are NULL or p==0.
    """

    q = (query or "")

    # Detect if user mentioned an HPC
    hpc_match = re.search(r"\bhpc[-_\s]*([0-9]+)\b", q, re.I)

    # Detect tile name if no HPC mentioned (supports .jpeg/.jpg/.png/.tif)
    tile_match = re.search(
        r"(?:\btile[_\-\s]*)?([A-Za-z0-9_\-]+\.(?:jpe?g|png|tif))\b",
        q,
        re.I,
    )

    # Optional: detect slide id so we can build slide_tile if user provided both
    slide_match = re.search(r"\b(TCGA-[A-Z0-9\-]+-?DX\d+)\b", q, re.I)
    slide_id = slide_match.group(1).strip().upper() if slide_match else None

    # ------------------------------
    # Case 1: Tile-based query (map tile -> hpc_id)
    # ------------------------------

    if tile_match and not hpc_match:
        tile_name = tile_match.group(1).strip()

        # If we have a slide id, try slide_tile exact match first (globally unique)
        hpc_id_val = None
        if slide_id:
            slide_tile = f"{slide_id}_{tile_name}"
            hpc_id_val = conn.execute(
                text(
                    """
                    SELECT hpc_id
                    FROM tile_registry
                    WHERE UPPER(slide_tile) = :st
                    LIMIT 1
                    """
                ),
                {"st": slide_tile},
            ).scalar()

        # Fallback: tiles ILIKE match
        if hpc_id_val is None:
            hpc_id_val = conn.execute(
                text(
                    """
                    SELECT hpc_id
                    FROM tile_registry
                    WHERE tiles ILIKE :tile
                    LIMIT 1
                    """
                ),
                {"tile": f"%{tile_name}%"},
            ).scalar()

        if hpc_id_val is None:
            return f"No HPC found for tile `{tile_name}`."
        
        try:
            hpc_id_int = int(hpc_id_val)
        except (TypeError, ValueError):
            return f"Tile `{tile_name}` has an invalid HPC ID mapping: {hpc_id_val}."

        # Treat as HPC query
        hpc_match = True
    else:
        hpc_id_int = None

    # ------------------------------
    # Case 2: Direct HPC query
    # ------------------------------

    if hpc_match:
        if hpc_id_int is None:
            try:
                hpc_id_int = int(hpc_match.group(1))
            except (TypeError, ValueError):
                return "Please specify a valid HPC ID."
        result = conn.execute(
            text(
                """
                SELECT *
                FROM hpc_survival_analysis
                WHERE hpc_id = :hpc_id
                LIMIT 1
                """
            ),
            {"hpc_id": hpc_id_int},
        ).fetchone()

        if not result:
            return f"No survival data found for HPC {hpc_id_int}."

        d = dict(result._mapping)

        def _fnum(v, nd=3, default="NA"):
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return default
                return f"{float(v):.{nd}f}"
            except Exception:
                return default

        p = d.get("p", None)
        try:
            p_float = float(p) if p is not None else None
        except Exception:
            p_float = None

        # Prefer precomputed log2_p column if present, else compute safely.
        log2_p_val = d.get("log2_p", None)
        if log2_p_val is None:
            if p_float is None:
                log2_p_txt = "NA"
            else:
                # avoid -inf when p==0
                eps = 1e-300
                log2_p_txt = _fnum(-np.log2(max(p_float, eps)), nd=3)
        else:
            log2_p_txt = _fnum(log2_p_val, nd=3)

        sig_txt = "Not statistically significant."
        if p_float is not None and p_float < 0.05:
            sig_txt = "Significant association with survival (p < 0.05)."

        return (
            f"### Survival analysis for HPC {hpc_id_int}\n"
            f"- **Hazard Ratio (HR):** {_fnum(d.get('expcoef', None), 3)}\n"
            f"- **Coefficient (β):** {_fnum(d.get('coef', None), 3)}\n"
            f"- **Standard Error (SE):** {_fnum(d.get('se', None), 3)}\n"
            f"- **95% CI (HR):** ({_fnum(d.get('expcoef_lower_95', None), 3)}, {_fnum(d.get('expcoef_upper_95', None), 3)})\n"
            f"- **Z-score:** {_fnum(d.get('z', None), 3)}\n"
            f"- **p-value:** {_fnum(p_float, 4)}\n"
            f"- **-log₂(p):** {log2_p_txt}\n\n"
            f"{sig_txt}"
        )

    # ------------------------------
    # Case 3: No specific tile/HPC — show top significant ones
    # ------------------------------
    top = conn.execute(
        text(
            """
            SELECT hpc_id, expcoef, p
            FROM hpc_survival_analysis
            WHERE p IS NOT NULL
            ORDER BY p ASC
            LIMIT 10
        """
    )
    ).fetchall()

    if top:
        lines = ["### 🧬 Top 10 HPCs associated with survival"]
        for row in top:
            try:
                hr_txt = f"{float(row.expcoef):.3f}" if row.expcoef is not None else "NA"
            except Exception:
                hr_txt = "NA"
            try:
                p_txt = f"{float(row.p):.4f}" if row.p is not None else "NA"
            except Exception:
                p_txt = "NA"
            lines.append(f"- **HPC {row.hpc_id}** → HR={hr_txt}, p={p_txt}")
            return "\n".join(lines)

    return "No survival analysis data available."

# # --- Path to your .h5 file ---


def canon_slide_id(x: str) -> str | None:
    if not x:
        return None
    return str(x).strip().upper()

def get_slide_context_from_query(prompt: str):
    slide_id = canon_slide_id(st.session_state.get("active_slide"))
    if slide_id:
        st.session_state.active_slide = slide_id
        return slide_id

    detected = detect_entity_patterns(prompt)
    slide_id = canon_slide_id(detected.get("slide") if detected else None)
    if slide_id:
        st.session_state.active_slide = slide_id
        return slide_id

    return None




def apply_adjacency_from_query(a: int, b: int, slide_id: str):
    slide_id = str(slide_id).strip().upper()

    st.session_state.active_slide = slide_id
    st.session_state.viewer_open = True
    st.session_state.highlight_mode = "Adjacency"

    st.session_state.selected_hpc = None
    st.session_state.selected_tile = None

    st.session_state.adj_hpc_a = int(a)
    st.session_state.adj_hpc_b = int(b)

    df_slide = tile_coords[tile_coords["slides"] == slide_id]

    needed = ["slide_tile", "x_native", "y_native", "hpc_id"]
    missing = [c for c in needed if c not in df_slide.columns]
    if missing:
        st.warning(f"Cannot compute adjacency, missing columns: {missing}")
        st.session_state.adj_tile_sets = {"a_touch": set(), "b_touch": set()}
        return

    

    pair_edge_counts, tile_has_neighbor_pair = compute_adjacency_cached_for_slide(slide_id)

    p = (int(a), int(b))
    p = (p[0], p[1]) if p[0] < p[1] else (p[1], p[0])

    st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(
        p, {"a_touch": set(), "b_touch": set()}
    )

st.subheader(" View Tiles for our dataset")

try:
    with h5py.File(h5_path, "r") as f:
        dataset_name = "train_img"  # adjust if needed
        dataset = f[dataset_name]
        total_tiles = dataset.shape[0]

        # Add a slider to pick which tile to show
        tile_index = st.slider("Select tile index", 0, total_tiles - 1, 0)

        # Extract the selected tile
        tile = dataset[tile_index]

        # Normalize the tile to 0–255
        tile_min, tile_max = np.min(tile), np.max(tile)
        if tile_max > tile_min:
            tile = (tile - tile_min) / (tile_max - tile_min) * 255
        tile = tile.astype(np.uint8)

        # Convert to PIL image
        if tile.ndim == 3 and tile.shape[-1] in [3, 4]:
            img = Image.fromarray(tile)
        else:
            img = Image.fromarray(tile).convert("RGB")

        # Display the tile
        st.image(img, caption=f"Tile {tile_index}", use_container_width=True)

        # Optionally allow saving
        if st.button(" Save this tile as JPEG"):
            save_path = f"tile_{tile_index}.jpeg"
            img.save(save_path, "JPEG")
            st.success(f"Saved: {save_path}")

except Exception as e:
    st.error(f" Error reading H5 file: {e}")





# --- Image uploader ---
uploaded_file = st.file_uploader("Upload an H&E tile or slide image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! 👋 Do you have questions regarding your H and E cancer patient slides? I’ve got you covered."}
    ]

if "active_slide" not in st.session_state:
    st.session_state.active_slide = None

if "viewer_open" not in st.session_state:
    st.session_state.viewer_open = False

# --------------------------------------------------
# Helpers
# --------------------------------------------------
    # 2) ensure slide context exists
def _ensure_active_slide(prompt_text: str | None) -> str:
    """Pick slide from prompt if present; otherwise keep existing; otherwise default."""
    det = detect_entity_patterns(prompt_text or "")
    slide_from_prompt = det.get("slide") if det else None

    if slide_from_prompt:
        st.session_state.active_slide = str(slide_from_prompt).strip().upper()

    if not st.session_state.active_slide:
        # Safe default: first slide in the map
        st.session_state.active_slide = list(WSI_MAP.keys())[0]

    return st.session_state.active_slide

def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


# --------------------------------------------------
# Display chat history
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something about HPCs...", key="main_chat_input")

# --------------------------------------------------
# Handle new prompt
# --------------------------------------------------
rerun_needed = False

if prompt:
    st.chat_message("user").markdown(prompt)
    _append_message("user", prompt)

    slide_id = _ensure_active_slide(prompt)

    # Update viewer state based on query (mode, selected HPC, adjacency etc.)
    apply_chat_query_to_wsi_state(prompt, slide_id)
    st.session_state.viewer_open = True


    # 4) normal response pipeline
    plan = build_query_plan(prompt)

    path = save_query_plan_to_file(
        plan,
        slide_id=plan["entities"]["slide"][0] if plan["entities"]["slide"] else None
    )

    with st.expander("🧩 Query Plan (debug)", expanded=False):
        st.json(plan)
        st.caption(f"Saved query plan → {path}")

    if plan["intent"] in ["general_query", "greeting", "help"]:
        structured_answer = None
    else:
        structured_answer = fetch_answer_from_db(prompt)

    try:
        final_answer = explain(plan, structured_answer)
    except Exception:
        final_answer = structured_answer or "Hi! How can I help you?"

    with st.chat_message("assistant"):
        st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # Optional: only rerun once at the very end if you want immediate viewer redraw
    st.rerun()

 

# --------------------------------------------------
# Persistent Slide Context UI
# --------------------------------------------------

# Load available slides from DB (cached in load_wsi_registry)
WSI_MAP = load_wsi_registry()
slide_options = sorted(WSI_MAP.keys())

if not slide_options:
    st.warning("No slides found in wsi_registry.")
    slide_id = None
else:
    # Default slide: keep existing active_slide if valid, else first slide
    current = (st.session_state.get("active_slide") or "").strip().upper()
    default_idx = slide_options.index(current) if current in slide_options else 0

    slide_id = st.selectbox(
        "Choose a slide",
        slide_options,
        index=default_idx,
        key="slide_selectbox",
    )

    # Keep the global slide context in sync
    st.session_state.active_slide = slide_id

    render_tile_preview(slide_id, limit=30)


# --------------------------------------------------
# Slide Viewer Toggle 
# --------------------------------------------------

if st.session_state.viewer_open:
    if st.button("Close Slide Viewer"):
        st.session_state.viewer_open = False
        rerun_needed = True
else:
    if st.button(f"Open Slide Viewer for {slide_id}"):
        st.session_state.viewer_open = True
        rerun_needed = True



# Only show viewer if viewer_open and slide_id is not None
if st.session_state.viewer_open and slide_id:
    show_wsi(slide_id)

# One rerun at the very end (prevents multiple reruns per user action)
if rerun_needed:
    st.rerun()