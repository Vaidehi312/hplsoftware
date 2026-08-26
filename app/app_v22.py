"""
app_v22.py — HPC Chatbot with Tile-Server Architecture

Key changes from v21:
  - WSI images are fetched from a FastAPI tile server (runs on HPCC) via HTTP.
  - No more openslide or sshfs mount on the laptop.
  - Tile metadata comes from the server as JSON → DataFrame.
  - Local disk cache prevents repeat network fetches.
  - DB queries that power the chat still run from the laptop
    (Phase 1 — will move to server in Phase 2).
  - Overlay drawing stays client-side (lightweight once metadata is local).
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import re
import numpy as np
import io
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_image_zoom import image_zoom
import colorsys
import hashlib
from query_planner import build_query_plan
from plan_query import save_query_plan_to_file
from llm_explainer import explain
import os
import time

from api_client import TileServerClient

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HPC Chatbot", page_icon="💬", layout="wide")
st.title("🧠 HPC Chatbot with Image Upload")

# ---------------------------------------------------------------------------
# Tile server client (talks to FastAPI on HPCC)
# ---------------------------------------------------------------------------
TILE_SERVER_URL = os.getenv("TILE_SERVER_URL", "http://localhost:8000")
client = TileServerClient(TILE_SERVER_URL)

# ---------------------------------------------------------------------------
# Local DB engine — still needed for chat query pipeline (Phase 1)
# ---------------------------------------------------------------------------
DB_USER = "vpandya"
DB_PASS = ""
DB_HOST = "127.0.0.1"
DB_PORT = "5433"
DB_NAME = "hpl_kb"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True,
)


# ---------------------------------------------------------------------------
# Slide registry — fetched once from the tile server
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_slide_list() -> list[str]:
    try:
        return client.list_slides()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tile metadata — fetched from server, cached per slide
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading tile metadata…", ttl=300)
def load_tile_coords_for_slide(slide_id: str) -> pd.DataFrame:
    slide_id = (slide_id or "").strip().upper()
    if not slide_id:
        return pd.DataFrame()
    df = client.get_tiles_meta(slide_id)
    if df.empty:
        return df
    # Normalise columns the same way v21 did
    df.columns = df.columns.astype(str).str.strip()
    if "slide_tile" in df.columns:
        df["slide_tile"] = df["slide_tile"].astype(str).str.strip().str.upper()
    if "slides" in df.columns:
        df["slides"] = df["slides"].astype(str).str.strip().str.upper()
    if "hpc_id" in df.columns:
        df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Adjacency — fetched from server, cached
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_adjacency_for_slide(slide_id: str):
    slide_id = (slide_id or "").strip().upper()
    data = client.get_adjacency(slide_id)
    pair_edge_counts = {}
    for k, v in data.get("pair_edge_counts", {}).items():
        a, b = k.split("_")
        pair_edge_counts[(int(a), int(b))] = v
    tile_neighbor_pairs = {}
    for k, v in data.get("tile_neighbor_pairs", {}).items():
        a, b = k.split("_")
        tile_neighbor_pairs[(int(a), int(b))] = {
            "a_touch": set(v.get("a_touch", [])),
            "b_touch": set(v.get("b_touch", [])),
        }
    return pair_edge_counts, tile_neighbor_pairs


# ---------------------------------------------------------------------------
# Color helpers (unchanged from v21)
# ---------------------------------------------------------------------------

def color_for_hpc(hpc_id):
    if hpc_id is None or pd.isna(hpc_id):
        return (160, 160, 160)
    h = int(hashlib.md5(str(int(hpc_id)).encode()).hexdigest(), 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def color_for_inflammation(label):
    if label is None or pd.isna(label) or str(label).strip() == "":
        return (160, 160, 160)
    mapping = {"none-sparse": (80, 200, 120), "mild-moderate": (255, 200, 0), "marked": (255, 80, 80)}
    return mapping.get(str(label).strip().lower(), (160, 160, 160))


def color_for_necrosis(label):
    if label is None or pd.isna(label) or str(label).strip() == "":
        return (160, 160, 160)
    s = str(label).strip().lower()
    if s == "none": return (60, 200, 120)
    if s == "some": return (255, 165, 0)
    if s == "universal": return (200, 40, 40)
    return (160, 160, 160)


def color_for_malignant(flag):
    if flag is None or pd.isna(flag):
        return (160, 160, 160)
    if isinstance(flag, (int, np.integer)):
        flag = bool(flag)
    if isinstance(flag, str):
        s = flag.strip().lower()
        if s in ("true", "t", "1", "yes", "y"): flag = True
        elif s in ("false", "f", "0", "no", "n"): flag = False
        else: return (160, 160, 160)
    return (255, 80, 80) if flag else (80, 200, 120)


def color_for_adjacency_group(group_name):
    if group_name == "a_touch": return (80, 200, 120)
    if group_name == "b_touch": return (255, 165, 0)
    return (160, 160, 160)


def rgb_to_css(rgb):
    r, g, b = map(int, rgb)
    return f"rgb({r}, {g}, {b})"


# ---------------------------------------------------------------------------
# Intent detection helpers (unchanged from v21)
# ---------------------------------------------------------------------------

def detect_adjacency_intent(q: str):
    q0 = (q or "").lower()
    keywords = ["beside", "besides", "next to", "adjacent", "touching", "near", "around",
                 "cooccur", "co-occur", "co occur", "cooccurrence", "co-occurrence", "co occurrence"]
    if not any(k in q0 for k in keywords):
        return None
    hpcs = re.findall(r"hpc\s*([0-9]+)", q0)
    if len(hpcs) >= 2:
        a, b = int(hpcs[0]), int(hpcs[1])
        if a != b:
            return a, b
    return None


def detect_single_hpc_adjacency_intent(q: str):
    q0 = (q or "").lower()
    adj_words = ["beside", "besides", "next to", "adjacent", "touching", "near", "around", "neighbors"]
    if not any(w in q0 for w in adj_words):
        return None
    hpcs = re.findall(r"hpc\s*([0-9]+)", q0)
    return int(hpcs[0]) if len(hpcs) == 1 else None


def parse_hpc_id(q: str):
    if not q: return None
    m = re.search(r"hpc\s*([0-9]+)", q.lower())
    return int(m.group(1)) if m else None


def parse_inflammation(q: str):
    if not q: return None
    q0 = q.lower()
    if "marked" in q0: return "marked"
    if "mild" in q0 or "moderate" in q0: return "mild-moderate"
    if "non" in q0 and "sparse" in q0: return "none-sparse"
    return None


def parse_necrosis(q: str):
    if not q: return None
    q0 = q.lower()
    if "universal" in q0: return "universal"
    if "some" in q0: return "some"
    if "none" in q0: return "none"
    return None


def parse_highlight_mode(q: str):
    q0 = (q or "").lower()
    if "heatmap" in q0: return "Heatmap"
    if "inflammation" in q0: return "Inflammation"
    if "necrosis" in q0: return "Necrosis"
    if "malignant" in q0: return "Malignant"
    if "adjacent" in q0 or "beside" in q0 or "cooccur" in q0: return "Adjacency"
    if "hpc" in q0 or "cluster" in q0: return "HPC clusters"
    return None


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
        if m: slide_match = m.group(1)
    if not tile_match:
        m = re.search(r"\b(tile[_\-]?\d+|[A-Za-z0-9_\-]+\.jpe?g|[A-Za-z0-9_\-]+\.png|[A-Za-z0-9_\-]+\.tif)\b", q)
        tile_match = m.group(0) if m else None
    if not hpc_match:
        m = re.search(r"\bhpc[-_\s]*([0-9]+)\b", q, re.I)
        hpc_match = m.group(1) if m else None
    return {"tile": tile_match, "slide": slide_match, "sample": sample_match,
            "hpc": [hpc_match] if hpc_match else None}


# ---------------------------------------------------------------------------
# apply_chat_query_to_wsi_state — same logic as v21, uses server-side data
# ---------------------------------------------------------------------------

def apply_chat_query_to_wsi_state(prompt: str, slide_id: str):
    det = detect_entity_patterns(prompt)
    if det and det.get("slide"):
        slide_id = str(det["slide"]).strip().upper()
        st.session_state.active_slide = slide_id
    if not prompt:
        return
    slide_id = str(slide_id).strip().upper()

    st.session_state.query_inflammation = None
    st.session_state.query_necrosis = None
    st.session_state.adj_hpc_a = None
    st.session_state.adj_hpc_b = None
    st.session_state.adj_tile_sets = None

    df = load_tile_coords_for_slide(slide_id)
    if df.empty:
        return

    pair = detect_adjacency_intent(prompt)
    single = detect_single_hpc_adjacency_intent(prompt)

    if pair or single is not None:
        needed = {"slide_tile", "x_native", "y_native", "hpc_id"}
        if needed - set(df.columns):
            return

    if pair:
        a, b = pair
        st.session_state.highlight_mode = "Adjacency"
        st.session_state.adj_hpc_a = a
        st.session_state.adj_hpc_b = b
        pair_edge_counts, tile_has_neighbor_pair = load_adjacency_for_slide(slide_id)
        p = (a, b) if a < b else (b, a)
        st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(p, {"a_touch": set(), "b_touch": set()})
        return

    if single is not None:
        st.session_state.highlight_mode = "Adjacency"
        pair_edge_counts, tile_has_neighbor_pair = load_adjacency_for_slide(slide_id)
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

    hpc_id = parse_hpc_id(prompt)
    infl = parse_inflammation(prompt)
    nec = parse_necrosis(prompt)
    mode = parse_highlight_mode(prompt)
    st.session_state.query_inflammation = infl
    st.session_state.query_necrosis = nec

    q0 = (prompt or "").lower()
    if "heatmap" in q0 and hpc_id is not None:
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


# ---------------------------------------------------------------------------
# HPC annotation renderer (calls tile server API)
# ---------------------------------------------------------------------------

def render_hpc_annotation(hpc_id):
    try:
        h = int(hpc_id)
    except (TypeError, ValueError):
        st.warning(f"Invalid HPC ID: {hpc_id}")
        return
    try:
        info = client.get_hpc_info(h)
    except Exception as e:
        st.warning(f"Could not fetch HPC {h} info: {e}")
        return

    st.markdown(f"### 🧠 HPC {h} Annotation")
    st.markdown("#### HPC Dictionary Entry")
    for k, v in info.items():
        if k in ("malignant_details", "non_malignant_details"):
            continue
        st.write(f"- **{k}**: {v}")

    mal = info.get("malignant_details")
    non = info.get("non_malignant_details")
    if mal:
        st.markdown("#### Malignant Epithelium Details")
        for k, v in mal.items():
            st.write(f"- **{k}**: {v}")
    elif non:
        st.markdown("#### Non Malignant Epithelium Details")
        for k, v in non.items():
            st.write(f"- **{k}**: {v}")
    else:
        st.info("No epithelial phenotype annotations found.")


# ---------------------------------------------------------------------------
# Tile info panel
# ---------------------------------------------------------------------------

def render_tile_info(tile_row, heat_hpc=None):
    st.subheader("Tile info")
    info = {
        "slide_tile": tile_row.get("slide_tile"),
        "tiles": tile_row.get("tiles"),
        "hpc_id": tile_row.get("hpc_id"),
        "inflammation": tile_row.get("inflammation"),
        "necrosis": tile_row.get("necrosis"),
        "malignant": tile_row.get("malignant"),
    }

    def _clean_value(v):
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        return str(v)

    info_df = pd.DataFrame(
        [{"Field": k, "Value": _clean_value(v)} for k, v in info.items()]
    )
    st.dataframe(info_df, width="stretch", hide_index=True)

    if heat_hpc is not None:
        col = f"p_hpc_{int(heat_hpc)}"
        if col in tile_row.index:
            p = tile_row.get(col)
            if p is not None and not pd.isna(p):
                st.metric(f"Heatmap probability for HPC {heat_hpc}", f"{float(p):.4f}")


# ===========================================================================
# show_wsi — THE MAIN WSI VIEWER
#
# Key difference from v21: thumbnail comes from tile server (HTTP JPEG),
# NOT from openslide over sshfs. Tile images on click also come from server.
# ===========================================================================

def show_wsi(slide_id):
    slide_id = (slide_id or "").strip().upper()

    coords_df = load_tile_coords_for_slide(slide_id)
    if coords_df is None or coords_df.empty:
        st.warning(f"No tile coordinates found for slide {slide_id}")
        return

    for key, default in [
        ("selected_tile", None), ("adj_hpc_a", None), ("adj_hpc_b", None),
        ("adj_tile_sets", None), ("query_inflammation", None), ("query_necrosis", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ------------------------------------------------------------------
    # 1. Fetch thumbnail from tile server (cached locally after first hit)
    # ------------------------------------------------------------------
    try:
        info = client.get_slide_info(slide_id)
        w0 = info["level_dimensions"][0]["width"]
        h0 = info["level_dimensions"][0]["height"]
        tile_size_native = int(info["tile_size_native"])
    except Exception as e:
        st.error(f"Cannot reach tile server for slide info: {e}")
        return

    thumb_width = 3000
    try:
        base_region = client.get_thumbnail(slide_id, max_width=thumb_width)
    except Exception as e:
        st.error(f"Cannot fetch thumbnail from tile server: {e}")
        return

    downsample = w0 / base_region.size[0]

    # ------------------------------------------------------------------
    # 2. Build working dataframe
    # ------------------------------------------------------------------
    df = coords_df.copy()
    base_cols = ["tiles", "slides", "x_native", "y_native", "h5_index", "hpc_id",
                 "inflammation", "necrosis", "malignant", "slide_tile"]
    prob_cols = [c for c in df.columns if str(c).startswith("p_hpc_")]
    cols_to_keep = [c for c in base_cols if c in df.columns] + prob_cols
    df = df[cols_to_keep].copy()

    if df.empty:
        st.warning(f"No tile data for slide {slide_id}")
        return

    # ------------------------------------------------------------------
    # 3. Grid overlay controls
    # ------------------------------------------------------------------
    show_grid = st.checkbox("Show tile grid overlays", value=True, key="grid-toggle")

    if "highlight_mode" not in st.session_state:
        st.session_state.highlight_mode = "HPC clusters"

    st.radio("Highlight mode",
             options=["HPC clusters", "Inflammation", "Necrosis", "Malignant", "Adjacency", "Heatmap"],
             key="highlight_mode", horizontal=True)

    # Heatmap controls
    if st.session_state.highlight_mode == "Heatmap":
        hpc_prob_cols = [c for c in df.columns if c.startswith("p_hpc_")]
        if not hpc_prob_cols:
            st.warning("No p_hpc_* probability columns found for this slide.")
        else:
            hpc_ids_heat = sorted(int(c.split("_")[-1]) for c in hpc_prob_cols)
            st.selectbox("Heatmap HPC", options=hpc_ids_heat, index=0, key="heat_hpc")
            st.slider("Heat intensity", 0.1, 1.0, 0.6, 0.05, key="heat_alpha")

    # Legend blocks (same HTML as v21)
    _render_legend(st.session_state.get("highlight_mode", "HPC clusters"))

    df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")

    # ------------------------------------------------------------------
    # 4. Draw overlay
    # ------------------------------------------------------------------
    highlight_mode = st.session_state.get("highlight_mode", "HPC clusters")
    selected_hpc = st.session_state.get("selected_hpc")

    if selected_hpc is not None and highlight_mode == "HPC clusters":
        grid_df = df[df["hpc_id"] == selected_hpc]
    else:
        grid_df = df

    base_rgba = base_region.convert("RGBA")
    overlay = base_rgba.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    heat_layer = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    heat_draw = ImageDraw.Draw(heat_layer, "RGBA")

    # HPC filter panel
    if "selected_hpc" not in st.session_state:
        st.session_state.selected_hpc = None
    hpc_list = sorted(df["hpc_id"].dropna().astype(int).unique().tolist())

    # Adjacency controls
    if st.session_state.highlight_mode == "Adjacency":
        pair_edge_counts, tile_has_neighbor_pair = load_adjacency_for_slide(slide_id)
        with st.expander("Adjacency and cooccurrence controls", expanded=False):
            top_pairs = sorted(pair_edge_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
            if top_pairs:
                options = [f"HPC {a} ↔ HPC {b} ({cnt} edges)" for (a, b), cnt in top_pairs]
                selected_pair = st.selectbox("Pick a top cooccurring pair", options, key="adj_top_pair_select")
                idx = options.index(selected_pair)
                (a_sel, b_sel), _ = top_pairs[idx]
                if st.button("Highlight selected top pair", key="adj_top_pair_btn"):
                    p = (a_sel, b_sel) if a_sel < b_sel else (b_sel, a_sel)
                    st.session_state.adj_hpc_a = a_sel
                    st.session_state.adj_hpc_b = b_sel
                    st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(p, {"a_touch": set(), "b_touch": set()})
                    st.rerun()
            else:
                st.info("No adjacent cross HPC pairs found on this slide.")

    # HPC cluster expander
    with st.expander("Highlight tiles by HPC cluster", expanded=False):
        if st.button("Show all HPCs"):
            st.session_state.selected_hpc = None
            st.rerun()
        cols = st.columns(4)
        for i, hpc_id in enumerate(hpc_list):
            color = color_for_hpc(hpc_id)
            css_color = rgb_to_css(color)
            with cols[i % 4]:
                st.markdown(f'<div style="display:flex;align-items:center;padding:6px;"><div style="width:14px;height:14px;background:{css_color};border:1px solid #000;margin-right:6px;"></div><span>HPC {hpc_id}</span></div>', unsafe_allow_html=True)
                if st.button(f"HPC_{hpc_id}", key=f"hpc_btn_{hpc_id}"):
                    st.session_state.selected_hpc = hpc_id
                    st.rerun()

    selected_hpc = st.session_state.get("selected_hpc")
    if selected_hpc is not None:
        with st.expander("🧠 HPC Biological Interpretation", expanded=True):
            render_hpc_annotation(selected_hpc)

    # Filters
    infl_f = st.session_state.get("query_inflammation")
    nec_f = st.session_state.get("query_necrosis")
    filtered_df = df.copy()
    if st.session_state.get("selected_hpc") is not None:
        filtered_df = filtered_df[filtered_df["hpc_id"] == st.session_state.selected_hpc]
    if infl_f is not None:
        filtered_df = filtered_df[filtered_df["inflammation"].astype(str).str.lower().str.strip() == infl_f]
    if nec_f is not None:
        filtered_df = filtered_df[filtered_df["necrosis"].astype(str).str.lower().str.strip() == nec_f]
    st.caption(f"Matched tiles: {len(filtered_df)}")

    # ------------------------------------------------------------------
    # 4a. Heatmap overlay (vectorized, same as v21)
    # ------------------------------------------------------------------
    if highlight_mode == "Heatmap":
        h = int(st.session_state.get("heat_hpc", 0))
        col_name = f"p_hpc_{h}"
        if col_name in df.columns:
            _draw_heatmap(
                heat_draw,
                df,
                col_name,
                downsample,
                float(st.session_state.get("heat_alpha", 0.6)),
                tile_size_native,
            )

    # ------------------------------------------------------------------
    # 4b. Grid outlines (vectorized where possible)
    # ------------------------------------------------------------------
    if show_grid and highlight_mode != "Heatmap":
        _draw_grid(
    draw,
    grid_df if selected_hpc is not None and highlight_mode not in ("Adjacency", "Heatmap") else df,
    downsample,
    highlight_mode,
    infl_f,
    nec_f,
    tile_size_native,
)

    # Selected tile highlight
    sel = st.session_state.get("selected_tile")
    if sel and "x_native" in sel and "y_native" in sel:
        sx = int(sel["x_native"] / downsample)
        sy = int(sel["y_native"] / downsample)
        ts = int(tile_size_native / downsample)
        draw.rectangle([sx - 3, sy - 3, sx + ts + 3, sy + ts + 3], outline="yellow", width=15)
        draw.rectangle([sx, sy, sx + ts, sy + ts], outline="lime", width=15)

    if st.session_state.highlight_mode == "Heatmap":
        overlay = Image.alpha_composite(overlay, heat_layer)

    overlay_np = np.asarray(overlay.convert("RGB"), dtype=np.uint8)

    # ------------------------------------------------------------------
    # 5. Viewer
    # ------------------------------------------------------------------
    st.subheader("Slide Viewer")
    mode = st.radio("Choose viewer mode:", ["Zoom", "Click"], horizontal=True, key="viewer-mode")

    if mode == "Zoom":
        image_zoom(np.asarray(overlay_np, dtype=np.uint8), zoom_factor=2, keep_aspect_ratio=True)
        return

    st.write("Click a tile to view it.")
    click = streamlit_image_coordinates(overlay_np, key="wsi-click-coords")
    if not click:
        st.info("Click anywhere on the slide to select a tile.")
        return

    cx, cy = click["x"], click["y"]
    native_x = cx * downsample
    native_y = cy * downsample

    # Vectorized tile lookup (fast)
    tol = tile_size_native  * 0.1
    mask = (
        (df["x_native"].astype(float) - tol <= native_x) &
        (native_x <= df["x_native"].astype(float) + tile_size_native + tol) &
        (df["y_native"].astype(float) - tol <= native_y) &
        (native_y <= df["y_native"].astype(float) + tile_size_native+ tol)
    )
    matches = df[mask]
    if matches.empty:
        st.warning("Clicked area does not match any tile.")
        return
    tile_row = matches.iloc[0]

    new_selected = {
        "slide_tile": str(tile_row.get("slide_tile", "")),
        "x_native": float(tile_row["x_native"]),
        "y_native": float(tile_row["y_native"]),
    }
    prev = st.session_state.get("selected_tile")
    same_tile = (isinstance(prev, dict)
                 and prev.get("slide_tile") == new_selected["slide_tile"]
                 and float(prev.get("x_native", -1)) == new_selected["x_native"]
                 and float(prev.get("y_native", -1)) == new_selected["y_native"])
    if not same_tile:
        st.session_state.selected_tile = new_selected
        st.rerun()

    # ------------------------------------------------------------------
    # 6. Show selected tile image — fetched from tile server
    # ------------------------------------------------------------------
    slide_tile_key = str(tile_row.get("slide_tile", ""))
    tile_name = tile_row.get("tiles", slide_tile_key)
    st.success(f"Tile selected: {tile_name}")

    try:
        tile_img = client.get_tile_image(slide_tile_key)
        st.image(tile_img, caption=f"{tile_name}", width="stretch")
    except Exception as e:
        st.warning(f"Could not load tile image: {e}")

    heat_hpc = st.session_state.get("heat_hpc") if st.session_state.highlight_mode == "Heatmap" else None
    render_tile_info(tile_row, heat_hpc=heat_hpc)

    hpc_id = tile_row.get("hpc_id")
    if hpc_id is not None and not pd.isna(hpc_id):
        render_hpc_annotation(int(hpc_id))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_heatmap(heat_draw, df, col_name, downsample, alpha_max, tile_size_native):
    p_series = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
    xs = (df["x_native"].astype(float) / downsample).astype(int).to_numpy()
    ys = (df["y_native"].astype(float) / downsample).astype(int).to_numpy()
    ps = p_series.to_numpy(dtype=np.float32)
    valid = np.isfinite(ps)
    if not valid.any():
        return
    pmin, pmax = float(np.nanmin(ps[valid])), float(np.nanmax(ps[valid]))
    if pmax <= pmin:
        pmax = pmin + 1e-9
    epsilon = 1e-6
    denom = np.log10(pmax + epsilon) - np.log10(pmin + epsilon)
    if denom == 0: denom = 1e-9
    t_all = np.clip((np.log10(ps + epsilon) - np.log10(pmin + epsilon)) / denom, 0.0, 1.0)
    alpha_min = 0.10
    alpha_max = max(0.2, min(alpha_max, 0.95))
    a_all = alpha_min + (alpha_max - alpha_min) * t_all
    alpha_all = (255.0 * a_all).astype(np.uint8)
    ts = int(tile_size_native / downsample)
    for x, y, t, alpha in zip(xs[valid], ys[valid], t_all[valid], alpha_all[valid]):
        R = int(255 * float(t))
        G = 255
        B = int(255 * (1 - float(t)))
        heat_draw.rectangle([int(x), int(y), int(x) + ts, int(y) + ts],
                            fill=(R, G, B, int(alpha)), outline=(255, 255, 255, 40))


def _draw_grid(draw, grid_df, downsample, highlight_mode, infl_f, nec_f, tile_size_native):
    ts = int(tile_size_native / downsample)
    # Pre-extract numpy arrays for speed
    xs = (grid_df["x_native"].astype(float) / downsample).astype(int).to_numpy()
    ys = (grid_df["y_native"].astype(float) / downsample).astype(int).to_numpy()

    for i, (x, y) in enumerate(zip(xs, ys)):
        r = grid_df.iloc[i]
        if infl_f is not None and str(r.get("inflammation", "")).strip().lower() != infl_f:
            continue
        if nec_f is not None and str(r.get("necrosis", "")).strip().lower() != nec_f:
            continue
        if highlight_mode == "Inflammation":
            color = color_for_inflammation(r.get("inflammation"))
        elif highlight_mode == "Necrosis":
            color = color_for_necrosis(r.get("necrosis"))
        elif highlight_mode == "Malignant":
            color = color_for_malignant(r.get("malignant"))
        elif highlight_mode == "Adjacency":
            adj = st.session_state.get("adj_tile_sets")
            if adj is None: continue
            tkey = str(r.get("slide_tile"))
            if tkey in adj.get("a_touch", set()):
                color = color_for_adjacency_group("a_touch")
            elif tkey in adj.get("b_touch", set()):
                color = color_for_adjacency_group("b_touch")
            else:
                continue
        else:
            color = color_for_hpc(r.get("hpc_id"))
        draw.rectangle([x, y, x + ts, y + ts], outline=color, width=5)


def _render_legend(mode):
    if mode == "Inflammation":
        st.markdown('<div style="display:flex;gap:18px;padding:6px 0;"><div><span style="width:14px;height:14px;background:#50C878;border:1px solid #000;display:inline-block;"></span> None-sparse</div><div><span style="width:14px;height:14px;background:#FFC800;border:1px solid #000;display:inline-block;"></span> Mild-moderate</div><div><span style="width:14px;height:14px;background:#FF5050;border:1px solid #000;display:inline-block;"></span> Marked</div><div><span style="width:14px;height:14px;background:#A0A0A0;border:1px solid #000;display:inline-block;"></span> Missing</div></div>', unsafe_allow_html=True)
    elif mode == "Necrosis":
        st.markdown('<div style="display:flex;gap:18px;padding:6px 0;"><div><span style="width:14px;height:14px;background:#3CC878;border:1px solid #000;display:inline-block;"></span> None</div><div><span style="width:14px;height:14px;background:#FFA500;border:1px solid #000;display:inline-block;"></span> Some</div><div><span style="width:14px;height:14px;background:#C82828;border:1px solid #000;display:inline-block;"></span> Universal</div><div><span style="width:14px;height:14px;background:#A0A0A0;border:1px solid #000;display:inline-block;"></span> Missing</div></div>', unsafe_allow_html=True)
    elif mode == "Malignant":
        st.markdown('<div style="display:flex;gap:18px;padding:6px 0;"><div><span style="width:14px;height:14px;background:#FF5050;border:1px solid #000;display:inline-block;"></span> Malignant</div><div><span style="width:14px;height:14px;background:#50C878;border:1px solid #000;display:inline-block;"></span> Non-malignant</div><div><span style="width:14px;height:14px;background:#A0A0A0;border:1px solid #000;display:inline-block;"></span> Missing</div></div>', unsafe_allow_html=True)
    elif mode == "Adjacency":
        a = st.session_state.get("adj_hpc_a")
        b = st.session_state.get("adj_hpc_b")
        st.markdown(f'<div style="display:flex;gap:18px;padding:6px 0;"><div><span style="width:14px;height:14px;background:#50C878;border:1px solid #000;display:inline-block;"></span> HPC {a} touching HPC {b}</div><div><span style="width:14px;height:14px;background:#FFA500;border:1px solid #000;display:inline-block;"></span> HPC {b} touching HPC {a}</div></div>', unsafe_allow_html=True)


# ===========================================================================
# Chat pipeline — DB queries still local (Phase 1)
# ===========================================================================
# The full fetch_answer_from_db / handle_* functions are imported from v21's
# logic. For brevity in v22 we keep the NL pipeline as-is. The heavy image
# and metadata paths have already been moved to the tile server above.
# ---------------------------------------------------------------------------

def classify_malignancy_polarity(q: str):
    q = q.lower()
    q_norm = re.sub(r'[_\-]+', ' ', q)
    neg_patterns = [r'\bnon\s*malignant\b', r'\bnot\s+malignant\b', r"\bwithout\s+malignant\b", r"\bno\s+malignant\b"]
    if any(re.search(p, q_norm) for p in neg_patterns):
        return 'non'
    if re.search(r'\bmalignant\b', q_norm):
        return 'malignant'
    return None


def fetch_answer_from_db(query: str):
    """Simplified — keeps the same DB query logic from v21."""
    q = (query or "").lower()
    polarity = classify_malignancy_polarity(q)
    is_malignant_question = polarity in ("malignant", "non")

    with engine.connect() as conn:
        if any(kw in q for kw in ["how many", "count", "total hpcs", "number of hpcs"]):
            return _handle_count_hpcs(conn, q, polarity)
        if any(kw in q for kw in ["survival", "cox", "hazard ratio"]):
            return _handle_survival_brief(conn, q)

    detected = detect_entity_patterns(query)
    parts = []
    if detected.get("slide"):
        slide_id = re.sub(r"^slide\s+", "", str(detected["slide"]), flags=re.I).strip().upper()
        parts.append(f"Showing slide **{slide_id}**.")
        st.session_state.active_slide = slide_id
    if detected.get("hpc"):
        for h in detected["hpc"]:
            try:
                info = client.get_hpc_info(int(h))
                parts.append(f"**HPC {h}** — malignant: {info.get('malignant')}, inflammation: {info.get('inflammation')}, necrosis: {info.get('necrosis')}")
            except Exception:
                parts.append(f"HPC {h} not found.")
    if not parts:
        return "Please specify a valid tile, slide, sample, or HPC ID."
    return "\n\n".join(parts)


def _handle_count_hpcs(conn, q, polarity):
    result = conn.execute(text("""
        SELECT
            SUM(CASE WHEN malignant IS TRUE THEN 1 ELSE 0 END) AS mal,
            SUM(CASE WHEN malignant IS FALSE THEN 1 ELSE 0 END) AS non_mal,
            COUNT(*) AS total
        FROM hpc_dictionary
    """)).fetchone()
    total, mal, non_mal = int(result.total), int(result.mal), int(result.non_mal)
    if polarity == "malignant":
        return f"There are **{mal} malignant HPCs** out of {total}."
    if polarity == "non":
        return f"There are **{non_mal} non-malignant HPCs** out of {total}."
    return f"Total HPCs: **{total}** (malignant: {mal}, non-malignant: {non_mal})"


def _handle_survival_brief(conn, q):
    m = re.search(r"hpc\s*([0-9]+)", q, re.I)
    if not m:
        return "Please specify an HPC ID for survival analysis."
    hpc_id = int(m.group(1))
    try:
        data = client.get_hpc_survival(hpc_id)
        return (f"### Survival — HPC {hpc_id}\n"
                f"- HR: {data.get('expcoef')}\n- p-value: {data.get('p')}\n"
                f"- 95% CI: ({data.get('expcoef_lower_95')}, {data.get('expcoef_upper_95')})")
    except Exception:
        return f"No survival data for HPC {hpc_id}."


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! 👋 Ask me about your H&E cancer slides — I'll fetch insights instantly."}
    ]
if "active_slide" not in st.session_state:
    st.session_state.active_slide = None
if "viewer_open" not in st.session_state:
    st.session_state.viewer_open = False


def _ensure_active_slide(prompt_text):
    det = detect_entity_patterns(prompt_text or "")
    slide_from_prompt = det.get("slide") if det else None
    if slide_from_prompt:
        st.session_state.active_slide = str(slide_from_prompt).strip().upper()
    if not st.session_state.active_slide:
        slides = load_slide_list()
        if slides:
            st.session_state.active_slide = slides[0]
    return st.session_state.active_slide


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something about HPCs...", key="main_chat_input")

rerun_needed = False
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    slide_id = _ensure_active_slide(prompt)
    apply_chat_query_to_wsi_state(prompt, slide_id)
    st.session_state.viewer_open = True

    plan = build_query_plan(prompt)
    path = save_query_plan_to_file(
        plan, slide_id=plan["entities"]["slide"][0] if plan["entities"]["slide"] else None
    )

    with st.expander("🧩 Query Plan (debug)", expanded=False):
        st.json(plan)

    if plan["intent"] in ("general_query", "greeting", "help"):
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
    st.rerun()


# ---------------------------------------------------------------------------
# Slide picker + viewer
# ---------------------------------------------------------------------------
slide_options = sorted(load_slide_list())
if not slide_options:
    st.warning("No slides available from tile server.")
    slide_id = None
else:
    current = (st.session_state.get("active_slide") or "").strip().upper()
    default_idx = slide_options.index(current) if current in slide_options else 0
    slide_id = st.selectbox("Choose a slide", slide_options, index=default_idx, key="slide_selectbox")
    st.session_state.active_slide = slide_id

if st.session_state.viewer_open:
    if st.button("Close Slide Viewer"):
        st.session_state.viewer_open = False
        rerun_needed = True
else:
    if slide_id and st.button(f"Open Slide Viewer for {slide_id}"):
        st.session_state.viewer_open = True
        rerun_needed = True

if st.session_state.viewer_open and slide_id:
    show_wsi(slide_id)

if rerun_needed:
    st.rerun()
