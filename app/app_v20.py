import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import text, inspect
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

st.set_page_config(page_title="HPC Chatbot", page_icon="💬", layout="wide")
st.title("🧠 HPC Chatbot with Image Upload")

# --- DB connection ---
DB_USER = "vaidehipandya"         
DB_PASS = "vjp007"  
DB_HOST = "localhost"       
DB_PORT = "5432"
DB_NAME = "hpl_kb"
engine = create_engine("postgresql+psycopg2:///hpl_kb")

df_heatmap = pd.read_csv("heatmap_table.csv")

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
    q0 = q.lower()

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
    q0 = q.lower()
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
    q0 = q.lower()
    if "universal" in q0:
        return "universal"
    if "some" in q0:
        return "some"
    if "none" in q0:
        return "none"
    return None

def parse_highlight_mode(q: str):
    q0 = (q or "").lower()
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
    # --- reset per-query filters so old questions don't leak ---
    st.session_state.query_inflammation = None
    st.session_state.query_necrosis = None

    # optional: reset these unless the prompt explicitly asks for them
    st.session_state.selected_hpc = None

    # clear adjacency unless this prompt is adjacency
    st.session_state.adj_hpc_a = None
    st.session_state.adj_hpc_b = None
    st.session_state.adj_tile_sets = None

    df = tile_coords[tile_coords["slides"] == slide_id]

    if "query_inflammation" not in st.session_state:
        st.session_state.query_inflammation = None
    if "query_necrosis" not in st.session_state:
        st.session_state.query_necrosis = None
    if "selected_hpc" not in st.session_state:
        st.session_state.selected_hpc = None
    if "highlight_mode" not in st.session_state:
        st.session_state.highlight_mode = "HPC clusters"

    pair = detect_adjacency_intent(prompt)
    single = detect_single_hpc_adjacency_intent(prompt)

    if pair:
        a, b = pair
        st.session_state.highlight_mode = "Adjacency"
        st.session_state.adj_hpc_a = a
        st.session_state.adj_hpc_b = b

        df_min = df[["tiles", "x_native", "y_native", "hpc_id"]].copy()
        pair_edge_counts, tile_has_neighbor_pair = compute_adjacency_cached(df_min)
        p = (a, b) if a < b else (b, a)
        st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(p, {"a_touch": set(), "b_touch": set()})

        st.session_state.selected_hpc = None
        return

    if single is not None:
        st.session_state.highlight_mode = "Adjacency"

    #    ✅ compute adjacency for this slide (minimal df for stable cache)
        df_min = df[["tiles", "x_native", "y_native", "hpc_id"]].copy()
        pair_edge_counts, tile_has_neighbor_pair = compute_adjacency_cached(df_min)

        # pick the strongest partner pair that involves this HPC
        candidates = [((a, b), cnt) for (a, b), cnt in pair_edge_counts.items()
                  if a == single or b == single]

        if candidates:
            (a_sel, b_sel), _ = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
            st.session_state.adj_hpc_a = a_sel
            st.session_state.adj_hpc_b = b_sel

            p = (a_sel, b_sel) if a_sel < b_sel else (b_sel, a_sel)
            st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(
                p, {"a_touch": set(), "b_touch": set()}
            )
        else:
            # optional but helpful: clear old adjacency selection
            st.session_state.adj_hpc_a = None
            st.session_state.adj_hpc_b = None
            st.session_state.adj_tile_sets = {"a_touch": set(), "b_touch": set()}

        st.session_state.selected_hpc = None
        return

    hpc_id = parse_hpc_id(prompt)
    infl = parse_inflammation(prompt)
    nec = parse_necrosis(prompt)
    mode = parse_highlight_mode(prompt)

    st.session_state.query_inflammation = infl
    st.session_state.query_necrosis = nec

    # If user asked for a specific HPC, keep HPC mode so we can intersect filters
    if hpc_id is not None:
        st.session_state.selected_hpc = hpc_id

# allow mode to be set even if HPC is present
    if mode:
        st.session_state.highlight_mode = mode
    else:
        # default if they only said "HPC 31"
        if hpc_id is not None:
            st.session_state.highlight_mode = "HPC clusters"

    # Otherwise allow mode switch (Inflammation, Necrosis, Malignant, Adjacency)
    if mode:
        st.session_state.highlight_mode = mode


def render_hpc_annotation(hpc_id):
    hpc_id = str(hpc_id).strip()

    with engine.connect() as conn:

        dict_row = conn.execute(
            text("SELECT * FROM hpc_dictionary WHERE hpc_id = :h"),
            {"h": hpc_id}
        ).fetchone()

        mal_row = conn.execute(
            text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :h"),
            {"h": hpc_id}
        ).fetchone()

        non_row = conn.execute(
            text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :h"),
            {"h": hpc_id}
        ).fetchone()

    # ------------------------------
    # DISPLAY
    # ------------------------------
    st.markdown(f"### 🧠 HPC {hpc_id} Annotation")

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

# -------------------------

def render_hpc_annotation_for_tile(tile_row):

    tile_name = str(tile_row["tiles"]).strip()

    with engine.connect() as conn:
        hpc_id = conn.execute(
            text("""
                SELECT hpc_id 
                FROM tile_registry
                WHERE tiles = :t
            """),
            {"t": tile_name}
        ).scalar()

    if hpc_id is None:
        st.info("This tile has no HPC ID mapping in tile_registry.")
        return

    st.markdown(f"### 🔍 Annotation for Tile `{tile_name}`")
    st.markdown(f"**Mapped HPC ID:** `{hpc_id}`")

    # Delegate ALL rendering to the HPC-level function
    render_hpc_annotation(hpc_id)





WSI_MAP = {
    "TCGA-55-7574-01Z-00-DX1":
        "/Users/vaidehipandya/Desktop/Work/TCGA-55-7574-01Z-00-DX1.09639e6a-d85f-4d84-abbc-0f5a6d679683.svs"
}

tile_coords = pd.read_csv("/Users/vaidehipandya/Desktop/Work/tile_coords_with_h5_index.csv")

TILE_SIZE_5X = 224
SCALE = 1.8 / 0.252
TILE_SIZE_NATIVE = int(TILE_SIZE_5X * SCALE)

H5_PATH = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"

def _grid_xy(x_native, y_native):
    gx = int(round(float(x_native) / float(TILE_SIZE_NATIVE)))
    gy = int(round(float(y_native) / float(TILE_SIZE_NATIVE)))
    return gx, gy

def _neighbors_8(gx, gy):
    return [
        (gx - 1, gy - 1), (gx, gy - 1), (gx + 1, gy - 1),
        (gx - 1, gy),                 (gx + 1, gy),
        (gx - 1, gy + 1), (gx, gy + 1), (gx + 1, gy + 1),
    ]



from collections import defaultdict

def compute_adjacency_and_cooccurrence(df_slide):
    """
    df_slide must contain for one slide:
    tiles, x_native, y_native, hpc_id

    Returns
    pair_edge_counts: dict[(a,b)] -> count of adjacency edges
    tile_has_neighbor_pair: dict[(a,b)] -> {"a_touch": set(tile_ids), "b_touch": set(tile_ids)}
    """
    df2 = df_slide.copy()

    df2["hpc_id"] = pd.to_numeric(df2["hpc_id"], errors="coerce")
    df2 = df2.dropna(subset=["hpc_id", "x_native", "y_native"])
    df2["hpc_id"] = df2["hpc_id"].astype(int)

    pos_to_row = {}
    for _, r in df2.iterrows():
        gx, gy = _grid_xy(r["x_native"], r["y_native"])
        pos_to_row[(gx, gy)] = r

    pair_edge_counts = {}               # plain dict
    tile_has_neighbor_pair = {}         # plain dict

    visited_edges = set()

    for (gx, gy), r in pos_to_row.items():
        a = int(r["hpc_id"])
        tile_a = str(r["tiles"])

        for nb in _neighbors_8(gx, gy):
            if nb not in pos_to_row:
                continue

            r2 = pos_to_row[nb]
            b = int(r2["hpc_id"])
            tile_b = str(r2["tiles"])

            if a == b:
                continue

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
def compute_adjacency_cached(df_slide):
    return compute_adjacency_and_cooccurrence(df_slide)


def load_tile_from_h5(index):
    with h5py.File(H5_PATH, "r") as f:
        tile = f["train_img"][index]
    tile = np.squeeze(tile).astype(np.uint8)
    return Image.fromarray(tile)



def load_tile_registry():
    try:
        query = """
            SELECT slides, tiles, hpc_id
            FROM tile_registry
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading tile registry: {e}")
        return None
    
    
def get_min_adj_df(df):
    return df[["tiles", "x_native", "y_native", "hpc_id"]].copy()

def load_hpc_dictionary():
    try:
        query = """
            SELECT hpc_id, inflammation, necrosis, malignant
            FROM hpc_dictionary
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error loading hpc dictionary: {e}")
        return None
    

tile_registry = load_tile_registry()
hpc_dict = load_hpc_dictionary()

if tile_registry is not None:
    tile_coords = tile_coords.merge(
        tile_registry,
        on=["slides", "tiles"],
        how="left"
    )

if hpc_dict is not None:
    tile_coords = tile_coords.merge(
        hpc_dict,
        on="hpc_id",
        how="left"
    )


def color_for_hpc(hpc_id):
    # Default color for missing HPC
    if hpc_id is None:
        return (255, 0, 0)  # red as RGB

    h = int(hashlib.md5(str(hpc_id).encode()).hexdigest(), 16)
    hue = (h % 360) / 360.0

    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


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
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"



def render_tile_preview(slide_id):
    with engine.connect() as conn:
        tile_rows = conn.execute(
            text("""
                SELECT tiles, image_index
                FROM tile_registry
                WHERE slides ILIKE :slide_id
            """),
            {"slide_id": slide_id}
        ).fetchall()

    if not tile_rows:
        return

    with st.expander(f"📂 Preview Tiles ({len(tile_rows)} tiles)", expanded=False):
        try:
            with h5py.File(H5_PATH, "r") as f:
                dataset = f["train_img"]
                cols = st.columns(3)

                for i, row in enumerate(tile_rows):
                    idx = row.image_index
                    if idx is None or idx >= dataset.shape[0]:
                        continue

                    tile = dataset[idx]
                    tile = np.squeeze(tile)
                    tile = (tile - tile.min()) / (tile.max() - tile.min() + 1e-5) * 255
                    tile = tile.astype(np.uint8)

                    img = Image.fromarray(tile)
                    cols[i % 3].image(
                        img,
                        caption=f"{row.tiles} (index {idx})",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error loading tiles: {e}")


            
def show_wsi(slide_id):

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

    wsi_path = WSI_MAP[slide_id]

    try:
        slide = openslide.OpenSlide(wsi_path)

        # 1. Pick display level (~3000 px width)
        target_width = 3000
        level = min(
            range(len(slide.level_dimensions)),
            key=lambda lvl: abs(slide.level_dimensions[lvl][0] - target_width)
        )
        level_dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]

        # 2. Load WSI base image
        base_region = slide.read_region((0, 0), level, level_dims).convert("RGB")

        # ------------------------------
        # 3. GRID OVERLAY + CLICK HIGHLIGHT
        # ------------------------------
        show_grid = st.checkbox("Show tile grid overlays", value=True, key="grid-toggle")

        if "highlight_mode" not in st.session_state:
            st.session_state.highlight_mode = "HPC clusters"

        st.radio(
            "Highlight mode",
            options=["HPC clusters", "Inflammation", "Necrosis", "Malignant", "Adjacency"],
            key="highlight_mode",
            horizontal=True
        )

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
        overlay = base_region.copy()
        draw = ImageDraw.Draw(overlay)

        df = tile_coords.loc[tile_coords["slides"] == slide_id, 
                     ["tiles", "slides", "x_native", "y_native", "h5_index", "hpc_id", "inflammation", "necrosis", "malignant"]
                    ].copy()

# HPC FILTER PANEL (COLORED)
        if "selected_hpc" not in st.session_state:
            st.session_state.selected_hpc = None

        df["hpc_id"] = df["hpc_id"].astype("Int64")
        hpc_list = sorted(df["hpc_id"].dropna().unique().tolist())


        if st.session_state.highlight_mode == "Adjacency":
            df_min = get_min_adj_df(df)
            pair_edge_counts, tile_has_neighbor_pair = compute_adjacency_cached(df_min)

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

        if show_grid:

            highlight_mode = st.session_state.get("highlight_mode", "HPC clusters")
            selected_hpc = st.session_state.get("selected_hpc", None)

            # If an HPC is selected, restrict drawing to that HPC for every mode except Adjacency
            if selected_hpc is not None and highlight_mode != "Adjacency":
                grid_df = df[df["hpc_id"] == selected_hpc]
            else:
                grid_df = df

            for _, r in grid_df.iterrows():
                x = int(r["x_native"] / downsample)
                y = int(r["y_native"] / downsample)
                ts = int(TILE_SIZE_NATIVE / downsample)


                # If user asked for a specific inflammation state, only draw those tiles
                if infl_f is not None:
                    val = str(r.get("inflammation","")).strip().lower()
                    if val != infl_f:
                        continue

                # If user asked for a specific necrosis state, only draw those tiles
                if nec_f is not None:
                    val = str(r.get("necrosis","")).strip().lower()
                    if val != nec_f:
                        continue


                if highlight_mode == "Inflammation":
                    color = color_for_inflammation(r.get("inflammation", None))

                elif highlight_mode == "Necrosis":
                    color = color_for_necrosis(r.get("necrosis", None))

                elif highlight_mode == "Malignant":
                    color = color_for_malignant(r.get("malignant", None))

                elif highlight_mode == "Adjacency":
                    adj = st.session_state.get("adj_tile_sets", None)
                    if adj is None:
                        continue

                    tname = str(r.get("tiles"))

                    # Skip tiles that are not part of the A-B adjacency set
                    if (tname not in adj.get("a_touch", set())) and (tname not in adj.get("b_touch", set())):
                        continue

                    if tname in adj.get("a_touch", set()):
                        color = color_for_adjacency_group("a_touch")
                    else:
                        color = color_for_adjacency_group("b_touch")

                else:
                    color = color_for_hpc(r.get("hpc_id", None))

                draw.rectangle(
                    [x, y, x + ts, y + ts],
                    outline=color,
                    width=5
                )

        # Draw highlight on selected tile
        sel = st.session_state.get("selected_tile")
        if sel is not None:
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

        overlay_np = np.array(overlay)

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
            "x_native": tile_row["x_native"],
            "y_native": tile_row["y_native"],
        }

        if st.session_state.selected_tile != new_selected:
            st.session_state.selected_tile = new_selected
            st.rerun()   # instant highlight refresh

        # ------------------------------
        # 9. Display selected tile image
        # ------------------------------
        idx = int(tile_row["h5_index"])
        tile_name = tile_row["tiles"]

        st.success(f"Tile selected: {tile_name}")

        tile_img = load_tile_from_h5(idx)
        st.image(tile_img, caption=f"{tile_name} (index {idx})")

        # ------------------------------
        # 10. Show HPC annotation for this tile
        # ------------------------------
        render_hpc_annotation_for_tile(tile_row)
        
    except Exception as e:
        st.error(f"WSI error: {e}")



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
        m = re.search(r"\b(TCGA-[A-Z0-9\-]+DX\d+)\b", query, re.I)
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

def fetch_answer_from_db(query):
    q = query.lower()
    conn = engine.connect()
    insp = inspect(engine)

    is_malignant_question = "malignant epithelium" in q or "not malignant epithelium" in q
    is_negative = "not malignant epithelium" in q  # i.e. asking for non-malignant explicitly

    if any(kw in q for kw in ["how many", "count", "total hpcs", "number of hpcs"]):
        return handle_analytics(conn, q, "count_hpcs")

    if any(kw in q for kw in ["portion", "coverage", "covered by malignant", "percent malignant"]):
        return handle_analytics(conn, q, "slide_malignant_coverage")

    if any(kw in q for kw in ["survival", "cox", "hazard ratio", "p-value", "significant", "regression"]):
        return handle_survival(conn, q)
    
    detected = detect_entity_patterns(query)

    intents = {
        "malignant": is_malignant_question,
        "negative": is_negative,        # tells us to fetch from non_malignant_details
        "tile": detected.get("tile"),
        "slide": detected.get("slide"),
        "sample": detected.get("sample"),
        "hpc": detected.get("hpc"),
        "survival": None  #  important to prevent KeyError
    }
# Add dynamically detected entities
    intents.update(detect_entity_patterns(query))
    output_parts = []

    # --- Dispatch table: entity -> handler ---
    handlers = {
    "tile": handle_tile,
    "slide": handle_slide,
    "sample": handle_sample,
    "hpc": handle_hpc,
    "survival": handle_survival
}

    # --- Run handlers dynamically ---
    for entity, func in handlers.items():
        match = intents[entity]
        if match:
            part = func(conn, match, intents)
            if part:
                output_parts.append(part)

    conn.close()

    if not output_parts:
        return "Please specify a valid tile, slide, sample, or HPC ID."

    return "\n\n---\n\n".join(output_parts)
import re

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




# ==========================================================
#  Each handler is clean and self-contained
# ==========================================================

def handle_tile(conn, match, intents):
    tile_name = match if isinstance(match, str) else match.group(1)
    row = conn.execute(
        text("""
            SELECT * FROM tile_registry
            WHERE tiles ILIKE :t OR h5_source_path ILIKE :t
            LIMIT 1
        """),
        {"t": f"%{tile_name}%"}
    ).fetchone()

    if not row:
        return f" No tile found matching '{tile_name}'."

    info = dict(row._mapping)
    out = [f"###  Tile `{tile_name}` Summary"]
    out += [f"- **{k}**: {v}" for k, v in info.items()]

    idx = info.get("image_index")
    hpc_id = info.get("hpc_id")

    if intents.get("malignant") and hpc_id:
        malignant_flag = not intents.get("negative") and conn.execute(
            text("SELECT malignant FROM hpc_dictionary WHERE hpc_id ILIKE :hpc_id"),
            {"hpc_id": f"%{hpc_id}%"}
        ).scalar()

        if malignant_flag:
            kb_row = conn.execute(
                text("SELECT * FROM hpc_malignant_details WHERE hpc_id ILIKE :hpc_id"),
                {"hpc_id": f"%{hpc_id}%"}
            ).fetchone()
            out.append(f" Tile belongs to **malignant epithelium** (HPC {hpc_id})")
        else:
            kb_row = conn.execute(
                text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id ILIKE :hpc_id"),
                {"hpc_id": f"%{hpc_id}%"}
            ).fetchone()
            out.append(f" Tile belongs to **non-malignant epithelium** (HPC {hpc_id})")

        if kb_row:
            out.append("**Epithelium details from KB:**")
            for k, v in dict(kb_row._mapping).items():
                out.append(f"- **{k}**: {v}")
            out.append("")
    # --- 2️⃣ Display image if index available ---
    try:
        h5_path = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"

        with h5py.File(h5_path, "r") as f:
            dataset = f["train_img"]  #  confirmed dataset name
            total = dataset.shape[0]
            st.write(f" Loaded H5 file with {total} total images.")

            if idx is None:
                st.warning(" No valid image index found for this tile.")
            elif idx < 0 or idx >= total:
                st.warning(f" Invalid index {idx}, skipping image.")
            else:
                tile = dataset[idx]
                tile = np.squeeze(tile)

                # Normalize
                tile_min, tile_max = tile.min(), tile.max()
                if tile_max > tile_min:
                    tile = (tile - tile_min) / (tile_max - tile_min) * 255
                tile = tile.astype(np.uint8)

                # Convert and display
                if tile.ndim == 2:
                    img = Image.fromarray(tile)
                elif tile.ndim == 3 and tile.shape[-1] in [1, 3]:
                    img = Image.fromarray(tile)
                else:
                    st.warning(f" Unexpected tile shape {tile.shape}")
                    return "\n".join(out)

                st.image(img, caption=f"{tile_name} (index {idx})", use_container_width=True)

    except Exception as e:
        st.error(f" Could not load image from H5 file: {e}")

    return "\n".join(out)
  

def handle_slide(conn, match, intents):
    # Normalize slide ID
    slide_id = match if isinstance(match, str) else match.group(0)
    slide_id = re.sub(r"^slide\s+", "", slide_id, flags=re.I).strip().upper()

    out = [f"### 🧫 Slide `{slide_id}` Summary\n"]

    # -------- 1) Malignancy question --------
    if intents.get("malignant"):
        malignant_ids = conn.execute(
            text("""
                SELECT DISTINCT hp.hpc_id
                FROM hpl_profile_proportion hp
                JOIN hpc_dictionary hd ON hp.hpc_id = hd.hpc_id
                WHERE hp.slides ILIKE :slide_id
                AND hd.malignant = TRUE
            """),
            {"slide_id": slide_id}
        ).scalars().all()

        if malignant_ids:
            out.append(f"⚠️ Malignant HPCs: {', '.join(map(str, malignant_ids))}")
        else:
            out.append(" No malignant HPCs detected.")

        st.session_state.active_slide = slide_id
        return "\n".join(out) + f"\n\nThe slide viewer is now updated for **{slide_id}**."

    # -------- 2) Slide summary --------
    summary = conn.execute(
        text("""
            SELECT * FROM hpl_profile_summary
            WHERE slides ILIKE :slide_id
            LIMIT 1
        """),
        {"slide_id": f"%{slide_id}%"}
    ).fetchone()

    if summary:
        out.append("**Slide Summary:**")
        for k, v in dict(summary._mapping).items():
            out.append(f"- **{k}**: {v}")
        out.append("")

    # -------- 3) Top HPCs --------
    proportions = conn.execute(
        text("""
            SELECT hpc_id, proportion, samples
            FROM hpl_profile_proportion
            WHERE slides = :slide_id
            ORDER BY proportion DESC
            LIMIT 5
        """),
        {"slide_id": slide_id}
    ).fetchall()

    if proportions:
        out.append("**Top HPCs:**")
        for row in proportions:
            d = dict(row._mapping)
            out.append(
                f"- **HPC {d['hpc_id']}** → proportion: {d['proportion']:.5f}, sample: {d['samples']}"
            )
        out.append("")

    

    # -------- 5) Update Slide Viewer --------
    st.session_state.active_slide = slide_id
    return "\n".join(out) + f"\n\nThe slide viewer is now updated for **{slide_id}**."

def handle_sample(conn, match, intents):
    sample_id = match if isinstance(match, str) else match.group(0)
    out = [f"### 🧬 Sample `{sample_id}` Summary"]

    summary = conn.execute(
        text("SELECT * FROM hpl_profile_summary WHERE samples ILIKE :s LIMIT 1"),
        {"s": f"%{sample_id}%"}
    ).fetchone()
    if summary:
        out += [f"- **{k}**: {v}" for k, v in dict(summary._mapping).items()]

    return "\n".join(out)


def handle_hpc(conn, ids, intents):
    insp = inspect(engine)
    out = []

    for hpc_id in ids:
        block = [f"## HPC {hpc_id} Summary\n"]

        # --- 1️ Dictionary Info ---
        base = conn.execute(
            text("SELECT * FROM hpc_dictionary WHERE hpc_id ILIKE :hpc_id"),
            {"hpc_id": hpc_id}
        ).fetchone()

        if not base:
            block.append(f"⚠️ No entry found for HPC {hpc_id}.\n")
            out.append("\n".join(block))
            continue

        d = dict(base._mapping)

        # --- 2️ If user asked about malignancy only ---
        if intents.get("malignant"):
            malignant_flag = bool(d.get("malignant"))

            if malignant_flag:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_malignant_details WHERE hpc_id ILIKE :hpc_id"),
                    {"hpc_id": hpc_id}
                ).fetchone()
                block.append(" **Malignant epithelium details:**")
            else:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id ILIKE :hpc_id"),
                    {"hpc_id": hpc_id}
                ).fetchone()
                block.append(" **Non-malignant epithelium details:**")

            if kb_row:
                for k, v in dict(kb_row._mapping).items():
                    block.append(f"- **{k}**: {v}")
                block.append("")

            # Stop here — do NOT fetch tiles or other tables
            out.append("\n".join(block))
            continue

        # --- 3️⃣ Otherwise (normal query, not malignancy question) ---
        block.append("**Dictionary Details:**")
        for k, v in d.items():
            block.append(f"- **{k}**: {v}")
        block.append("")

        # Fetch all other related data only for normal lookups
        tables = insp.get_table_names()
        for table in tables:
            if table in ["hpc_dictionary", "h_latent_vectors"]:
                continue

            cols = [c["name"] for c in insp.get_columns(table)]
            id_col = "hpc_id" if "hpc_id" in cols else "dominant_hpc" if "dominant_hpc" in cols else None
            if not id_col:
                continue

            rows = conn.execute(
                text(f'SELECT * FROM {table} WHERE TRIM({id_col}::text) ILIKE TRIM(:h) LIMIT 5'),
                {"h": f"%{str(hpc_id).strip()}%"}
            ).fetchall()

            if not rows:
                continue

            block.append(f"**{table.replace('_', ' ').title()}:**")
            for r in rows:
                for k, v in dict(r._mapping).items():
                    block.append(f"- **{k}**: {v}")
                block.append("")
                
        
        # --- 5️⃣ NEW: Display linked tile images ---
        tile_rows = conn.execute(
            text("SELECT tiles, image_index FROM tile_registry WHERE hpc_id ILIKE :hpc_id LIMIT 5"),
            {"hpc_id": hpc_id}
        ).fetchall()
        
        if tile_rows:
            block.append("")
            st.subheader(f" Example Tiles for HPC {hpc_id}")

            try:
                h5_path = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"
                with h5py.File(h5_path, "r") as f:
                    dataset = f["train_img"]
                    total = dataset.shape[0]
                    st.write(f"Loaded H5 file. Total images: {total}")

                    cols = st.columns(3)

                    for i, row in enumerate(tile_rows[:6]):
                        # Handle row formats
                        if hasattr(row, "_mapping"):
                            tile_name = row.tiles
                            idx = row.image_index
                        else:
                            try:
                                tile_name, idx = eval(row)
                            except Exception:
                                continue

                        st.write(f"Fetching tile {tile_name}, index {idx}")

                        if idx is None or idx >= total:
                            st.warning(f"Skipping invalid index {idx}")
                            continue

                        # Extract and normalize
                        tile = dataset[idx]
                        tile = np.squeeze(tile)  # remove singleton dimensions if any
                        tile = (tile - tile.min()) / (tile.max() - tile.min() + 1e-5) * 255
                        tile = tile.astype(np.uint8)

                        # Handle grayscale or RGB mismatch
                        if tile.ndim == 2:
                            img = Image.fromarray(tile)
                        elif tile.ndim == 3 and tile.shape[-1] == 3:
                            img = Image.fromarray(tile)
                        else:
                            st.warning(f" Unexpected tile shape {tile.shape}")
                            continue

                        cols[i % 3].image(img, caption=f"{tile_name} (index {idx})", use_container_width=True)

            except Exception as e:
                st.error(f"Error loading tiles: {e}")

        out.append("\n".join(block))
        return "\n\n".join(out)

def handle_analytics(conn, query, intent_type):
    # Detect whether the question is about malignant, non-malignant, or both
    polarity = classify_malignancy_polarity(query)

    if intent_type == "count_hpcs":
        result = conn.execute(text("""
            SELECT 
                SUM(CASE WHEN malignant THEN 1 ELSE 0 END) AS malignant_count,
                SUM(CASE WHEN NOT malignant THEN 1 ELSE 0 END) AS non_malignant_count,
                COUNT(*) AS total_hpcs
            FROM hpc_dictionary;
        """)).fetchone()

        total = result.total_hpcs or 0
        malignant = result.malignant_count or 0
        non_malignant = result.non_malignant_count or 0

        # Calculate percentages safely
        malignant_percent = round(100 * malignant / total, 2) if total else 0
        non_malignant_percent = round(100 * non_malignant / total, 2) if total else 0

        # Respond dynamically based on detected polarity
        if polarity == "malignant":
            return f"There are **{malignant} malignant HPCs** out of {total} total ({malignant_percent}%)."
        elif polarity == "non":
            return f"There are **{non_malignant} non-malignant HPCs** out of {total} total ({non_malignant_percent}%)."
        else:
            return (
                f"Total HPCs: **{total}**\n"
                f"- Malignant: {malignant} ({malignant_percent}%)\n"
                f"- Non-malignant: {non_malignant} ({non_malignant_percent}%)"
            )

    # -------------------------------------------------------------------
    elif intent_type == "slide_malignant_coverage":
        slide = re.search(r"(?:slide\s*)?([A-Z0-9\-_]+DX\d+)", query, re.I)
        if not slide:
            return "Please specify a valid slide ID."

        slide_id = slide.group(1).strip().upper()

        # Compute malignant vs total proportion coverage
        result = conn.execute(text("""
            SELECT 
                ROUND(
                    CAST(
                        100 * SUM(CASE WHEN hd.malignant THEN hp.proportion ELSE 0 END)
                        / NULLIF(SUM(hp.proportion), 0)
                        AS numeric
                    ), 
                    2
                ) AS malignant_percent
            FROM hpl_profile_proportion hp
            JOIN hpc_dictionary hd ON hp.hpc_id = hd.hpc_id
            WHERE hp.slides ILIKE :slide;
        """), {"slide": slide_id}).fetchone()

        if result and result.malignant_percent is not None:
            if polarity == "malignant" or polarity is None:
                return f"Malignant epithelium covers **{result.malignant_percent}%** of slide `{slide_id}`."
            else:
                non_malignant_percent = round(100 - result.malignant_percent, 2)
                return f"Non-malignant covers **{non_malignant_percent}%** of slide `{slide_id}`."
        else:
            return f"No data found for slide `{slide_id}`."

    # -------------------------------------------------------------------
    else:
        return "Sorry, I couldn’t identify what kind of analytics you want."

def handle_survival(conn, query):
    # Detect if user mentioned an HPC
    hpc_match = re.search(r"hpc[-_\s]*([0-9]+)", query, re.I)

    # Detect tile name if no HPC mentioned
    tile_match = re.search(r"(?:tile[_\-\s]*)?([A-Za-z0-9_\-]+\.jpe?g|[A-Za-z0-9_\-]+\.png|[A-Za-z0-9_\-]+\.tif)", query, re.I)

    # Case 1: Tile-based query (map tile → hpc_id)
    if tile_match and not hpc_match:
        tile_name = tile_match.group(1)
        hpc_id = conn.execute(
            text("SELECT hpc_id FROM tile_registry WHERE tiles ILIKE :tile LIMIT 1"),
            {"tile": f"%{tile_name}%"}
        ).scalar()

        if not hpc_id:
            return f"No HPC found for tile `{tile_name}`."
        
        # Recurse with found HPC ID
        query = f"HPC {hpc_id}"  # inject for next step
        hpc_match = re.search(r"hpc[-_\s]*([0-9]+)", query, re.I)

    # Case 2: Direct HPC query
    if hpc_match:
        hpc_id = hpc_match.group(1)
        result = conn.execute(
            text("SELECT * FROM hpc_survival_analysis WHERE hpc_id = :hpc_id"),
            {"hpc_id": hpc_id}
        ).fetchone()

        if result:
            d = dict(result._mapping)
            return (
                f"### Survival analysis for HPC {hpc_id}\n"
                f"- **Hazard Ratio (HR):** {d.get('expcoef', 0):.3f}\n"
                f"- **Coefficient (β):** {d.get('coef', 0):.3f}\n"
                f"- **Standard Error (SE):** {d.get('se', 0):.3f}\n"
                f"- **95% CI (HR):** ({d.get('expcoef_lower_95', 0):.3f}, {d.get('expcoef_upper_95', 0):.3f})\n"
                f"- **Z-score:** {d.get('z', 0):.3f}\n"
                f"- **p-value:** {d.get('p', 0):.4f}\n"
                f"- **–log₂(p):** {d.get('log2_p', -np.log2(d['p'])):.3f}\n\n"
                f"{' Significant association with survival (p < 0.05)' if d.get('p', 1) < 0.05 else ' Not statistically significant.'}"
            )
        else:
            return f"No survival data found for HPC {hpc_id}."

    # Case 3: No specific tile/HPC — show top significant ones
    top = conn.execute(
        text("""
            SELECT hpc_id, expcoef, p 
            FROM hpc_survival_analysis 
            WHERE p IS NOT NULL 
            ORDER BY p ASC 
            LIMIT 10
        """)
    ).fetchall()

    if top:
        lines = ["### 🧬 Top 10 HPCs associated with survival"]
        for row in top:
            lines.append(f"- **HPC {row.hpc_id}** → HR={row.expcoef:.3f}, p={row.p:.4f}")
        return "\n".join(lines)

    return "No survival analysis data available."

# --- Path to your .h5 file ---
h5_path = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"

def get_slide_context_from_query(prompt: str):
    """
    Option A:
    1) Use active_slide if it exists
    2) Else try to parse slide from the query
    3) Else return None
    """
    # 1) session state wins
    slide_id = st.session_state.get("active_slide", None)
    if slide_id:
        return str(slide_id).strip().upper()

    # 2) try detect in query
    detected = detect_entity_patterns(prompt)
    if detected and detected.get("slide"):
        return str(detected["slide"]).strip().upper()

    return None




def apply_adjacency_from_query(a: int, b: int, slide_id: str):
    st.session_state.active_slide = slide_id
    st.session_state.viewer_open = True

    # Force viewer mode
    st.session_state.highlight_mode = "Adjacency"

    # Clear anything that can override or confuse highlighting
    st.session_state.selected_hpc = None
    st.session_state.selected_tile = None

    st.session_state.adj_hpc_a = a
    st.session_state.adj_hpc_b = b

    df_slide = tile_coords[tile_coords["slides"] == slide_id]

    pair_edge_counts, tile_has_neighbor_pair = compute_adjacency_cached(df_slide)

    p = (a, b) if a < b else (b, a)
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

# --- Streamlit app ---



# --- Image uploader ---
uploaded_file = st.file_uploader("Upload an H&E tile or slide image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! 👋 Do you have questions on HPCs or something in your tiles? I’ve got you covered."}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input(
    "Ask something about HPCs...",
    key="main_chat_input"
)

if prompt:
    # 1) show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2) ensure slide context exists
    det = detect_entity_patterns(prompt)
    slide_from_prompt = det.get("slide") if det else None

    if slide_from_prompt:
        st.session_state.active_slide = str(slide_from_prompt).strip().upper()
    elif "active_slide" not in st.session_state or not st.session_state.active_slide:
        st.session_state.active_slide = list(WSI_MAP.keys())[0]

    slide_id = st.session_state.active_slide

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

if "active_slide" in st.session_state and st.session_state.active_slide:
    slide_id = st.session_state.active_slide

    # ✅ THIS stays visible across viewer open/close
    render_tile_preview(slide_id)

# --------------------------------------------------
# Slide Viewer Toggle (CORRECT)
# --------------------------------------------------

if "active_slide" in st.session_state and st.session_state.active_slide:

    slide_id = st.session_state.active_slide

    if "viewer_open" not in st.session_state:
        st.session_state.viewer_open = False

    if st.session_state.viewer_open:
        if st.button("Close Slide Viewer"):
            st.session_state.viewer_open = False
            st.rerun()
    else:
        if st.button(f"Open Slide Viewer for {slide_id}"):
            st.session_state.viewer_open = True
            st.rerun()

    if st.session_state.viewer_open:
        show_wsi(slide_id)