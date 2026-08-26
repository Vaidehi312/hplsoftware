import streamlit as st
import pandas as pd
from sqlalchemy import bindparam, create_engine, inspect, text
import re
import html
import numpy as np
import io
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
# from streamlit_image_zoom import image_zoom
import streamlit.components.v1 as components
import colorsys
import hashlib
from query_planner_v25 import build_query_plan_v25
from plan_query import save_query_plan_to_file
from llm_layer_v25 import explain_answer, llm_enabled, llm_planner_enabled, should_fetch_from_db
from ui_actions_v25 import ViewerContext, apply_plan_to_session
import os
import time
import json
import requests
from api_client import TileServerClient
from hpc_chat_handlers_v23 import detect_entity_patterns as chat_detect_entity_patterns
from hpc_chat_handlers_v23 import fetch_answer_from_db as chat_fetch_answer_from_db

detect_entity_patterns = chat_detect_entity_patterns

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HPC Chatbot", page_icon="💬", layout="wide")
st.title("🧠 HPC Chatbot")


# ---------------------------------------------------------------------------
# Tile server client (talks to FastAPI on HPCC)
# ---------------------------------------------------------------------------
TILE_SERVER_URL = os.getenv("TILE_SERVER_URL", "http://localhost:8000")
client = TileServerClient(TILE_SERVER_URL)


def render_wsi_upload_panel():
    """Upload a new WSI to the FastAPI backend.

    The backend queues tissue masking + tiling as a background job right
    after saving the file; this panel polls /processing-status for progress.
    """
    with st.sidebar.expander("Upload new WSI", expanded=False):
        st.caption("Step 1: Click 'Browse files' and select a WSI from your computer")

        uploaded_file = st.file_uploader(
            "Browse for WSI file",
            type=["svs", "ndpi", "tif", "tiff", "isyntax"],
            key="wsi_upload_file",
        )

        user_slide_id = st.text_input(
            "Slide ID optional",
            placeholder="Example: TCGA-XX-XXXX-DX1",
            key="wsi_upload_slide_id",
        )


        if st.button("Send selected file to server", key="wsi_upload_submit", use_container_width=True):
            if uploaded_file is None:
                st.error("Please choose a WSI file first.")
                return

            safe_slide_id = (user_slide_id or "").strip()

            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }
                data = {"slide_id": safe_slide_id}

                response = requests.post(
                    f"{TILE_SERVER_URL}/upload-slide",
                    files=files,
                    data=data,
                    timeout=600,
                )

                if response.status_code == 200:
                    payload = response.json()
                    st.success("Slide uploaded successfully.")
                    st.json(payload)

                    returned_slide_id = str(payload.get("slide_id") or "").strip().upper()
                    if returned_slide_id:
                        st.session_state.active_slide = returned_slide_id
                        st.session_state.viewer_open = False
                        st.session_state.processing_slide_id = returned_slide_id
                else:
                    st.error(f"Upload failed with status {response.status_code}.")
                    st.text(response.text)

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to tile server at {TILE_SERVER_URL}.")
            except requests.exceptions.Timeout:
                st.error("Upload timed out. Try a smaller test file first, or increase the backend/proxy timeout.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

        processing_slide_id = st.session_state.get("processing_slide_id")
        if processing_slide_id:
            st.divider()
            st.caption(f"Background processing: **{processing_slide_id}**")

            status_payload = None
            try:
                status_payload = client.get_processing_status(processing_slide_id)
            except Exception as e:
                st.warning(f"Could not fetch processing status: {e}")

            stage = (status_payload or {}).get("status", "unknown")
            if stage == "done":
                st.success("Tissue masking + tiling complete. Tiles are ready.")
                if st.button("Dismiss", key="wsi_processing_dismiss_done"):
                    del st.session_state["processing_slide_id"]
            elif stage == "error":
                st.error(f"Processing failed: {(status_payload or {}).get('error')}")
                if st.button("Dismiss", key="wsi_processing_dismiss_error"):
                    del st.session_state["processing_slide_id"]
            else:
                st.info(f"Status: **{stage}** (queued → masking → tiling → done)")
                st.button("Refresh status", key="wsi_processing_refresh")


with st.sidebar:
    st.subheader("v25 NL pipeline")
    if llm_enabled():
        st.success(f"Ollama: **on** (`{os.getenv('OLLAMA_MODEL', 'deepseek-r1:1.5b')}`)")
    else:
        st.warning("Ollama disabled (`LLM_ENABLED=0`)")
    if llm_planner_enabled():
        st.caption("Planner: **NLP + regex + LLM**")
    else:
        st.caption("Planner: **NLP + regex** (LLM planner off)")

    render_wsi_upload_panel()


# OpenSeadragon pyramid viewer helper
# ---------------------------------------------------------------------------
def render_openseadragon_viewer(slide_id: str, viewer_key: str = "main", height: int = 780, overlay_tiles=None):
    """Render an OpenSeadragon pyramid viewer for a slide with optional SVG tile overlays."""
    slide_id = (slide_id or "").strip().upper()
    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", f"{slide_id}_{viewer_key}")
    dzi_url = f"{TILE_SERVER_URL}/dzi/{slide_id}.dzi"

    overlay_tiles = overlay_tiles or []
    overlay_json = json.dumps(overlay_tiles)

    st.caption(f"OpenSeadragon DZI source: {dzi_url}")

    components.html(
        f"""
        <div id="osd_wrap_{safe_key}" style="position:relative;width:100%;height:{height}px;">
            <div id="osd_{safe_key}" style="
                width:100%;
                height:{height}px;
                background:#111827;
                border:1px solid #374151;
                border-radius:10px;
                overflow:hidden;
            "></div>

            <svg id="osd_overlay_{safe_key}" style="
                position:absolute;
                left:0;
                top:0;
                width:100%;
                height:100%;
                pointer-events:none;
                z-index:5;
            "></svg>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/openseadragon.min.js"></script>
        <script>
            const overlayTiles = {overlay_json};
            const viewer = OpenSeadragon({{
                id: "osd_{safe_key}",
                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
                tileSources: "{dzi_url}",
                showNavigator: true,
                navigatorPosition: "BOTTOM_RIGHT",
                animationTime: 0.4,
                blendTime: 0.1,
                constrainDuringPan: true,
                visibilityRatio: 0.8,
                minZoomImageRatio: 0.8,
                maxZoomPixelRatio: 3.0,
                showRotationControl: false,
                gestureSettingsMouse: {{
                    clickToZoom: true,
                    dblClickToZoom: true,
                    dragToPan: true,
                    scrollToZoom: true
                }}
            }});

            const overlaySvg = document.getElementById("osd_overlay_{safe_key}");

            function drawTileOverlay() {{
                if (!viewer || !viewer.viewport || !viewer.world || viewer.world.getItemCount() === 0) {{
                    return;
                }}

                const container = viewer.container.getBoundingClientRect();
                overlaySvg.setAttribute("viewBox", `0 0 ${{container.width}} ${{container.height}}`);
                overlaySvg.innerHTML = "";

                overlayTiles.forEach(function(t) {{
                    const rectVp = viewer.viewport.imageToViewportRectangle(
                        Number(t.x),
                        Number(t.y),
                        Number(t.w),
                        Number(t.h)
                    );

                    const p1 = viewer.viewport.pixelFromPoint(rectVp.getTopLeft(), true);
                    const p2 = viewer.viewport.pixelFromPoint(rectVp.getBottomRight(), true);

                    const x = Math.min(p1.x, p2.x);
                    const y = Math.min(p1.y, p2.y);
                    const w = Math.abs(p2.x - p1.x);
                    const h = Math.abs(p2.y - p1.y);

                    if (x + w < 0 || y + h < 0 || x > container.width || y > container.height) {{
                        return;
                    }}

                    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                    r.setAttribute("x", x);
                    r.setAttribute("y", y);
                    r.setAttribute("width", Math.max(w, 1));
                    r.setAttribute("height", Math.max(h, 1));
                    r.setAttribute("fill", t.fill || "none");
                    r.setAttribute("stroke", t.stroke || t.color || "yellow");
                    r.setAttribute("stroke-width", t.stroke_width || Math.max(1, Math.min(4, w / 35)));
                    r.setAttribute("opacity", t.opacity || "0.95");
                    overlaySvg.appendChild(r);
                }});
            }}

            viewer.addHandler("open", drawTileOverlay);
            viewer.addHandler("animation", drawTileOverlay);
            viewer.addHandler("animation-finish", drawTileOverlay);
            viewer.addHandler("resize", drawTileOverlay);
            viewer.addHandler("zoom", drawTileOverlay);
            viewer.addHandler("pan", drawTileOverlay);

            viewer.addHandler("open-failed", function(event) {{
                console.error("OpenSeadragon open failed", event);
                const el = document.getElementById("osd_{safe_key}");
                el.innerHTML = '<div style="color:#fecaca;padding:16px;font-family:Arial;">OpenSeadragon failed to open the DZI source. Try opening this in browser: {dzi_url}</div>';
            }});

            viewer.addHandler("tile-load-failed", function(event) {{
                console.error("OpenSeadragon tile load failed", event);
            }});
        </script>
        """,
        height=height + 20,
    )


# ---------------------------------------------------------------------------
# Fallback: fetch tile region directly from SVS if H5 tile endpoint fails
def fetch_tile_region_from_svs(slide_id: str, x_native, y_native, tile_size_native: int):
    """Fetch the selected tile directly from the SVS using native coordinates.

    This is a fallback for cases where the H5-backed `/tile_image/{slide_tile}`
    endpoint fails. It uses the existing tile server `/slide/{slide_id}/region`
    endpoint and should be sharper because it comes from the original WSI.
    """
    url = f"{TILE_SERVER_URL}/slide/{str(slide_id).strip().upper()}/region"
    params = {
        "x": int(float(x_native)),
        "y": int(float(y_native)),
        "w": int(tile_size_native),
        "h": int(tile_size_native),
        "level": 0,
        "quality": 95,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Local DB engine — still needed for chat query pipeline (Phase 1)
# ---------------------------------------------------------------------------
# DB_USER = "vpandya"
# DB_PASS = ""
# DB_HOST = "127.0.0.1"
# DB_PORT = "5433"
# DB_NAME = "hpl_kb"

# engine = create_engine(
#     f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
#     pool_pre_ping=True,
# )



# Local Streamlit connects through the SSH tunnel on the Mac.
# Keep DB_HOST/DB_PORT fixed here to avoid old shell variables overriding the tunnel.
DB_USER = os.getenv("HPL_DB_USER", "vpandya")
DB_PASS = os.getenv("HPL_DB_PASS", "")
DB_HOST = "127.0.0.1"
DB_PORT = "5433"
DB_NAME = os.getenv("HPL_DB_NAME", "hpl_kb")

# For local Streamlit + SSH tunnel, the app should connect to the local
# forwarded port, usually 127.0.0.1:5433.
# Do not silently prefer DATABASE_URL/HPL_DB_URL because old shell values can
# point to localhost:5432 and cause confusing connection refused errors.
if DB_PASS:
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

SAFE_DATABASE_URL = DATABASE_URL
if DB_PASS:
    SAFE_DATABASE_URL = DATABASE_URL.replace(f":{DB_PASS}@", ":****@")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
)
st.sidebar.caption(f"Streamlit DB URL: {SAFE_DATABASE_URL}")

@st.cache_data(show_spinner=False, ttl=300)
def load_hpc_titles() -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT
                        hpc_id,
                        hpc_title
                    FROM hpc_dictionary
                    ORDER BY hpc_id
                    """
                ),
                conn,
            )
    except Exception:
        return pd.DataFrame(columns=["hpc_id", "hpc_title"])

    if df.empty:
        return pd.DataFrame(columns=["hpc_id", "hpc_title"])

    df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")
    df["hpc_title"] = df["hpc_title"].fillna("").astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_hpc_title_map() -> dict[int, str]:
    df = load_hpc_titles()
    if df.empty:
        return {}
    out = {}
    for _, row in df.dropna(subset=["hpc_id"]).iterrows():
        hid = int(row["hpc_id"])
        title = str(row.get("hpc_title") or "").strip()
        if title:
            out[hid] = title
    return out

@st.cache_data(show_spinner=False, ttl=300)
def load_survival_coefficients(p_threshold=0.05):
    """
    Load Cox survival coefficients for HPCs.

    Returns:
        dict:
        {
            hpc_id: {
                "coef": float,
                "expcoef": float,
                "p": float,
                "ci_low": float,
                "ci_high": float
            }
        }
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT
                        hpc_id,
                        coef,
                        expcoef,
                        p,
                        expcoef_lower_95,
                        expcoef_upper_95
                    FROM hpc_survival_analysis
                    WHERE hpc_id IS NOT NULL
                      AND coef IS NOT NULL
                    """
                ),
                conn,
            )
    except Exception as e:
        st.warning(
            "Could not load survival coefficients. Check that the PostgreSQL database or SSH tunnel is running. "
            f"Current DATABASE_URL: {DATABASE_URL}. Error: {e}"
        )
        return {}

    if df.empty:
        return {}

    df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce")
    df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
    df["expcoef"] = pd.to_numeric(df["expcoef"], errors="coerce")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["expcoef_lower_95"] = pd.to_numeric(df["expcoef_lower_95"], errors="coerce")
    df["expcoef_upper_95"] = pd.to_numeric(df["expcoef_upper_95"], errors="coerce")

    df = df.dropna(subset=["hpc_id", "coef"])

    if p_threshold is not None:
        df = df[df["p"] < float(p_threshold)]

    survival_map = {}

    for _, row in df.iterrows():
        hid = int(row["hpc_id"])

        survival_map[hid] = {
            "coef": float(row["coef"]),
            "expcoef": float(row["expcoef"]) if not pd.isna(row["expcoef"]) else np.nan,
            "p": float(row["p"]) if not pd.isna(row["p"]) else np.nan,
            "ci_low": float(row["expcoef_lower_95"]) if not pd.isna(row["expcoef_lower_95"]) else np.nan,
            "ci_high": float(row["expcoef_upper_95"]) if not pd.isna(row["expcoef_upper_95"]) else np.nan,
        }

    return survival_map


def add_survival_risk_score(tile_df, survival_map):
    """
    Compute smooth survival risk score per tile.

    Uses only:
    1. HPCs present in hpc_survival_analysis
    2. HPC probability columns present in this WSI
    3. Continuous p_hpc_* probabilities, not class labels

    Formula:
        tile_risk_score = sum(p_hpc_i * coef_i)
    """
    df = tile_df.copy()

    prob_cols = [c for c in df.columns if str(c).startswith("p_hpc_")]

    available_hpcs_in_wsi = set()
    for col in prob_cols:
        try:
            hid = int(str(col).replace("p_hpc_", ""))
            available_hpcs_in_wsi.add(hid)
        except Exception:
            continue

    survival_hpcs = set(survival_map.keys())

    usable_hpcs = sorted(survival_hpcs & available_hpcs_in_wsi)

    risk = np.zeros(len(df), dtype=float)

    contribution_cols = []

    for hid in usable_hpcs:
        col = f"p_hpc_{hid}"
        coef = survival_map[hid]["coef"]

        probs = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        contribution_col = f"risk_contrib_hpc_{hid}"
        df[contribution_col] = probs * coef
        contribution_cols.append(contribution_col)

        risk += probs * coef

    df["survival_risk_score"] = risk

    if len(risk) == 0:
        max_abs = 0
    else:
        max_abs = float(np.nanmax(np.abs(risk)))

    if max_abs == 0 or np.isnan(max_abs):
        df["survival_risk_norm"] = 0.0
    else:
        df["survival_risk_norm"] = df["survival_risk_score"] / max_abs

    df["survival_risk_abs"] = np.abs(df["survival_risk_norm"])

    return df, usable_hpcs


def risk_to_rgba(score_norm, alpha_min=20, alpha_max=170):
    """
    Convert normalized survival score into overlay colour.

    Positive score:
        red, poor survival associated

    Negative score:
        blue, protective survival associated

    Near zero:
        almost transparent
    """
    try:
        score_norm = float(score_norm)
    except Exception:
        return (255, 255, 255, 0)

    score_norm = max(-1.0, min(1.0, score_norm))

    intensity = abs(score_norm)

    if intensity < 0.02:
        return (255, 255, 255, 0)

    alpha = int(alpha_min + (alpha_max - alpha_min) * intensity)

    if score_norm > 0:
        return (255, 0, 0, alpha)

    if score_norm < 0:
        return (0, 80, 255, alpha)

    return (255, 255, 255, 0)



def hpc_label(hpc_id, max_title_chars: int = 80) -> str:
    if hpc_id is None or pd.isna(hpc_id):
        return "HPC unknown"

    hid = int(hpc_id)
    title = load_hpc_title_map().get(hid, "")

    if not title:
        return f"HPC {hid}"

    if len(title) > max_title_chars:
        title = title[: max_title_chars - 1] + "…"

    return f"HPC {hid}: {title}"



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
# HPC → slides from KB only (same tables as handle_slide: proportion + summary)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def slides_for_hpc_from_kb(hpc_id: int):
    hid = int(hpc_id)
    try:
        with engine.connect() as conn:
            base = pd.read_sql(
                text(
                    """
                    SELECT UPPER(TRIM(hp.slides)) AS slide_id,
                           hp.proportion,
                           hp.samples
                    FROM hpl_profile_proportion hp
                    WHERE hp.hpc_id = :hid
                      AND hp.slides IS NOT NULL
                    ORDER BY hp.proportion DESC NULLS LAST
                    """
                ),
                conn,
                params={"hid": hid},
            )

        if base is None or base.empty:
            return {"ok": True, "df": pd.DataFrame(columns=["slide_id", "proportion", "samples", "kb_summary"]), "error": None}

        base["slide_id"] = base["slide_id"].astype(str).str.strip().str.upper()
        base = base.drop_duplicates(subset=["slide_id"], keep="first")
        slides = sorted(base["slide_id"].unique().tolist())

        previews = {}
        with engine.connect() as conn:
            ss = text(
                """
                SELECT * FROM hpl_profile_summary
                WHERE UPPER(TRIM(slides)) IN :slides
                """
            ).bindparams(bindparam("slides", expanding=True))
            s_df = pd.read_sql(ss, conn, params={"slides": slides})

        if s_df is not None and not s_df.empty and "slides" in s_df.columns:
            s_df = s_df.copy()
            s_df["_sk"] = s_df["slides"].astype(str).str.strip().str.upper()
            for _, srow in s_df.iterrows():
                d = srow.to_dict()
                sk = str(d.get("_sk") or d.get("slides", "")).strip().upper()
                d.pop("_sk", None)
                d.pop("slides", None)
                if sk:
                    previews[sk] = _kb_summary_preview(d)

        base["kb_summary"] = base["slide_id"].map(lambda x: previews.get(str(x).strip().upper(), ""))
        return {"ok": True, "df": base.reset_index(drop=True), "error": None}

    except Exception as e:
        return {"ok": False, "df": None, "error": str(e)}


@st.cache_data(show_spinner=False, ttl=300)
def load_valid_hpc_ids() -> set[int]:
    df = load_hpc_titles()
    if df.empty:
        return set()
    return {int(x) for x in df["hpc_id"].dropna().astype(int).tolist()}


_VIEWER_CTX = None  # reserved


def _kb_summary_preview(row_dict: dict, max_chars: int = 520) -> str:
    """Turn a summary row into a compact line (v21-style KB fields, truncated for the table)."""
    parts = []
    n = 0
    for k, v in row_dict.items():
        lk = str(k).lower()
        if lk == "slides" or v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        chunk = f"{k}={v}"
        if len(chunk) > 120:
            chunk = chunk[:117] + "…"
        if n + len(chunk) > max_chars:
            parts.append("…")
            break
        parts.append(chunk)
        n += len(chunk) + 2
    return " · ".join(parts) if parts else ""


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


def get_viewer_ctx() -> ViewerContext:
    return ViewerContext(
        load_tile_coords_for_slide=load_tile_coords_for_slide,
        load_adjacency_for_slide=load_adjacency_for_slide,
        slides_for_hpc_from_kb=slides_for_hpc_from_kb,
        detect_entity_patterns=detect_entity_patterns,
        detect_adjacency_intent=detect_adjacency_intent,
        detect_single_hpc_adjacency_intent=detect_single_hpc_adjacency_intent,
        parse_hpc_id=parse_hpc_id,
        parse_inflammation=parse_inflammation,
        parse_necrosis=parse_necrosis,
        parse_highlight_mode=parse_highlight_mode,
    )


# ---------------------------------------------------------------------------
# apply_chat_query_to_wsi_state — legacy; prefer apply_plan_to_session (v25)
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

    hpc_title = str(info.get("hpc_title") or load_hpc_title_map().get(h, "") or "").strip()
    title_text = hpc_title if hpc_title else "No title available"

    st.markdown(
        f"""
        <div style="
            background:#111827;
            border:1px solid #374151;
            border-left:6px solid {rgb_to_css(color_for_hpc(h))};
            border-radius:10px;
            padding:12px 14px;
            margin-top:8px;
            margin-bottom:12px;
            color:#f3f4f6;
            line-height:1.45;
        ">
            <div style="font-size:15px;font-weight:800;margin-bottom:5px;">HPC {h}</div>
            <div style="font-size:16px;font-weight:700;color:#ffffff;">{html.escape(title_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_fields = [
        ("malignant", "Malignant"),
        ("inflammation", "Inflammation"),
        ("necrosis", "Necrosis"),
        ("cluster_homogeneity", "Cluster homogeneity"),
    ]

    rows = []
    for key, label in summary_fields:
        value = info.get(key)
        if value is not None:
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            rows.append({"Field": label, "Value": str(value)})

    mal = info.get("malignant_details") or {}
    non = info.get("non_malignant_details") or {}
    detail_source = mal if mal else non

    for key, value in detail_source.items():
        if key in ("id", "hpc_id"):
            continue
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        rows.append({"Field": str(key).replace("_", " ").title(), "Value": str(value)})

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No additional phenotype details found.")


# ---------------------------------------------------------------------------
# Tile info panel
# ---------------------------------------------------------------------------

def render_tile_info(tile_row, heat_hpc=None):
    st.subheader("Tile info")
    info = {
            "slide_tile": tile_row.get("slide_tile"),
            "tiles": tile_row.get("tiles"),
            "hpc_id": tile_row.get("hpc_id"),
            "hpc_title": tile_row.get("hpc_title"),
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

def show_wsi(slide_id, ui_suffix=""):
    """WSI viewer. ``ui_suffix`` disambiguates Streamlit keys when two viewers are open."""

    def _k(base: str) -> str:
        return f"{base}__{ui_suffix}" if ui_suffix else base

    slide_id = (slide_id or "").strip().upper()

    coords_df = load_tile_coords_for_slide(slide_id)
    if coords_df is None or coords_df.empty:
        st.warning(f"No tile coordinates found for slide {slide_id}")
        return

    for key, default in [
        (_k("selected_tile"), None),
        (_k("query_malignant"), None),
        ("adj_hpc_a", None),
        ("adj_hpc_b", None),
        ("adj_tile_sets", None),
        ("query_inflammation", None),
        ("query_necrosis", None),
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
    if _k("highlight_mode") not in st.session_state:
        st.session_state[_k("highlight_mode")] = "HPC clusters"

    viewer_mode = st.radio(
        "Viewer mode",
        ["Pyramid zoom", "Click tile inspector"],
        horizontal=True,
        key=_k("viewer_mode"),
    )

    show_grid = st.checkbox(
        "Show tile grid overlays",
        value=True,
        key=_k("grid-toggle"),
    )

    st.radio(
        "Highlight mode",
        options=["HPC clusters", "Inflammation", "Necrosis", "Malignant", "Adjacency", "Heatmap", "Survival Risk Heatmap"],
        key=_k("highlight_mode"),
        horizontal=True,
    )

    # Heatmap controls
    if st.session_state.get(_k("highlight_mode")) == "Heatmap":
        hpc_prob_cols = [c for c in df.columns if str(c).startswith("p_hpc_")]
        available_prob_hpcs = set()
        for col in hpc_prob_cols:
            try:
                available_prob_hpcs.add(int(str(col).replace("p_hpc_", "")))
            except Exception:
                continue

        if "hpc_id" in df.columns:
            present_hpcs = set(
                pd.to_numeric(df["hpc_id"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
        else:
            present_hpcs = set()

        hpc_ids_heat = sorted(present_hpcs & available_prob_hpcs)
        st.write("DEBUG unique HPCs on slide:", len(present_hpcs))
        st.write(sorted(list(present_hpcs))[:100])
        st.write("DEBUG heatmap dropdown HPCs:", len(hpc_ids_heat))
        

        if not hpc_ids_heat:
            st.warning("No heatmap cluster options found for this WSI. This slide may not have matching hpc_id values and p_hpc_* probability columns.")
        else:
            current_heat_hpc = st.session_state.get(_k("heat_hpc"))
            if current_heat_hpc not in hpc_ids_heat:
                st.session_state[_k("heat_hpc")] = hpc_ids_heat[0]

            title_map = load_hpc_title_map()
            heat_options = {
                f"HPC {hid}" + (f": {title_map.get(hid)}" if title_map.get(hid) else ""): hid
                for hid in hpc_ids_heat
            }

            selected_heat_label = st.selectbox(
                "Heatmap HPC clusters present in this WSI",
                options=list(heat_options.keys()),
                index=list(heat_options.values()).index(st.session_state[_k("heat_hpc")]),
                key=_k("heat_hpc_label"),
            )
            st.session_state[_k("heat_hpc")] = heat_options[selected_heat_label]
            st.caption(f"Showing heatmap options for {len(hpc_ids_heat)} HPC clusters present on this WSI.")
            st.slider("Heat intensity", 0.1, 1.0, 0.6, 0.05, key=_k("heat_alpha"))

    df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")

    used_survival_hpcs = []

    if st.session_state.get(_k("highlight_mode")) == "Survival Risk Heatmap":
        survival_map = load_survival_coefficients(p_threshold=0.05)

        df, used_survival_hpcs = add_survival_risk_score(
            df,
            survival_map,
        )

        st.session_state[_k("selected_hpc")] = None
        selected_hpc = None

        if not used_survival_hpcs:
            st.warning(
                "No survival-linked HPC probability columns found for this slide. "
                "This slide may not contain HPCs that also have survival data."
            )
        else:
            st.caption(
                f"Survival risk heatmap uses {len(used_survival_hpcs)} HPCs that are both present in this WSI and available in survival analysis."
            )

    if st.session_state.get(_k("highlight_mode")) != "Survival Risk Heatmap":
        st.session_state[_k("survival_risk_filter")] = None

    hpc_titles_df = load_hpc_titles()

    if not hpc_titles_df.empty and "hpc_id" in df.columns:
        df = df.merge(hpc_titles_df, on="hpc_id", how="left")
    else:
        df["hpc_title"] = ""
        
    # ------------------------------------------------------------------
    # 4. Draw overlay
    # ------------------------------------------------------------------
    highlight_mode = st.session_state.get(_k("highlight_mode"), "HPC clusters")
    selected_hpc = st.session_state.get(_k("selected_hpc"))

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
    if _k("selected_hpc") not in st.session_state:
        st.session_state[_k("selected_hpc")] = None
    hpc_list = sorted(df["hpc_id"].dropna().astype(int).unique().tolist())

    # Adjacency controls
    if highlight_mode == "Adjacency":
        pair_edge_counts, tile_has_neighbor_pair = load_adjacency_for_slide(slide_id)
        with st.expander("Adjacency and cooccurrence controls", expanded=False):
            top_pairs = sorted(pair_edge_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
            if top_pairs:
                options = [f"HPC {a} ↔ HPC {b} ({cnt} edges)" for (a, b), cnt in top_pairs]
                selected_pair = st.selectbox(
                    "Pick a top cooccurring pair", options, key=_k("adj_top_pair_select")
                )
                idx = options.index(selected_pair)
                (a_sel, b_sel), _ = top_pairs[idx]
                if st.button("Highlight selected top pair", key=_k("adj_top_pair_btn")):
                    p = (a_sel, b_sel) if a_sel < b_sel else (b_sel, a_sel)
                    st.session_state.adj_hpc_a = a_sel
                    st.session_state.adj_hpc_b = b_sel
                    st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(
                        p, {"a_touch": set(), "b_touch": set()}
                    )
                    st.rerun()
            else:
                st.info("No adjacent cross HPC pairs found on this slide.")


    selected_hpc = st.session_state.get(_k("selected_hpc"))
    if selected_hpc is not None:
                selected_hpc_title = load_hpc_title_map().get(int(selected_hpc), "")

                if selected_hpc_title:
                    st.markdown(
                        f"""
                        <div style="
                            background:#111827;
                            border:1px solid #374151;
                            border-left:6px solid {rgb_to_css(color_for_hpc(selected_hpc))};
                            border-radius:10px;
                            padding:12px 14px;
                            margin-top:10px;
                            margin-bottom:10px;
                            color:#f3f4f6;
                            font-size:15px;
                            line-height:1.5;
                        ">
                            <span style="font-weight:700;">HPC {selected_hpc}</span><br>
                            <span style="opacity:0.92;">{html.escape(selected_hpc_title)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("🧠 HPC Biological Interpretation", expanded=True):
                    render_hpc_annotation(selected_hpc)

    # Filters
    infl_f = st.session_state.get("query_inflammation")
    nec_f = st.session_state.get("query_necrosis")
    mal_f = st.session_state.get(_k("query_malignant"))
    filtered_df = df.copy()
    
    if st.session_state.get(_k("selected_hpc")) is not None:
        filtered_df = filtered_df[filtered_df["hpc_id"] == st.session_state[_k("selected_hpc")]]
    if infl_f is not None:
        filtered_df = filtered_df[filtered_df["inflammation"].astype(str).str.lower().str.strip() == infl_f]
    if nec_f is not None:
        filtered_df = filtered_df[filtered_df["necrosis"].astype(str).str.lower().str.strip() == nec_f]
    if mal_f is not None and "malignant" in filtered_df.columns:
        def _mal_filter_value(v):
            if isinstance(v, (int, np.integer, bool, np.bool_)):
                return "malignant" if bool(v) else "non-malignant"
            s = str(v).strip().lower()
            if s in ("true", "t", "1", "yes", "y", "malignant"):
                return "malignant"
            if s in ("false", "f", "0", "no", "n", "non-malignant", "non malignant"):
                return "non-malignant"
            return "missing"

        filtered_df = filtered_df[filtered_df["malignant"].map(_mal_filter_value) == mal_f]
    st.caption(f"Matched tiles: {len(filtered_df)}")

    # ------------------------------------------------------------------
    # 4a. Heatmap overlay (vectorized, same as v21)
    # ------------------------------------------------------------------
    if highlight_mode == "Heatmap":
        h = int(st.session_state.get(_k("heat_hpc"), 0))
        col_name = f"p_hpc_{h}"
        if col_name in df.columns:
            _draw_heatmap(
                heat_draw,
                df,
                col_name,
                downsample,
                float(st.session_state.get(_k("heat_alpha"), 0.6)),
                tile_size_native,
            )


    if highlight_mode == "Survival Risk Heatmap":
            _draw_survival_risk_heatmap(
                heat_draw,
                df,
                downsample,
                tile_size_native,
                alpha=140,
                risk_filter=st.session_state.get(_k("survival_risk_filter")),
            )
    # ------------------------------------------------------------------
    # 4b. Grid outlines (vectorized where possible)
    # ------------------------------------------------------------------
    if show_grid and highlight_mode not in ("Heatmap", "Survival Risk Heatmap"):
        _draw_grid(
            draw,
            grid_df if selected_hpc is not None and highlight_mode not in ("Adjacency", "Heatmap", "Survival Risk Heatmap") else df,
            downsample,
            highlight_mode,
            infl_f,
            nec_f,
            mal_f,
            tile_size_native,
        )


    # Selected tile highlight
    sel = st.session_state.get(_k("selected_tile"))
    if sel and "x_native" in sel and "y_native" in sel:
        sx = int(sel["x_native"] / downsample)
        sy = int(sel["y_native"] / downsample)
        ts = int(tile_size_native / downsample)
        draw.rectangle([sx - 3, sy - 3, sx + ts + 3, sy + ts + 3], outline="yellow", width=15)
        draw.rectangle([sx, sy, sx + ts, sy + ts], outline="lime", width=15)

    if highlight_mode in ("Heatmap", "Survival Risk Heatmap"):
        overlay = Image.alpha_composite(overlay, heat_layer)

    overlay_np = np.asarray(overlay.convert("RGB"), dtype=np.uint8)

    # ------------------------------------------------------------------
    # 5. Viewer + sticky legend (same viewport column)
    # ------------------------------------------------------------------
    # st.subheader(f"Slide Viewer — {slide_id}")
    col_img, col_leg = st.columns([5, 1])
    with col_leg:
        _hh = int(st.session_state.get(_k("heat_hpc"), 0)) if highlight_mode == "Heatmap" else None
        if highlight_mode == "Survival Risk Heatmap" and used_survival_hpcs:
            st.markdown("**Legend**")
            st.caption("Red = higher risk-associated signal. Blue = protective-associated signal. Transparent = near neutral.")
            st.markdown(
                """
                <div style="background:#f1f5f9;color:#0f172a;border:1px solid #64748b;border-radius:10px;padding:10px;font-size:12px;line-height:1.35;">
                    <div style="display:flex;align-items:center;gap:6px;margin:4px 0;"><span style="width:14px;height:14px;background:rgba(255,0,0,0.75);border:1px solid #0f172a;"></span><span>Risk-associated contribution</span></div>
                    <div style="display:flex;align-items:center;gap:6px;margin:4px 0;"><span style="width:14px;height:14px;background:rgba(0,80,255,0.75);border:1px solid #0f172a;"></span><span>Protective-associated contribution</span></div>
                    <div style="display:flex;align-items:center;gap:6px;margin:4px 0;"><span style="width:14px;height:14px;background:rgba(255,255,255,0.15);border:1px solid #0f172a;"></span><span>Near neutral</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if highlight_mode == "Survival Risk Heatmap":
            st.markdown("**Survival Risk Legend**")
            st.caption("Red = risk-associated. Blue = protective. Transparent = neutral.")

            if used_survival_hpcs:
                st.caption(f"Using survival HPCs: {used_survival_hpcs}")
            else:
                st.warning("No survival-linked HPCs found.")

        elif highlight_mode == "HPC clusters":
            st.markdown("**Legend**")
            st.caption("Click an HPC to isolate it.")

            if st.button("Show all", key=_k("legend_show_all_hpcs"), use_container_width=True):
                st.session_state[_k("selected_hpc")] = None
                st.rerun()

            n_tiles = max(len(df), 1)
            hpc_counts = pd.to_numeric(df["hpc_id"], errors="coerce").dropna().astype(int).value_counts()

            for hid, count in hpc_counts.head(14).items():
                hid = int(hid)
                pct = 100.0 * float(count) / n_tiles
                css_color = rgb_to_css(color_for_hpc(hid))

                st.markdown(
                    f"""
                    <div style="
                        width:100%;
                        height:4px;
                        background:{css_color};
                        border-radius:4px 4px 0 0;
                        margin-top:6px;
                        margin-bottom:-6px;
                    "></div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"HPC {hid} · {pct:.1f}%",
                    key=_k(f"legend_hpc_btn_{hid}"),
                    use_container_width=True,
                ):
                    st.session_state[_k("selected_hpc")] = hid
                    st.rerun()

        elif highlight_mode == "Inflammation":
            st.markdown("**Legend**")
            st.caption("Click a category to isolate it.")

            if st.button("Show all inflammation", key=_k("legend_all_inflammation"), use_container_width=True):
                st.session_state.query_inflammation = None
                st.rerun()

            n_tiles = max(len(df), 1)
            infl_counts = (
                df["inflammation"]
                .fillna("Missing")
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"": "missing"})
                .value_counts()
            )

            for label, count in infl_counts.items():
                pct = 100.0 * float(count) / n_tiles
                css_color = rgb_to_css(color_for_inflammation(label))
                pretty = str(label).title()

                st.markdown(
                    f'<div style="width:100%;height:4px;background:{css_color};border-radius:4px 4px 0 0;margin-top:6px;margin-bottom:-6px;"></div>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"{pretty} · {pct:.1f}%",
                    key=_k(f"legend_infl_btn_{label}"),
                    use_container_width=True,
                ):
                    st.session_state.query_inflammation = label
                    st.rerun()

        elif highlight_mode == "Necrosis":
            st.markdown("**Legend**")
            st.caption("Click a category to isolate it.")

            if st.button("Show all necrosis", key=_k("legend_all_necrosis"), use_container_width=True):
                st.session_state.query_necrosis = None
                st.rerun()

            n_tiles = max(len(df), 1)
            nec_counts = (
                df["necrosis"]
                .fillna("Missing")
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"": "missing"})
                .value_counts()
            )

            for label, count in nec_counts.items():
                pct = 100.0 * float(count) / n_tiles
                css_color = rgb_to_css(color_for_necrosis(label))
                pretty = str(label).title()

                st.markdown(
                    f'<div style="width:100%;height:4px;background:{css_color};border-radius:4px 4px 0 0;margin-top:6px;margin-bottom:-6px;"></div>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"{pretty} · {pct:.1f}%",
                    key=_k(f"legend_nec_btn_{label}"),
                    use_container_width=True,
                ):
                    st.session_state.query_necrosis = label
                    st.rerun()

        elif highlight_mode == "Malignant":
            st.markdown("**Legend**")
            st.caption("Click a category to isolate it.")

            if st.button("Show all malignant", key=_k("legend_all_malignant"), use_container_width=True):
                st.session_state[_k("query_malignant")] = None
                st.rerun()

            n_tiles = max(len(df), 1)
            mal_counts = (
                df["malignant"]
                .fillna("missing")
                .astype(str)
                .str.strip()
                .str.lower()
                .value_counts()
            )

            for label, count in mal_counts.items():
                pct = 100.0 * float(count) / n_tiles
                s = str(label).lower()

                if s in ("true", "t", "1", "yes", "y", "malignant"):
                    sample_val = True
                    label_key = "malignant"
                elif s in ("false", "f", "0", "no", "n", "non-malignant", "non malignant"):
                    sample_val = False
                    label_key = "non-malignant"
                else:
                    sample_val = None
                    label_key = "missing"

                css_color = rgb_to_css(color_for_malignant(sample_val))
                pretty = label_key.replace("-", " ").title()

                st.markdown(
                    f'<div style="width:100%;height:4px;background:{css_color};border-radius:4px 4px 0 0;margin-top:6px;margin-bottom:-6px;"></div>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"{pretty} · {pct:.1f}%",
                    key=_k(f"legend_mal_btn_{label_key}"),
                    use_container_width=True,
                ):
                    st.session_state[_k("query_malignant")] = label_key
                    st.rerun()

        else:
            st.markdown(build_legend_panel_html(highlight_mode, df, _hh), unsafe_allow_html=True)

    with col_img:
        if st.session_state.get(_k("viewer_mode"), "Pyramid zoom") == "Pyramid zoom":
            # st.error(f"DEBUG highlight_mode = {highlight_mode}")
            st.caption("OpenSeadragon pyramid viewer with optional tile-grid overlay.")

            if show_grid:
                if highlight_mode == "Survival Risk Heatmap":
                    osd_overlay_records = build_survival_osd_overlay_records(
                        df=df,
                        tile_size_native=tile_size_native,
                    )
                else:
                    osd_overlay_records = build_osd_overlay_records(
                        df=grid_df if selected_hpc is not None and highlight_mode not in ("Adjacency", "Heatmap", "Survival Risk Heatmap") else df,
                        highlight_mode=highlight_mode,
                        infl_f=infl_f,
                        nec_f=nec_f,
                        mal_f=mal_f,
                        tile_size_native=tile_size_native,
                        heat_hpc=st.session_state.get(_k("heat_hpc")),
                        heat_alpha=st.session_state.get(_k("heat_alpha"), 0.6),
                    )
            else:
                osd_overlay_records = []

            if highlight_mode == "Survival Risk Heatmap":
                st.caption(f"Survival OSD overlay records: {len(osd_overlay_records)}")

            render_openseadragon_viewer(
                slide_id,
                viewer_key=f"{_k('osd')}_{highlight_mode}_{len(osd_overlay_records)}_survival_v2_{st.session_state.get(_k('heat_hpc'), 'none')}_{st.session_state.get(_k('heat_alpha'), 0.6)}",
                height=780,
                overlay_tiles=osd_overlay_records,
            )
            return

        st.write("Click a tile to view it.")
        click = streamlit_image_coordinates(overlay_np, key=_k("wsi-click-coords"))
        if not click:
            st.info("Click anywhere on the slide to select a tile.")
            return

        cx, cy = click["x"], click["y"]
        native_x = cx * downsample
        native_y = cy * downsample

        tol = tile_size_native * 0.1
        mask = (
            (df["x_native"].astype(float) - tol <= native_x)
            & (native_x <= df["x_native"].astype(float) + tile_size_native + tol)
            & (df["y_native"].astype(float) - tol <= native_y)
            & (native_y <= df["y_native"].astype(float) + tile_size_native + tol)
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
        prev = st.session_state.get(_k("selected_tile"))
        same_tile = isinstance(prev, dict) and prev.get("slide_tile") == new_selected["slide_tile"] and float(
            prev.get("x_native", -1)
        ) == new_selected["x_native"] and float(prev.get("y_native", -1)) == new_selected["y_native"]
        if not same_tile:
            st.session_state[_k("selected_tile")] = new_selected
            st.rerun()

        # ------------------------------------------------------------------
        # 6. Show selected tile image — fetched from tile server
        # ------------------------------------------------------------------
        slide_tile_key = str(tile_row.get("slide_tile", ""))
        tile_name = tile_row.get("tiles", slide_tile_key)
        tile_hpc = tile_row.get("hpc_id")
        st.success(f"Tile selected: {tile_name}")

        if tile_hpc is not None and not pd.isna(tile_hpc):
            st.info(hpc_label(tile_hpc, max_title_chars=160))

        try:
            tile_img = fetch_tile_region_from_svs(
                slide_id=slide_id,
                x_native=tile_row.get("x_native"),
                y_native=tile_row.get("y_native"),
                tile_size_native=tile_size_native,
            )
            display_width = min(max(int(tile_img.width * 2), tile_img.width), 700)
            st.image(
                tile_img,
                caption=f"{tile_name} · native SVS region · {tile_img.width}×{tile_img.height}px",
                width=display_width,
            )
        except Exception as e:
            st.warning(f"Could not load native SVS tile region: {e}")

        heat_hpc = (
            st.session_state.get(_k("heat_hpc"))
            if st.session_state.get(_k("highlight_mode")) == "Heatmap"
            else None
        )
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
        

def _draw_survival_risk_heatmap(heat_draw, df, downsample, tile_size_native, alpha=140, risk_filter=None):
    if "survival_risk_norm" not in df.columns:
        return

    ts = int(tile_size_native / downsample)

    xs = (df["x_native"].astype(float) / downsample).astype(int).to_numpy()
    ys = (df["y_native"].astype(float) / downsample).astype(int).to_numpy()
    scores = pd.to_numeric(df["survival_risk_norm"], errors="coerce").fillna(0).to_numpy()

    for x, y, score in zip(xs, ys, scores):
        score = float(score)

        if risk_filter == "risky" and score <= 0.02:
            continue
        if risk_filter == "protective" and score >= -0.02:
            continue
        if risk_filter == "neutral" and not (-0.02 <= score <= 0.02):
            continue

        fill = risk_to_rgba(score, alpha_max=alpha)
        heat_draw.rectangle(
            [int(x), int(y), int(x) + ts, int(y) + ts],
            fill=fill,
            outline=(255, 255, 255, 35),
        )


def _draw_grid(draw, grid_df, downsample, highlight_mode, infl_f, nec_f, mal_f, tile_size_native):
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
        if mal_f is not None:
            val = r.get("malignant")

            if isinstance(val, (int, np.integer, bool, np.bool_)):
                current = "malignant" if bool(val) else "non-malignant"
            else:
                s = str(val).strip().lower()

                if s in ("true", "t", "1", "yes", "y", "malignant"):
                    current = "malignant"
                elif s in ("false", "f", "0", "no", "n", "non-malignant", "non malignant"):
                    current = "non-malignant"
                else:
                    current = "missing"

            if current != mal_f:
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




def build_survival_osd_overlay_records(df, tile_size_native, max_tiles=6000):
    records = []

    if df is None or df.empty or "survival_risk_norm" not in df.columns:
        return records

    use_df = df.head(max_tiles).copy()
    scores = pd.to_numeric(use_df["survival_risk_norm"], errors="coerce").fillna(0.0)

    for idx, r in use_df.iterrows():
        score = float(scores.loc[idx])
        score = max(-1.0, min(1.0, score))
        intensity = abs(score)

        if intensity < 0.02:
            continue

        alpha = 0.25 + 0.65 * intensity

        if score > 0:
            fill = f"rgba(255, 0, 0, {alpha:.3f})"
            stroke = "rgba(180, 0, 0, 0.95)"
        else:
            fill = f"rgba(0, 80, 255, {alpha:.3f})"
            stroke = "rgba(0, 45, 180, 0.95)"

        records.append({
            "x": float(r["x_native"]),
            "y": float(r["y_native"]),
            "w": int(tile_size_native),
            "h": int(tile_size_native),
            "fill": fill,
            "stroke": stroke,
            "stroke_width": 2,
            "opacity": "1.0",
            "slide_tile": str(r.get("slide_tile", "")),
            "survival_risk_norm": score,
        })

    return records


def build_osd_overlay_records(df, highlight_mode, infl_f, nec_f, mal_f, tile_size_native, heat_hpc=None, heat_alpha=0.6, max_tiles=6000):
    records = []

    if df is None or df.empty:
        return records

    use_df = df.head(max_tiles).copy()

    for _, r in use_df.iterrows():
        if infl_f is not None and str(r.get("inflammation", "")).strip().lower() != infl_f:
            continue

        if nec_f is not None and str(r.get("necrosis", "")).strip().lower() != nec_f:
            continue
        if highlight_mode == "Heatmap":
            if heat_hpc is None:
                continue
            col = f"p_hpc_{int(heat_hpc)}"

            if col not in use_df.columns:
                continue

            pval = float(pd.to_numeric(pd.Series([r.get(col, 0.0)]), errors="coerce").fillna(0.0).iloc[0])

            p_all = pd.to_numeric(use_df[col], errors="coerce").fillna(0.0)
            pmin, pmax = float(p_all.min()), float(p_all.max())
            if pmax <= pmin:
                pmax = pmin + 1e-9

            t = (pval - pmin) / (pmax - pmin)
            red = int(255 * t)
            green = 255
            blue = int(255 * (1 - t))

            records.append({
                "x": float(r["x_native"]),
                "y": float(r["y_native"]),
                "w": int(tile_size_native),
                "h": int(tile_size_native),
                "fill": f"rgba({red}, {green}, {blue}, 0.55)",
                "stroke": "rgba(255,255,255,0.25)",
                "stroke_width": 1,
                "opacity": "1.0",
            })
            continue
        if highlight_mode == "Survival Risk Heatmap":
            if "survival_risk_norm" not in use_df.columns:
                continue

            score = float(pd.to_numeric(pd.Series([r.get("survival_risk_norm", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            score = max(-1.0, min(1.0, score))
            intensity = abs(score)

            if intensity < 0.02:
                continue

            alpha = 0.15 + 0.60 * intensity

            if score > 0:
                fill = f"rgba(255, 0, 0, {alpha:.3f})"
            else:
                fill = f"rgba(0, 80, 255, {alpha:.3f})"

            records.append({
                "x": float(r["x_native"]),
                "y": float(r["y_native"]),
                "w": int(tile_size_native),
                "h": int(tile_size_native),
                "fill": fill,
                "stroke": "rgba(255,255,255,0.20)",
                "stroke_width": 1,
                "opacity": "1.0",
            })
            continue
        if highlight_mode == "Inflammation":
            color = color_for_inflammation(r.get("inflammation"))
        elif highlight_mode == "Necrosis":
            color = color_for_necrosis(r.get("necrosis"))
        elif highlight_mode == "Malignant":
            color = color_for_malignant(r.get("malignant"))
        
        elif highlight_mode == "Adjacency":
            adj = st.session_state.get("adj_tile_sets")
            if adj is None:
                continue

            tkey = str(r.get("slide_tile"))
            if tkey in adj.get("a_touch", set()):
                color = color_for_adjacency_group("a_touch")
            elif tkey in adj.get("b_touch", set()):
                color = color_for_adjacency_group("b_touch")
            else:
                continue
        else:
            color = color_for_hpc(r.get("hpc_id"))
            
        records.append({
            "x": float(r["x_native"]),
            "y": float(r["y_native"]),
            "w": int(tile_size_native),
            "h": int(tile_size_native),
            "color": rgb_to_css(color),
            "slide_tile": str(r.get("slide_tile", "")),
            "hpc_id": "" if pd.isna(r.get("hpc_id")) else str(int(r.get("hpc_id"))),
        })

    return records


def _legend_coverage_pairs(df: pd.DataFrame, mode: str, heat_hpc: int | None) -> list[tuple[str, float]]:
    """(label, percent) for each category; percents sum to ~100 over all tiles in ``df``."""
    n = len(df)
    if n == 0:
        return []

    if mode == "Inflammation":
        if "inflammation" not in df.columns:
            return [("Missing / unknown", 100.0)]

        def infl_lab(x):
            if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
                return "Missing"
            return str(x).strip().lower()

        vc = df["inflammation"].map(infl_lab).value_counts()
        return [(str(k), 100.0 * float(v) / n) for k, v in vc.items()]

    if mode == "Necrosis":
        if "necrosis" not in df.columns:
            return [("Missing / unknown", 100.0)]

        def nec_lab(x):
            if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
                return "Missing"
            return str(x).strip().lower()

        vc = df["necrosis"].map(nec_lab).value_counts()
        return [(str(k), 100.0 * float(v) / n) for k, v in vc.items()]

    if mode == "Malignant":
        if "malignant" not in df.columns:
            return [("Missing / unknown", 100.0)]

        def mal_lab(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "Missing"
            if isinstance(x, (int, np.integer)):
                return "Malignant" if bool(x) else "Non-malignant"
            s = str(x).strip().lower()
            if s in ("true", "t", "1", "yes", "y"):
                return "Malignant"
            if s in ("false", "f", "0", "no", "n"):
                return "Non-malignant"
            return "Missing"

        vc = df["malignant"].map(mal_lab).value_counts()
        return [(str(k), 100.0 * float(v) / n) for k, v in vc.items()]

    if mode == "Adjacency":
        adj = st.session_state.get("adj_tile_sets")
        if adj is None or "slide_tile" not in df.columns:
            return [("(select an HPC pair)", 100.0)]
        keys = df["slide_tile"].astype(str)
        in_a = keys.isin(adj.get("a_touch", set()))
        in_b = keys.isin(adj.get("b_touch", set()))
        a = st.session_state.get("adj_hpc_a", "?")
        b = st.session_state.get("adj_hpc_b", "?")
        other = ~(in_a | in_b)
        pairs = [
            (f"HPC {a} → {b} (green)", 100.0 * float(in_a.sum()) / n),
            (f"HPC {b} → {a} (orange)", 100.0 * float(in_b.sum()) / n),
            ("Not in selected pair", 100.0 * float(other.sum()) / n),
        ]
        return pairs

    if mode == "Heatmap":
        if heat_hpc is None:
            return []
        col = f"p_hpc_{int(heat_hpc)}"
        if col not in df.columns:
            return [("(no probability column)", 100.0)]
        p = pd.to_numeric(df[col], errors="coerce")
        valid = p.dropna()
        if valid.empty:
            return [("Missing score", 100.0)]
        q1, q2 = float(valid.quantile(0.33)), float(valid.quantile(0.66))

        def bucket(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "Missing"
            xf = float(x)
            if xf <= q1:
                return "Low P"
            if xf <= q2:
                return "Mid P"
            return "High P"

        vc = p.map(bucket).value_counts()
        return [(str(k), 100.0 * float(v) / n) for k, v in vc.items()]

    # HPC clusters (default)
    if "hpc_id" not in df.columns:
        return [("(no hpc_id)", 100.0)]
    h = pd.to_numeric(df["hpc_id"], errors="coerce")
    unlabeled = int(h.isna().sum())
    pairs: list[tuple[str, float]] = []
    vc = h.dropna().astype(int).value_counts()
    for hid, c in vc.head(14).items():
        pairs.append((f"HPC {int(hid)}", 100.0 * float(c) / n))
    if unlabeled:
        pairs.append(("Unlabeled", 100.0 * float(unlabeled) / n))
    if len(vc) > 14:
        shown = int(vc.head(14).sum())
        rest = int(vc.iloc[14:].sum())
        pairs.append((f"Other HPCs ({len(vc) - 14} ids)", 100.0 * float(rest) / n))
    return pairs


def _swatch_row(items: list[tuple[str, str]]) -> str:
    """items: (hex_bg, label)"""
    parts = []
    for bg, lab in items:
        parts.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin:4px 0;color:#0f172a;">'
            f'<span style="width:14px;height:14px;background:{bg};border:1px solid #0f172a;flex-shrink:0;"></span>'
            f"<span>{html.escape(lab)}</span></div>"
        )
    return '<div style="display:flex;flex-direction:column;gap:2px;">' + "".join(parts) + "</div>"


def build_legend_panel_html(mode: str, df: pd.DataFrame, heat_hpc: int | None) -> str:
    """Single HTML block: high-contrast panel + swatches + % tile coverage."""
    outer = (
        "position:sticky;top:3.5rem;max-height:88vh;overflow-y:auto;"
        "background:#f1f5f9;color:#0f172a;border:1px solid #64748b;border-radius:10px;"
        "padding:10px 10px 12px 10px;font-size:12px;line-height:1.35;"
        "box-shadow:0 1px 3px rgba(0,0,0,0.12);"
    )
    parts = [f'<div style="{outer}">']
    parts.append('<div style="font-weight:700;font-size:13px;color:#020617;margin-bottom:6px;">Legend</div>')

    if mode == "Inflammation":
        parts.append(
            _swatch_row(
                [
                    ("#50C878", "None-sparse"),
                    ("#FFC800", "Mild-moderate"),
                    ("#FF5050", "Marked"),
                    ("#A0A0A0", "Missing"),
                ]
            )
        )
    elif mode == "Necrosis":
        parts.append(
            _swatch_row(
                [
                    ("#3CC878", "None"),
                    ("#FFA500", "Some"),
                    ("#C82828", "Universal"),
                    ("#A0A0A0", "Missing"),
                ]
            )
        )
    elif mode == "Malignant":
        parts.append(
            _swatch_row(
                [
                    ("#FF5050", "Malignant"),
                    ("#50C878", "Non-malignant"),
                    ("#A0A0A0", "Missing"),
                ]
            )
        )
    elif mode == "Adjacency":
        a = st.session_state.get("adj_hpc_a", "?")
        b = st.session_state.get("adj_hpc_b", "?")
        parts.append(
            _swatch_row(
                [
                    ("#50C878", f"HPC {a} touching HPC {b}"),
                    ("#FFA500", f"HPC {b} touching HPC {a}"),
                ]
            )
        )
    elif mode == "Heatmap":
        hh = int(heat_hpc) if heat_hpc is not None else 0
        parts.append(
            f'<div style="color:#0f172a;font-size:11px;margin:4px 0;">'
            f"Colormap for <b>P(HPC {hh})</b>: warm = higher probability; "
            "alpha = confidence.</div>"
        )
    else:
        parts.append(
            '<div style="color:#0f172a;font-size:11px;margin:4px 0;">'
            "Tile outlines use a <b>stable color per HPC id</b>. "
            "Use <i>Highlight tiles by HPC cluster</i> to isolate one HPC.</div>"
        )

    pairs = _legend_coverage_pairs(df, mode, heat_hpc)
    parts.append('<hr style="border:none;border-top:1px solid #94a3b8;margin:10px 0;">')
    parts.append(
        '<div style="font-weight:600;color:#0f172a;margin-bottom:4px;">'
        "Tile coverage (% of tiles on this slide)</div>"
    )
    if not pairs:
        parts.append('<div style="color:#475569;font-size:11px;">No coverage data.</div>')
    else:
        for lab, pct in pairs:
            parts.append(
                f'<div style="display:flex;justify-content:space-between;gap:8px;margin:3px 0;'
                f'color:#0f172a;font-size:11px;"><span>{html.escape(lab)}</span>'
                f'<span style="font-weight:600;white-space:nowrap;">{pct:.1f}%</span></div>'
            )

    parts.append("</div>")
    return "".join(parts)


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
if "hpc_wsi_explore_id" not in st.session_state:
    st.session_state.hpc_wsi_explore_id = None
if "hpc_wsi_ranking_df" not in st.session_state:
    st.session_state.hpc_wsi_ranking_df = None


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

    slide_list = load_slide_list()
    plan = build_query_plan_v25(
        prompt,
        slide_list=slide_list,
        hpc_title_map=load_hpc_title_map(),
        valid_hpc_ids=load_valid_hpc_ids(),
    )
    path = save_query_plan_to_file(
        plan, slide_id=plan["entities"]["slide"][0] if plan["entities"]["slide"] else None
    )

    slide_id = apply_plan_to_session(plan, prompt, slide_id, get_viewer_ctx())

    with st.expander("🧩 Query Plan (debug)", expanded=False):
        st.json(plan)
        st.caption(f"Saved → {path} · planner: {plan.get('planner', '?')}")

    if should_fetch_from_db(plan):
        query_for_db = plan.get("enriched_query") or prompt
        try:
            structured_answer = chat_fetch_answer_from_db(query_for_db, engine, client)
        except Exception as e:
            structured_answer = f"⚠️ DB query failed: {e}"
    else:
        structured_answer = None

    try:
        final_answer = explain_answer(
            plan,
            structured_answer,
            user_query=prompt,
            history=st.session_state.messages,
        )
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

if st.session_state.get("hpc_wsi_explore_id") is not None:
    st.divider()
    _hid = int(st.session_state.hpc_wsi_explore_id)
    h1, h2 = st.columns([4, 1])
    with h1:
        st.subheader(f"HPC {_hid} — slides from KB (profile proportion)")
    with h2:
        if st.button("Clear HPC explorer", key="clear_hpc_wsi_explorer"):
            st.session_state.hpc_wsi_explore_id = None
            st.session_state.hpc_wsi_ranking_df = None
            for _k in ("hpc_dd_compare_a", "hpc_dd_compare_b"):
                if _k in st.session_state:
                    del st.session_state[_k]
            st.rerun()
    st.caption(
        "Same KB sources as **handle_slide**: ``hpl_profile_proportion`` (this HPC) + ``hpl_profile_summary``. "
        "Choose two slides from the dropdowns below to compare WSIs side by side."
    )
    rank_result = st.session_state.get("hpc_wsi_ranking_df")

    if rank_result is None:
        st.error("No HPC lookup has been run yet.")

    elif not rank_result["ok"]:
        st.error(f"KB lookup failed: {rank_result['error']}")

    elif rank_result["df"].empty:
        st.info("KB query succeeded, but no rows were found in `hpl_profile_proportion` for this HPC.")

    else:
        rank_df = rank_result["df"]
        _ph = "— Select slide —"
        opts_list = rank_df["slide_id"].astype(str).str.strip().str.upper().tolist()
        _choices = [_ph] + opts_list

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.selectbox("Compare — slide A", _choices, key="hpc_dd_compare_a")
        with c2:
            st.selectbox("Compare — slide B", _choices, key="hpc_dd_compare_b")
        with c3:
            if st.button("Clear A & B", key="hpc_cmp_clear_ab"):
                st.session_state.hpc_dd_compare_a = _ph
                st.session_state.hpc_dd_compare_b = _ph
                st.rerun()

        show_cols = ["slide_id", "proportion", "samples", "kb_summary"]
        show_cols = [c for c in show_cols if c in rank_df.columns]
        st.dataframe(
            rank_df[show_cols].rename(
                columns={
                    "slide_id": "Slide",
                    "proportion": "Proportion (KB)",
                    "samples": "Samples (KB)",
                    "kb_summary": "Slide summary (KB)",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        left = st.session_state.get("hpc_dd_compare_a")
        right = st.session_state.get("hpc_dd_compare_b")
        if left == _ph:
            left = None
        if right == _ph:
            right = None

        if left and right and left != right:
            st.divider()
            st.subheader("Side-by-side viewers (same controls as main `show_wsi`)")
            v1, v2 = st.columns(2)
            with v1:
                st.markdown(f"### {left}")
                show_wsi(left, ui_suffix="cmpL")
            with v2:
                st.markdown(f"### {right}")
                show_wsi(right, ui_suffix="cmpR")
        elif left and right and left == right:
            st.info("Slide A and B are the same — pick **two different** slides in the dropdowns.")

if st.session_state.viewer_open and slide_id:
    show_wsi(slide_id)

if rerun_needed:
    st.rerun()
