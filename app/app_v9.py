import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import text, inspect
from textblob import TextBlob
import re
import numpy as np
from PIL import Image
import io
import h5py

# --- DB connection ---
DB_USER = "vaidehipandya"         
DB_PASS = "vjp007"  
DB_HOST = "localhost"       
DB_PORT = "5432"
DB_NAME = "hpl_kb"
engine = create_engine( f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
#h5_path = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"

def detect_entity_patterns(query):
    """
    Dynamically detect entities (sample, slide, tile, HPC)
    using keywords like 'sample', 'slide', etc.
    """
    q = query.strip().lower()

    # Helper: get whatever text follows a keyword (e.g., "slide TCGA-44-A47B-01Z-00-DX1")
    def after(keyword):
        m = re.search(rf"{keyword}\s*[:=]?\s*([^\s,;]+)", q, re.I)
        return m.group(1).strip() if m else None

    # Direct keyword-based matches
    tile_match = after("tile")
    slide_match = after("slide")
    sample_match = after("sample")
    hpc_match = after("hpc")

    # Fallbacks (in case user didn’t type the keyword)
    if not tile_match:
        m = re.search(r"\b(tile[_\-]?\d+|[A-Za-z0-9_\-]+\.jpe?g|[A-Za-z0-9_\-]+\.png|[A-Za-z0-9_\-]+\.tif)\b", q)
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

    return "\n".join(out)

def handle_slide(conn, match, intents):
    slide_id = match if isinstance(match, str) else match.group(0)
    slide_id = re.sub(r"^slide\s+", "", slide_id, flags=re.I).strip().upper()
    out = [f"### 🧫 Slide `{slide_id}` Summary\n"]

    # --- 1️⃣ If user is asking about malignancy ---
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
            out.append(f" Malignant HPCs: {', '.join(map(str, malignant_ids))}\n")
        else:
            out.append(" No malignant HPCs detected.\n")

        return "\n".join(out)

    # --- 2️⃣ Otherwise: show slide-level summaries ---
    summary = conn.execute(
        text("SELECT * FROM hpl_profile_summary WHERE slides ILIKE :slide_id LIMIT 1"),
        {"slide_id": f"%{slide_id}%"}
    ).fetchone()

    if summary:
        out.append("**Slide Summary (from hpl_profile_summary):**")
        for k, v in dict(summary._mapping).items():
            out.append(f"- **{k}**: {v}")
        out.append("")

    proportions = conn.execute(
        text("""
            SELECT hpc_id, proportion, samples
            FROM hpl_profile_proportion
            WHERE slides = :slide_id
            ORDER BY proportion DESC
            LIMIT 10
        """),
        {"slide_id": slide_id}
    ).fetchall()

    if proportions:
        out.append("**Top HPCs (from hpl_profile_proportion):**")
        for row in proportions:
            d = dict(row._mapping)
            out.append(
                f"- **HPC {d['hpc_id']}** → proportion: {d['proportion']:.5f}, sample: {d['samples']}"
            )
        out.append("")

    if not summary and not proportions:
        out.append("⚠️ No records found for this slide in either table.")

    return "\n".join(out)


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

            # 🚫 Stop here — do NOT fetch tiles or other tables
            out.append("\n".join(block))
            continue

        # --- 3️ Otherwise (normal query, not malignancy question) ---
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

        # --- 4 NEW: Display linked tile images ---
        h5_path = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"
        tile_rows = conn.execute(
            text("SELECT tiles, image_index FROM tile_registry WHERE hpc_id ILIKE :hpc_id LIMIT 5"),
            {"hpc_id": hpc_id}
        ).fetchall()

        if tile_rows:
            block.append("**Linked Tiles (with images):**")
            st.subheader(f" Example Tiles for HPC {hpc_id}")

            try:
                with h5py.File(h5_path, "r") as f:
                    dataset = f["train_img"]
                    total = dataset.shape[0]

                    # Create 3-column layout for images
                    cols = st.columns(3)

                    for i, row in enumerate(tile_rows[:6]):
                        idx = row.image_index
                        if idx is None or idx >= total:
                            continue

                        tile = dataset[idx]
                        tile_min, tile_max = np.min(tile), np.max(tile)
                        if tile_max > tile_min:
                            tile = (tile - tile_min) / (tile_max - tile_min) * 255
                        tile = tile.astype(np.uint8)

                        img = Image.fromarray(tile).convert("RGB")
                        col = cols[i % 3]
                        col.image(img, caption=f"{row.tiles} (index {idx})", use_container_width=True)

            except Exception as e:
                st.warning(f" Could not load tiles from H5: {e}")

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
        if st.button("💾 Save this tile as JPEG"):
            save_path = f"tile_{tile_index}.jpeg"
            img.save(save_path, "JPEG")
            st.success(f"Saved: {save_path}")

except Exception as e:
    st.error(f" Error reading H5 file: {e}")

# --- Streamlit app ---

st.set_page_config(page_title="HPC Chatbot", page_icon="💬")

st.title("🧠 HPC Chatbot with Image Upload")

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

# Chat input
prompt = st.chat_input("Ask something about HPCs...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer = fetch_answer_from_db(prompt)

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})




    

