import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import text, inspect
from textblob import TextBlob
import re

# --- DB connection ---
DB_USER = "vaidehipandya"         
DB_PASS = "vjp007"  
DB_HOST = "localhost"       
DB_PORT = "5432"
DB_NAME = "hpl_kb"
engine = create_engine( f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def fetch_answer_from_db(query):
    q = query.lower()
    conn = engine.connect()
    insp = inspect(engine)

    is_malignant_question = "malignant epithelium" in q or "not malignant epithelium" in q
    is_negative = "not malignant epithelium" in q  # i.e. asking for non-malignant explicitly

    sample_match = re.search(r"\bTCGA-\d{2}-\d{4}\b", query, re.I)
    slide_match  = re.search(r"\bTCGA-\d{2}-\d{4}-[A-Z0-9\-]+DX\d+\b", query, re.I)
    tile_match   = re.search(r"tile\s*([A-Za-z0-9_\-\.]+)", query, re.I)
    hpc_ids = re.findall(r"(?i)\bHPC(?:_ID)?[-_\s=]*([0-9]+)\b", query)

    intents = {
        "malignant": is_malignant_question,
        "negative": is_negative,        # tells us to fetch from non_malignant_details
        "tile": tile_match,
        "slide": slide_match,
        "sample": sample_match,
        "hpc": hpc_ids if hpc_ids else None
    }

    output_parts = []

    # --- Dispatch table: entity -> handler ---
    handlers = {
        "tile": handle_tile,
        "slide": handle_slide,
        "sample": handle_sample,
        "hpc": handle_hpc
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


# ==========================================================
# 👇 Each handler is clean and self-contained
# ==========================================================

def handle_tile(conn, match, intents):
    tile_name = match.group(1).replace("tile", "").strip()
    row = conn.execute(
        text("SELECT * FROM tile_registry WHERE tiles ILIKE :t OR h5_source_path ILIKE :t LIMIT 1"),
        {"t": f"%{tile_name}%"}
    ).fetchone()

    if not row:
        return f"❌ No tile found matching '{tile_name}'."

    info = dict(row._mapping)
    out = [f"### 🧩 Tile `{tile_name}` Summary"]
    out += [f"- **{k}**: {v}" for k, v in info.items()]

    hpc_id = info.get("hpc_id")

    if intents.get("malignant") and hpc_id:
        malignant_flag = not intents.get("negative") and conn.execute(
            text("SELECT malignant FROM hpc_dictionary WHERE hpc_id = :hpc_id"),
            {"hpc_id": hpc_id}
        ).scalar()

        if malignant_flag:
            kb_row = conn.execute(
                text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :hpc_id"),
                {"hpc_id": hpc_id}
            ).fetchone()
            out.append(f"⚠️ Tile belongs to **malignant epithelium** (HPC {hpc_id})")
        else:
            kb_row = conn.execute(
                text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id"),
                {"hpc_id": hpc_id}
            ).fetchone()
            out.append(f"✅ Tile belongs to **non-malignant epithelium** (HPC {hpc_id})")

        if kb_row:
            out.append("**Epithelium details from KB:**")
            for k, v in dict(kb_row._mapping).items():
                out.append(f"- **{k}**: {v}")
            out.append("")

    return "\n".join(out)

def handle_slide(conn, match, intents):
    slide_id = match.group(0).strip()
    out = [f"### 🧫 Slide `{slide_id}` Summary\n"]

    # --- 1️⃣ If user is asking about malignancy ---
    if intents.get("malignant"):
        malignant_ids = conn.execute(
            text("""
                SELECT DISTINCT hp.hpc_id
                FROM hpl_profile_proportion hp
                JOIN hpc_dictionary hd ON hp.hpc_id = hd.hpc_id
                WHERE hp.slides = :slide_id
                AND hd.malignant = TRUE
            """),
            {"slide_id": slide_id}
        ).scalars().all()

        if malignant_ids:
            out.append(f"⚠️ Malignant HPCs: {', '.join(map(str, malignant_ids))}\n")
        else:
            out.append("✅ No malignant HPCs detected.\n")

        return "\n".join(out)

    # --- 2️⃣ Otherwise: show slide-level summaries ---
    summary = conn.execute(
        text("SELECT * FROM hpl_profile_summary WHERE slides = :slide_id LIMIT 1"),
        {"slide_id": slide_id}
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
    sample_id = match.group(0)
    out = [f"### 🧬 Sample `{sample_id}` Summary"]

    summary = conn.execute(
        text("SELECT * FROM hpl_profile_summary WHERE samples = :s"),
        {"s": sample_id}
    ).fetchone()

    if summary:
        out += [f"- **{k}**: {v}" for k, v in dict(summary._mapping).items()]

    return "\n".join(out)


def handle_hpc(conn, ids, intents):
    insp = inspect(engine)
    out = []

    for hpc_id in ids:
        block = [f"## 🧠 HPC {hpc_id} Summary\n"]

        # --- 1️⃣ Dictionary Info ---
        base = conn.execute(
            text("SELECT * FROM hpc_dictionary WHERE hpc_id = :hpc_id"),
            {"hpc_id": hpc_id}
        ).fetchone()

        if not base:
            block.append(f"⚠️ No entry found for HPC {hpc_id}.\n")
            out.append("\n".join(block))
            continue

        d = dict(base._mapping)

        # --- 2️⃣ If user asked about malignancy only ---
        if intents.get("malignant"):
            malignant_flag = bool(d.get("malignant"))

            if malignant_flag:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :hpc_id"),
                    {"hpc_id": hpc_id}
                ).fetchone()
                block.append("🧬 **Malignant epithelium details:**")
            else:
                kb_row = conn.execute(
                    text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :hpc_id"),
                    {"hpc_id": hpc_id}
                ).fetchone()
                block.append("🧫 **Non-malignant epithelium details:**")

            if kb_row:
                for k, v in dict(kb_row._mapping).items():
                    block.append(f"- **{k}**: {v}")
                block.append("")

            # 🚫 Stop here — do NOT fetch tiles or other tables
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
                text(f'SELECT * FROM {table} WHERE "{id_col}" = :h LIMIT 5'),
                {"h": hpc_id}
            ).fetchall()

            if not rows:
                continue

            block.append(f"**{table.replace('_', ' ').title()}:**")
            for r in rows:
                for k, v in dict(r._mapping).items():
                    block.append(f"- **{k}**: {v}")
                block.append("")

        out.append("\n".join(block))

    return "\n\n".join(out)

# --- Streamlit app ---
st.set_page_config(page_title="HPC Chatbot", page_icon="💬")

st.title("💬 HPC Knowledge Chatbot")

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




    

