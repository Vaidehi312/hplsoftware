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

    # --- Detect query type ---
    sample_match = re.search(r"\bTCGA-\d{2}-\d{4}\b", query, re.IGNORECASE)
    slide_match = re.search(r"\bTCGA-\d{2}-\d{4}-[A-Z0-9\-]+DX\d+\b", query, re.IGNORECASE)
    tile_match = re.search(r"tile\s*([A-Za-z0-9_\-\.]+)", query, re.IGNORECASE)
    hpc_ids = re.findall(r"\b\d+\b", query)

    with engine.connect() as conn:
        insp = inspect(engine)
        tables = insp.get_table_names()
        output = ""

        # ---------- 1️⃣ TILE LOOKUP ----------
        if tile_match:
            tile_name = tile_match.group(1).replace("tile", "").strip()
            tile_row = conn.execute(
                text("SELECT * FROM tile_registry WHERE tiles ILIKE :t OR h5_source_path ILIKE :t LIMIT 1"),
                {"t": f"%{tile_name}%"}
            ).fetchone()

            if not tile_row:
                return f"❌ No tile found matching '{tile_name}'."

            output += f"### 🧩 Tile `{tile_name}` Summary\n\n"
            for k, v in dict(tile_row._mapping).items():
                output += f"- **{k}**: {v}\n"
            output += "\n"

    # Basic dictionary info for the tile's own hpc_id (direct assignment)
            hpc_id = getattr(tile_row, "hpc_id", None)
            if hpc_id is not None:
                hpc_info = conn.execute(
                    text("SELECT * FROM hpc_dictionary WHERE hpc_id = :hpc_id"),
                    {"hpc_id": hpc_id}
                ).fetchone()
                if hpc_info:
                    output += f"**Linked HPC {hpc_id} Info:**\n"
                    for k, v in dict(hpc_info._mapping).items():
                         output += f"- **{k}**: {v}\n"
                    output += "\n"

    # Look up proportions for the same slide/sample as the tile
            sample_name = getattr(tile_row, "samples", None)
            slide_name  = getattr(tile_row, "slides",  None)

    # Optional: include slide profile summary if available
            if slide_name:
                slide_sum = conn.execute(
                    text("SELECT * FROM hpl_profile_summary WHERE slides = :slide"),
                    {"slide": slide_name}
                ).fetchone()
                if slide_sum:
                        output += "**Slide Profile Summary:**\n"
                        for k, v in dict(slide_sum._mapping).items():
                            output += f"- **{k}**: {v}\n"
                        output += "\n"

    # Top HPCs by proportion for the same region (slide OR sample)
            props = conn.execute(
                text("""
                    SELECT hpc_id, proportion
                    FROM hpl_profile_proportion
                    WHERE (:slide IS NOT NULL AND slides = :slide)
                    OR (:sample IS NOT NULL AND samples = :sample)
                    ORDER BY proportion DESC
                    LIMIT 5
                    """),
                 {"slide": slide_name, "sample": sample_name}
            ).fetchall()

            if props:
                output += "**Top HPCs in this region (by proportion):**\n"
                top_hpc_id = props[0].hpc_id
                for i, r in enumerate(props, start=1):
                    trophy = " 🏆 dominant" if i == 1 else ""
                    output += f"- HPC {r.hpc_id}: {r.proportion:.4f}{trophy}\n"
                output += "\n"

        # Dictionary info for the dominant HPC
                top_info = conn.execute(
                    text("SELECT * FROM hpc_dictionary WHERE hpc_id = :hid"),
                    {"hid": top_hpc_id}
                ).fetchone()
                if top_info:
                    output += f"**Top HPC {top_hpc_id} Dictionary Info:**\n"
                    for k, v in dict(top_info._mapping).items():
                      output += f"- **{k}**: {v}\n"
                    output += "\n"

            return output

        # ---------- 2️⃣ SLIDE LOOKUP ----------
        if slide_match:
            slide_id = slide_match.group(0)
            output += f"### 🧫 Slide `{slide_id}` Summary\n\n"

            # from profile summary
            summary = conn.execute(
                text("SELECT * FROM hpl_profile_summary WHERE slides = :s"),
                {"s": slide_id}
            ).fetchone()
            if summary:
                output += "**Profile Summary:**\n"
                for k, v in dict(summary._mapping).items():
                    output += f"- **{k}**: {v}\n"
                output += "\n"

            # from proportion table
            props = conn.execute(
                text("SELECT hpc_id, proportion FROM hpl_profile_proportion WHERE slides = :s ORDER BY proportion DESC"),
                {"s": slide_id}
            ).fetchall()
            if props:
                ids = [str(r.hpc_id) for r in props]
                top = props[0]
                output += "**Linked HPCs:**\n"
                output += f"- HPC IDs: {', '.join(ids)}\n"
                output += f"- Top HPC: **{top.hpc_id}** (proportion {top.proportion:.4f})\n\n"

                # add top HPC dictionary info
                hpc_info = conn.execute(
                    text("SELECT * FROM hpc_dictionary WHERE hpc_id = :hid"),
                    {"hid": top.hpc_id}
                ).fetchone()
                if hpc_info:
                    output += f"**Top HPC {top.hpc_id} Dictionary Info:**\n"
                    for k, v in dict(hpc_info._mapping).items():
                        output += f"- **{k}**: {v}\n"
                    output += "\n"

            # linked tiles
            tile_count = conn.execute(
                text("SELECT COUNT(*) AS total FROM tile_registry WHERE slides = :s"),
                {"s": slide_id}
            ).fetchone()
            if tile_count and tile_count.total > 0:
                output += f"**Tile Registry:**\n- Total tiles linked: {tile_count.total}\n\n"

            return output or f"❌ No details found for slide `{slide_id}`."

        # ---------- 3️⃣ SAMPLE LOOKUP ----------
        if sample_match:
            sample_id = sample_match.group(0)
            output += f"### 🧬 Sample `{sample_id}` Summary\n\n"

            summary = conn.execute(
                text("SELECT * FROM hpl_profile_summary WHERE samples = :s"),
                {"s": sample_id}
            ).fetchone()
            props = conn.execute(
                text("SELECT hpc_id, proportion FROM hpl_profile_proportion WHERE samples = :s ORDER BY proportion DESC"),
                {"s": sample_id}
            ).fetchall()
            tiles = conn.execute(
                text("SELECT COUNT(*) AS total FROM tile_registry WHERE samples = :s"),
                {"s": sample_id}
            ).fetchone()

            if summary:
                output += "**Profile Summary:**\n"
                for k, v in dict(summary._mapping).items():
                    output += f"- **{k}**: {v}\n"
                output += "\n"

            if props:
                ids = [str(r.hpc_id) for r in props]
                top = props[0]
                output += "**Linked HPCs:**\n"
                output += f"- HPC IDs: {', '.join(ids)}\n"
                output += f"- Top HPC: **{top.hpc_id}** (proportion {top.proportion:.4f})\n\n"

            if tiles and tiles.total > 0:
                output += f"**Tile Registry:**\n- Total tiles linked: {tiles.total}\n\n"

            return output or f"❌ No details found for sample `{sample_id}`."

        # ---------- 4️⃣ HPC LOOKUP ----------
        if hpc_ids:
            insp = inspect(engine)
            tables = insp.get_table_names()
            output = ""

            for hpc_id in hpc_ids:
                output += f"## 🧠 HPC {hpc_id} Summary\n\n"
                base = conn.execute(
                    text("SELECT * FROM hpc_dictionary WHERE hpc_id = :hpc_id"),
                    {"hpc_id": hpc_id}
                ).fetchone()
                if not base:
                    output += f"⚠️ No entry found for HPC {hpc_id}.\n\n"
                    continue

                output += "**Dictionary Details:**\n"
                for k, v in dict(base._mapping).items():
                    output += f"- **{k}**: {v}\n"
                output += "\n"

                # fetch related rows
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

                    if rows:
                        output += f"**{table.replace('_',' ').title()}:**\n"
                        for r in rows:
                            for k, v in dict(r._mapping).items():
                                output += f"- **{k}**: {v}\n"
                            output += "\n"

                # tile count
                tile_count = conn.execute(
                    text("SELECT COUNT(*) AS total FROM tile_registry WHERE hpc_id = :hpc_id"),
                    {"hpc_id": hpc_id}
                ).fetchone()
                if tile_count and tile_count.total > 0:
                    output += f"**Tile Registry:**\n- Total tiles linked: {tile_count.total}\n\n"

            return output or "No matching HPC data found."

        # ---------- fallback ----------
        return "Please specify an HPC ID, sample, slide, or tile to search."

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




    

