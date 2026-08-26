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

    # ---------- 1) SAMPLE LOOKUP ----------
    sample_match = re.search(r"\bTCGA-\d{2}-\d{4}\b", query, re.IGNORECASE)
    if sample_match:
        sample_id = sample_match.group(0)
        with engine.connect() as conn:
            # profile summary
            summary = conn.execute(
                text("SELECT * FROM hpl_profile_summary WHERE samples = :s"),
                {"s": sample_id}
            ).fetchone()

            # proportions (sorted largest first)
            props = conn.execute(
                text(
                    "SELECT hpc_id, proportion "
                    "FROM hpl_profile_proportion "
                    "WHERE samples = :s "
                    "ORDER BY proportion DESC"
                ),
                {"s": sample_id}
            ).fetchall()

            # tiles count (optional, if present)
            tiles = conn.execute(
                text("SELECT COUNT(*) AS tile_count FROM tile_registry WHERE samples = :s"),
                {"s": sample_id}
            ).fetchone()

        if not summary and not props:
            return f"❌ No data found for sample **{sample_id}**."

        out = [f"### 🧬 Sample {sample_id} Summary\n"]

        if summary:
            out.append("**Profile Summary:**")
            for k, v in dict(summary._mapping).items():
                out.append(f"- **{k}**: {v}")
            out.append("")

        if props:
            ids = [str(r.hpc_id) for r in props]
            top = props[0]
            out.append("**Linked HPCs (from proportion):**")
            out.append(f"- HPC IDs: {', '.join(ids)}")
            out.append(f"- Top contributing HPC: **{top.hpc_id}** (proportion {top.proportion:.4f})")
            out.append("")

        if tiles and getattr(tiles, "tile_count", 0) > 0:
            out.append("**Tile Registry:**")
            out.append(f"- Total tiles linked: {tiles.tile_count}")
            out.append("")

        return "\n".join(out)

    # ---------- 2) HPC LOOKUP ----------
    # (sample handled first, so numbers inside TCGA won't confuse this)
    hpc_ids = re.findall(r"\b\d+\b", query)
    if not hpc_ids:
        return "Please mention either an HPC ID (e.g., 'HPC 35') or a valid sample ID (e.g., 'TCGA-35-3615')."

    out_lines = []
    with engine.connect() as conn:
        insp = inspect(engine)
        tables = insp.get_table_names()

        for hpc_id in hpc_ids:
            out_lines.append(f"## 🧠 HPC {hpc_id} Summary\n")

            # base dictionary
            base = conn.execute(
                text("SELECT * FROM hpc_dictionary WHERE hpc_id = :hid"),
                {"hid": hpc_id}
            ).fetchone()

            if not base:
                out_lines.append(f"⚠️ No entry found for HPC {hpc_id}.\n")
                continue

            out_lines.append("**Dictionary Details:**")
            for k, v in dict(base._mapping).items():
                out_lines.append(f"- **{k}**: {v}")
            out_lines.append("")

            # scan other tables that relate via hpc_id or dominant_hpc
            for table in tables:
                if table in ("hpc_dictionary", "h_latent_vectors"):
                    continue
                cols = [c["name"] for c in insp.get_columns(table)]
                if "hpc_id" in cols:
                    id_col = "hpc_id"
                elif "dominant_hpc" in cols:
                    id_col = "dominant_hpc"
                else:
                    continue

                rows = conn.execute(
                    text(f'SELECT * FROM {table} WHERE "{id_col}" = :hid LIMIT 5'),
                    {"hid": hpc_id}
                ).fetchall()

                if rows:
                    out_lines.append(f"**{table.replace('_',' ').title()}:**")
                    for r in rows:
                        for k, v in dict(r._mapping).items():
                            out_lines.append(f"- **{k}**: {v}")
                        out_lines.append("")

            # tiles count by hpc_id
            tiles = conn.execute(
                text("SELECT COUNT(*) AS total_tiles FROM tile_registry WHERE hpc_id = :hid"),
                {"hid": hpc_id}
            ).fetchone()
            if tiles and tiles.total_tiles > 0:
                out_lines.append("**Tile Registry:**")
                out_lines.append(f"- Total tiles linked: {tiles.total_tiles}")
                out_lines.append("")

    return "\n".join(out_lines) if out_lines else "No matching data found."
    
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




    

