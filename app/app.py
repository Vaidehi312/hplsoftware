import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import text, inspect
# --- DB connection ---
DB_USER = "vaidehipandya"         
DB_PASS = "vjp007"  
DB_HOST = "localhost"       
DB_PORT = "5432"
DB_NAME = "hpl_kb"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

from sqlalchemy import text, inspect

def fetch_answer_from_db(query):
    with engine.connect() as conn:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        results = []

        # Lowercase query for flexible matching
        q = query.lower()

        for table in tables:
            # Match if the table name itself is relevant
            if q in table.lower():
                results.append((table, {"_meta": f"Match found in table name: {table}"}))
                continue

            # Get all columns and their types
            columns = inspector.get_columns(table)
            text_columns = [c['name'] for c in columns if c['type'].__class__.__name__ in ['TEXT', 'VARCHAR']]

            # --- 1️⃣ Match on column names ---
            for col in columns:
                if q in col['name'].lower():
                    results.append((table, {"_meta": f"Match found in column name: {col['name']} (table {table})"}))

            # --- 2️⃣ Search inside column values ---
            if not text_columns:
                continue
            conditions = " OR ".join([f"{col} ILIKE '%{query}%'" for col in text_columns])
            sql = text(f"SELECT * FROM {table} WHERE {conditions} LIMIT 2;")
            try:
                fetched = conn.execute(sql).fetchall()
                for row in fetched:
                    results.append((table, dict(row._mapping)))
            except Exception:
                continue

        # --- Format results ---
        if results:
            response = "### Here’s what I found across your knowledge base:\n\n"
            for table, record in results:
                response += f"**Table:** `{table}`\n"
                for key, value in record.items():
                    if key == "_meta":
                        response += f"- {value}\n"
                    else:
                        response += f"- **{key}:** {value}\n"
                response += "\n"
            return response
        else:
            return "🤔 I couldn’t find that in any table name, column name, or values."

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