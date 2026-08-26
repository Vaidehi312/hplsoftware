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

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

import re

def fetch_answer_from_db(query):
    with engine.connect() as conn:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        results = []

        # Extract any numbers from the user query (e.g., "HPC 35" → 35)
        numbers = [int(n) for n in re.findall(r'\d+', query)]
        if not numbers:
            return "Please specify which HPC ID or system you’d like to know about."

        # Exclude columns you don't want (like 'h_latent')
        exclude_cols = {"h_latent", "h_latent_vector"}

        for table in tables:
            columns = inspector.get_columns(table)
            col_names = [c["name"] for c in columns]

            # Only consider tables that have hpc_id or a related key
            id_cols = [c for c in col_names if "hpc_id" in c.lower()]
            if not id_cols:
                continue  # skip unrelated tables

            for num in numbers:
                for id_col in id_cols:
                    sql = text(f"SELECT * FROM {table} WHERE {id_col} = :num LIMIT 5;")
                    try:
                        fetched = conn.execute(sql, {"num": num}).fetchall()
                        for row in fetched:
                            record = dict(row._mapping)
                            # Remove excluded columns
                            record = {k: v for k, v in record.items() if k not in exclude_cols}
                            results.append((table, record))
                    except Exception as e:
                        continue

        # --- Format results ---
        if results:
            response = f"### Results for HPC ID(s) {', '.join(map(str, numbers))}:\n\n"
            for table, record in results:
                response += f"**Table:** `{table}`\n"
                for key, value in record.items():
                    response += f"- **{key}:** {value}\n"
                response += "\n"
            return response
        else:
            return f"🤔 I couldn’t find any details for HPC ID {', '.join(map(str, numbers))}."
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




    

