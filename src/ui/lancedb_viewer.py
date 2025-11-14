import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import lancedb
import pandas as pd
from src.config import LANCEDB_FOLDER, EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings

# --- Page Config ---
st.set_page_config(
    page_title="LanceDB Vector Store Viewer",
    page_icon="🔎",
    layout="wide"
)

# --- Load Data ---
@st.cache_resource
def load_lancedb_table():
    """Loads the LanceDB table and embeddings model."""
    db = lancedb.connect(LANCEDB_FOLDER)
    table = db.open_table("rag_table")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return table, embeddings

def fetch_all_data(table):
    """Fetches all data from the table and returns a Pandas DataFrame."""
    all_data = table.to_pandas()
    return all_data

# --- UI Components ---
st.title("LanceDB Vector Store Inspector 🔎")
st.markdown(
    """
    This dashboard allows you to visually inspect, search, and monitor
    the embeddings and metadata stored in your RAG system's LanceDB vector store.
    """
)

try:
    table, embeddings = load_lancedb_table()
    df = fetch_all_data(table)

    st.sidebar.success(f"Successfully connected to LanceDB table 'rag_table' at `{LANCEDB_FOLDER}`.")
    st.sidebar.info(f"**{len(df)}** vectors loaded.")
    st.sidebar.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL}`")
    st.sidebar.markdown("---")
    st.sidebar.info("Data is cached for performance. Click below to refresh.")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()


    # --- Search ---
    st.subheader("Search Stored Embeddings")
    search_query = st.text_input("Enter a query to search for similar documents:", placeholder="e.g., What are the latest sales figures?")

    if search_query:
        with st.spinner("Searching for similar vectors..."):
            query_embedding = embeddings.embed_query(search_query)
            search_results = table.search(query_embedding).limit(5).to_pandas()

            st.success(f"Found **{len(search_results)}** similar results for '{search_query}':")

            for i, row in search_results.iterrows():
                with st.expander(f"**Result {i+1}** (Distance: {row['_distance']:.4f}) | **Source:** `{row['source']}`"):
                    st.text_area("Text", row['text'], height=150, key=f"search_text_{i}")
                    st.json({k: v for k, v in row.items() if k not in ['text', 'vector', '_distance']})


    # --- Data Viewer ---
    st.subheader("Browse All Stored Data")
    st.markdown("A complete view of all the text chunks and their associated metadata.")

    # Create a dataframe for display, excluding the vector for readability
    display_df = df.drop(columns=['vector'])

    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "text": st.column_config.TextColumn("Text Content", help="The actual text chunk.", width="large"),
            "source": st.column_config.TextColumn("Source", help="The origin of the document."),
        }
    )


except Exception as e:
    st.error(
        f"""
        **Failed to load LanceDB table.**

        Please ensure that:
        1. You have run the `ingest.py` script to create the vector store.
        2. The `LANCEDB_FOLDER` in your `config.py` is correct (`{LANCEDB_FOLDER}`).
        3. The table name is 'rag_table'.

        **Error:** `{e}`
        """
    )
