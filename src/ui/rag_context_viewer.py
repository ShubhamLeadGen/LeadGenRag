import streamlit as st
import lancedb
import pandas as pd
import sys
import os

# Add the project root directory to the Python path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.config import LANCEDB_FOLDER

# --- Page Config ---
st.set_page_config(
    page_title="RAG Context Viewer",
    page_icon="📚",
    layout="wide"
)

# --- Load Data ---
@st.cache_resource
def load_rag_context_table():
    """Loads the rag_context_table from LanceDB."""
    db = lancedb.connect(LANCEDB_FOLDER)
    table_names = db.table_names()
    if "rag_context_table" not in table_names:
        st.error(f"Error: Table 'rag_context_table' not found in LanceDB. Available tables: {table_names}")
        return None
    table = db.open_table("rag_context_table")
    return table

def fetch_all_data(table):
    """Fetches all data from the table and returns a Pandas DataFrame."""
    if table is None:
        return pd.DataFrame()
    all_data = table.to_pandas()
    return all_data

# --- UI Components ---
st.title("RAG Context Viewer 📚")
st.markdown(
    """
    This dashboard allows you to view the conversation history (queries and answers)
    stored in your RAG system's `rag_context_table`.
    """
)

try:
    table = load_rag_context_table()
    if table:
        df = fetch_all_data(table)

        st.sidebar.success(f"Successfully connected to LanceDB table 'rag_context_table' at `{LANCEDB_FOLDER}`.")
        st.sidebar.info(f"**{len(df)}** records loaded.")
        st.sidebar.markdown("---")
        st.sidebar.info("Data is cached for performance. Click below to refresh.")
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_resource.clear()
            st.rerun()

        # --- Data Viewer ---
        st.subheader("Browse Stored Q&A")
        st.markdown("A complete view of all the queries and their corresponding answers.")

        if not df.empty:
            # Create a dataframe for display, excluding the vector for readability
            display_df = df.copy()
            if 'vector' in display_df.columns:
                display_df = display_df.drop(columns=['vector'])

            # Ensure 'query' and 'response' columns exist
            required_columns = {'query', 'response'}
            if not required_columns.issubset(display_df.columns):
                st.warning(f"The table must contain 'query' and 'response' columns to display the Q&A. Found columns: {list(display_df.columns)}")
            else:
                st.dataframe(
                    display_df,
                    width='stretch',
                    hide_index=True,
                    column_order=("query", "response"), # Prioritize query and response
                    column_config={
                        "query": st.column_config.TextColumn("Query", help="The user's query.", width="medium"),
                        "response": st.column_config.TextColumn("Response", help="The generated answer.", width="large"),
                    }
                )
        else:
            st.info("The 'rag_context_table' is currently empty.")


except Exception as e:
    st.error(
        f"""
        **Failed to load LanceDB table.**

        Please ensure that:
        1. The application has run and saved some context.
        2. The `LANCEDB_FOLDER` in your `config.py` is correct (`{LANCEDB_FOLDER}`).
        3. The table name is 'rag_context_table'.

        **Error:** `{e}`
        """
    )
