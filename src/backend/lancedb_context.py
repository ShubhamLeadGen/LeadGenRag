import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import lancedb
import pandas as pd
from dotenv import load_dotenv
import shutil # Import shutil for rmtree
from datetime import datetime

# Load environment variables
load_dotenv()

# Import necessary components from your project
from src.config import LANCEDB_FOLDER
from src.backend.context_storage import CONTEXT_TABLE_NAME, get_context_table, embeddings as context_embeddings

def lancedb_context_dashboard():
    st.set_page_config(page_title="LanceDB Context Dashboard", layout="wide")
    st.title("LanceDB Context Storage Dashboard")

    st.markdown("""
        This dashboard allows you to view and manage the entries stored in the LanceDB context table.
        These entries are liked query-response pairs used to optimize RAG responses.
    """)

    st.subheader("Current Context Entries")

    try:
        table = get_context_table()

        st.write(f"Total entries in LanceDB table (reported by LanceDB): {table.count_rows()}")

        # Fetch all data from the table
        all_entries = table.to_pandas()
        
        if not all_entries.empty:
            # Drop the 'vector' column for display as it's a large array
            if 'vector' in all_entries.columns:
                all_entries = all_entries.drop(columns=['vector'])

            # Check if '_id' column exists. If not, we'll use index for keys and query/response for deletion.
            use_id_for_deletion = '_id' in all_entries.columns

            # Display entries with a delete button for each
            for index, row in all_entries.iterrows():
                col1, col2, col3, col4, col5 = st.columns([0.1, 0.2, 0.4, 0.2, 0.1])
                with col1:
                    st.write(index + 1) # Display row number
                with col2:
                    st.write(row["query"])
                with col3:
                    st.write(row["response"])
                with col4:
                    st.write(row["timestamp"])
                with col5:
                    if use_id_for_deletion:
                        delete_button_key = f"delete_{row['_id']}"
                    else:
                        delete_button_key = f"delete_idx_{index}" # Use index as key if _id is not available

                    if st.button("🗑️", key=delete_button_key):
                        if use_id_for_deletion:
                            table.delete(f"_id = '{row['_id']}'")
                        else:
                            # Fallback deletion using query and response (less reliable if not unique)
                            escaped_query = row['query'].replace("'", "''")
                            escaped_response = row['response'].replace("'", "''")
                            table.delete(f"query = '{escaped_query}' AND response = '{escaped_response}'")
                        st.success(f"Entry deleted.")
                        st.rerun()
        else:
            st.info("No entries in context storage yet.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Option to delete all entries
    if st.button("🗑️ Delete All Context Entries", help="This will permanently delete all stored query-response pairs."):
        # Forcefully remove the LanceDB directory to ensure a clean slate
        if os.path.exists(LANCEDB_FOLDER):
            shutil.rmtree(LANCEDB_FOLDER)
            st.warning(f"Forcefully removed LanceDB directory: {LANCEDB_FOLDER}")

        # Explicitly create a new, empty table
        db = lancedb.connect(LANCEDB_FOLDER)
        # We need a schema for the table. Let's define it with a dummy entry.
        # The vector dimension will be inferred from the embedding model.
        # Note: embeddings object is not directly available here, so we need to get it from context_storage
        
        sample_query_vector = context_embeddings.embed_query("sample query")
        schema = {
            "vector": sample_query_vector,
            "query": "sample query",
            "response": "sample response",
            "timestamp": datetime.now().isoformat() # datetime is not imported here
        }
        
        table = db.create_table(CONTEXT_TABLE_NAME, data=[schema])
        table.delete("query = 'sample query'")

        st.success("All context entries deleted and table reinitialized!")
        st.rerun()
if __name__ == "__main__":
    lancedb_context_dashboard()