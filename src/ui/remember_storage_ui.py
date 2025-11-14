import streamlit as st
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.backend.remember_storage import get_remember_user_data_table
from src.config import EMBEDDING_MODEL, LANCEDB_FOLDER

def main():
    load_dotenv()
    st.set_page_config(page_title="Remembered User Data Viewer", layout="wide")
    st.title("Remembered User Data Viewer")

    try:
        # Get the remember user data table
        remember_user_data_table = get_remember_user_data_table()

        # Fetch all data from the table
        # LanceDB's search() method can be used to retrieve all entries by searching for an empty query
        # or by iterating through the table directly if a full scan is desired.
        # For simplicity, we'll fetch all records.
        # Note: LanceDB tables are not directly iterable like a list.
        # We need to use a query to get all data.
        # A common way to get all data is to query with a vector that matches everything,
        # or if the schema allows, a simple SQL-like query.
        # For now, let's assume we can fetch all records.
        # If the table is empty, `to_pandas()` might return an empty DataFrame.

        # A more robust way to get all data from LanceDB is to read it as a pandas DataFrame
        # or iterate through its fragments. For a simple viewer, to_pandas() is convenient.
        all_data = remember_user_data_table.to_pandas()

        if not all_data.empty:
            st.subheader("All Remembered Data")
            # Display the data, excluding the 'vector' column for readability
            st.dataframe(all_data.drop(columns=['vector'], errors='ignore'))
        else:
            st.info("No remembered user data found yet.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure the LanceDB folder is correctly configured and data has been added using the 'remember this' feature.")

if __name__ == "__main__":
    main()
