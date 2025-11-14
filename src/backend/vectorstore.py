import streamlit as st
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from src.config import LANCEDB_FOLDER, EMBEDDING_MODEL

@st.cache_resource
def load_vectorstore():
    db = lancedb.connect(LANCEDB_FOLDER)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Ensure rag_table exists
    if "rag_table" not in db.table_names():
        # Create table if it doesn't exist
        # Define a dummy schema for the table
        sample_vector = embeddings.embed_query("sample text")
        schema = {
            "vector": sample_vector,
            "text": "sample text", # Document content
            "metadata": {} # For any associated metadata
        }
        # Create with a dummy entry to define the schema
        db.create_table("rag_table", data=[schema])
        # Open the newly created table to delete the dummy entry
        table = db.open_table("rag_table")
        table.delete("text = 'sample text'")
    
    # Open the rag_table
    table = db.open_table("rag_table")

    return LanceDB(connection=db, table_name="rag_table", embedding=embeddings)
