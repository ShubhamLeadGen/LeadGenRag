import os
from dotenv import load_dotenv
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.config import EMBEDDING_MODEL, LANCEDB_FOLDER
from src.backend.remember_storage import get_embedding, REMEMBER_USER_DATA_TABLE_NAME

def check_remember_data_retrieval(query_text: str):
    load_dotenv()

    print(f"\n--- Checking Retrieval from {REMEMBER_USER_DATA_TABLE_NAME}.lance ---")
    print(f"Query: '{query_text}'")

    try:
        # Initialize embedding model
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Connect to LanceDB
        db = lancedb.connect(LANCEDB_FOLDER)

        if REMEMBER_USER_DATA_TABLE_NAME not in db.table_names():
            print(f"Error: LanceDB table '{REMEMBER_USER_DATA_TABLE_NAME}' not found.")
            print("Please ensure you have used the 'remember this' feature at least once.")
            return

        table = db.open_table(REMEMBER_USER_DATA_TABLE_NAME)

        # Generate embedding for the query
        query_vector = embeddings.embed_query(query_text)

        # Perform similarity search
        # We'll retrieve a few documents to see if anything comes up
        search_results = table.search(query_vector).limit(5).to_list()

        if search_results:
            print(f"Found {len(search_results)} relevant documents:")
            for i, doc in enumerate(search_results):
                print(f"\n--- Document {i+1} ---")
                print(f"Query: {doc.get('query')}")
                print(f"Response: {doc.get('response')}")
                print(f"Timestamp: {doc.get('timestamp')}")
                print(f"_distance: {doc.get('_distance')}") # Lower distance means higher similarity
        else:
            print("No relevant documents found in 'rememberUserData.lance' for the given query.")

    except Exception as e:
        print(f"An error occurred during retrieval check: {e}")

if __name__ == "__main__":
    # Example usage:
    # Replace with a query that you expect to retrieve remembered data
    sample_query = input("Enter a sample query to test retrieval from remembered data: ")
    if sample_query:
        check_remember_data_retrieval(sample_query)
    else:
        print("No query provided. Exiting.")
