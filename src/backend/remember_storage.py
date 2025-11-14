import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
from src.config import EMBEDDING_MODEL, LANCEDB_FOLDER, SIMILARITY_THRESHOLD
from functools import lru_cache

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# LanceDB table name for context storage
REMEMBER_USER_DATA_TABLE_NAME = "rememberUserData"

# In-memory cache for embeddings
_embedding_cache = {}

def get_embedding(query):
    """
    Gets the embedding for a query, using a cache to avoid re-computation.
    """
    if query not in _embedding_cache:
        _embedding_cache[query] = embeddings.embed_query(query)
    return _embedding_cache[query]

@lru_cache(maxsize=1)
def get_remember_user_data_table():
    """Connects to LanceDB and returns the remember user data table, creating it if it doesn't exist."""
    db = lancedb.connect(LANCEDB_FOLDER)
    
    if REMEMBER_USER_DATA_TABLE_NAME not in db.table_names():
        # Create table if it doesn't exist
        sample_query_vector = get_embedding("sample query")
        schema = {
            "vector": sample_query_vector,
            "query": "sample query",
            "response": "sample response",
            "timestamp": datetime.now().isoformat()
        }
        table = db.create_table(REMEMBER_USER_DATA_TABLE_NAME, data=[schema])
        table.delete("query = 'sample query'")
    else:
        table = db.open_table(REMEMBER_USER_DATA_TABLE_NAME)

    # Create an index if it doesn't exist for faster search
    if not table.list_indices():
        num_rows = table.count_rows()
        if num_rows > 1:
            num_partitions = 8
            if num_rows < 16:
                num_partitions = max(1, num_rows // 2)
            try:
                table.create_index(metric="cosine", num_partitions=num_partitions, num_sub_vectors=96)
            except Exception as e:
                print(f"Error creating index: {e}")

    return table

def add_to_remember_user_data_storage(query, response):
    """Adds a new query-response pair to the LanceDB remember user data table."""
    table = get_remember_user_data_table()
    query_vector = get_embedding(query)
    data = [{
        "vector": query_vector,
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }]
    table.add(data)

def find_in_remember_user_data_storage(query):
    """
    Finds a similar response from the LanceDB remember user data table.
    Returns the response if a similar query is found, otherwise None.
    """
    table = get_remember_user_data_table()
    query_vector = get_embedding(query)
    
    # Perform similarity search
    search_results = table.search(query_vector).limit(1).to_list()
    
    if search_results:
        best_match = search_results[0]
        similarity_score = 1 - best_match["_distance"]
        
        if similarity_score >= SIMILARITY_THRESHOLD:
            return best_match["response"]
    
    return None
