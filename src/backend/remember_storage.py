import os


import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
from src.config import EMBEDDING_MODEL, LANCEDB_FOLDER, SIMILARITY_THRESHOLD
from functools import lru_cache
import math

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


def _cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    try:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception:
        return 0.0

@lru_cache(maxsize=1)
def get_remember_user_data_table():
    """Connects to LanceDB and returns the remember user data table, creating it if it doesn't exist."""
    db = lancedb.connect(LANCEDB_FOLDER)
    
    table_exists = REMEMBER_USER_DATA_TABLE_NAME in db.table_names()
    
    if table_exists:
        table = db.open_table(REMEMBER_USER_DATA_TABLE_NAME)
        # Check if the schema is compatible with LangChain (i.e., has 'text' and 'metadata' columns)
        # Assuming that if 'text' is not present, the schema is old.
        if "text" not in table.schema.names:
            print(f"Warning: Incompatible schema for '{REMEMBER_USER_DATA_TABLE_NAME}'. Dropping and recreating table.")
            db.drop_table(REMEMBER_USER_DATA_TABLE_NAME)
            table_exists = False
        else:
            print(f"Connected to existing compatible table: {REMEMBER_USER_DATA_TABLE_NAME}")
            return table

    if not table_exists:
        # Create table with LangChain-compatible schema
        sample_query_vector = get_embedding("sample response") # Embedding for text content
        schema = {
            "vector": sample_query_vector,
            "text": "sample response", # Content for LangChain Document.page_content
            "metadata": {             # Metadata for LangChain Document.metadata
                "query": "sample query",
                "timestamp": datetime.now().isoformat()
            }
        }
        table = db.create_table(REMEMBER_USER_DATA_TABLE_NAME, data=[schema])
        table.delete("text = 'sample response'") # Delete dummy entry
        print(f"Created new compatible table: {REMEMBER_USER_DATA_TABLE_NAME}")

    # Create an index if it doesn't exist for faster search
    if not table.list_indices():
        num_rows = table.count_rows()
        if num_rows > 1:
            num_partitions = 8
            if num_rows < 16:
                num_partitions = max(1, num_rows // 2)
            try:
                table.create_index(metric="cosine", num_partitions=num_partitions, num_sub_vectors=96)
                print(f"Created index for {REMEMBER_USER_DATA_TABLE_NAME}")
            except Exception as e:
                print(f"Error creating index for {REMEMBER_USER_DATA_TABLE_NAME}: {e}")

    return table

def add_to_remember_user_data_storage(query, response):
    """Adds a new query-response pair to the LanceDB remember user data table using LangChain-compatible schema."""
    table = get_remember_user_data_table()
    # Embed the response as the main text content for similarity search
    response_vector = get_embedding(response) 
    data = [{
        "vector": response_vector,
        "text": response, # Store response as main text content
        "metadata": {      # Store query and timestamp in metadata
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
    }]
    table.add(data)

def find_in_remember_user_data_storage(query):
    """
    Finds a similar response from the LanceDB remember user data table.
    Returns the response if a similar query is found, otherwise None.
    """
    table = get_remember_user_data_table()
    query_vector = get_embedding(query)

    # Perform similarity search (top-k)
    # The search should now be on the 'text' column implicitly
    search_results = table.search(query_vector).limit(5).to_list()

    if not search_results:
        return None

    best = None
    best_sim = 0.0
    candidates = []
    for cand in search_results:
        # LangChain's LanceDB integration stores the original vector under '_vector' if it
        # doesn't match the embedding function's vector field. Or directly in 'vector'.
        # Assuming 'vector' is the stored embedding
        stored_vec = cand.get("vector") 
        
        # Check if metadata exists and get query/timestamp from there
        metadata = cand.get("metadata", {})
        original_query = metadata.get("query")
        timestamp = metadata.get("timestamp")

        # The actual response content is now in 'text'
        response_content = cand.get("text")
        
        # Calculate similarity (this part of the helper _cosine_similarity is not strictly needed if _distance is good)
        if stored_vec:
            sim = _cosine_similarity(query_vector, stored_vec)
        else: # Fallback to using _distance provided by LanceDB
            sim = 1 - float(cand.get("_distance", 1.0))

        candidates.append({
            "query": original_query, # Use original_query from metadata
            "response": response_content, # Use response_content from text
            "timestamp": timestamp, # Use timestamp from metadata
            "_distance": cand.get("_distance"),
            "similarity": sim,
        })

        if sim > best_sim:
            best_sim = sim
            best = cand

    if best and best_sim >= SIMILARITY_THRESHOLD:
        return best.get("text") # Return text, which is the response

    return None


def debug_find_in_remember(query, top_k=5):
    table = get_remember_user_data_table()
    query_vector = get_embedding(query)
    results = table.search(query_vector).limit(top_k).to_list()
    out = []
    for r in results:
        sim = 0.0
        stored_vec = r.get("vector")
        metadata = r.get("metadata", {})
        original_query = metadata.get("query")
        timestamp = metadata.get("timestamp")
        response_content = r.get("text")

        if stored_vec:
            sim = _cosine_similarity(query_vector, stored_vec)
        else:
            sim = 1 - float(r.get("_distance", 1.0))
        out.append({
            "query": original_query,
            "response": response_content,
            "timestamp": timestamp,
            "_distance": r.get("_distance"),
            "similarity": sim,
        })
    return out

