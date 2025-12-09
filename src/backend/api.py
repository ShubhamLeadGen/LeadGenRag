import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add the project root directory to the Python path.
project_root = os.path.dirname(os.path.abspath(__file__))
# Assuming api.py is in src/backend, project root is two levels up
project_root = os.path.abspath(os.path.join(project_root, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

from src.backend.qa_chain import build_qa_chain

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from the Chrome extension
# In a production environment, you should restrict origins to your extension's ID or specific domains.
origins = [
    "http://localhost",
    "http://localhost:8000", # FastAPI default
    "http://localhost:8501", # Streamlit default
    "chrome-extension://*", # Allow all Chrome extensions for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all origins. RESTRICT THIS IN PRODUCTION.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the query request body
class QueryRequest(BaseModel):
    query: str
    verbosity: str = "Normal"
    strict_mode: bool = False

# Global QA chain instance (initialized on first request or startup)
qa_chain_instance = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the QA chain when the FastAPI application starts up.
    """
    global qa_chain_instance
    print("Initializing QA chain on startup...")
    # Initialize with default values for verbosity and strict_mode.
    # These can be overridden per request.
    try:
        qa_chain_instance, _ = build_qa_chain(verbosity="Normal", strict_mode=False)
        print("QA chain initialized successfully.")
    except Exception as e:
        print(f"Error initializing QA chain on startup: {e}")
        # Depending on severity, you might want to exit or log a critical error
        # but allow the app to start to provide an error message.

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Endpoint to receive a query, process it using the RAG QA chain, and return the response.
    """
    global qa_chain_instance

    if qa_chain_instance is None:
        raise HTTPException(status_code=503, detail="QA chain not initialized. Please try again in a moment.")

    try:
        # Rebuild QA chain with specified verbosity/strict_mode from request,
        # or use the cached one if settings haven't changed.
        # Note: build_qa_chain itself is not cached, but init_llm_cached etc. are.
        # This ensures the prompt is updated correctly.
        current_qa_chain, _ = build_qa_chain(
            verbosity=request.verbosity,
            strict_mode=request.strict_mode
        )
        
        response = current_qa_chain.invoke({"query": request.query})
        
        # The 'result' key contains the generated answer
        answer = response.get("result", "Sorry, I could not generate a response.")
        
        # You might want to include source documents in the API response too
        source_documents = [doc.metadata for doc in response.get("source_documents", [])]

        return {"answer": answer, "sources": source_documents}
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "RAGHF API is running! Access /query for RAG functionality."}
