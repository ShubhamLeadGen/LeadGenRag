import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import base64
import hashlib
import pandas as pd
import pyarrow as pa
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
import lancedb
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain.docstore.document import Document
from unstructured.partition.docx import partition_docx
from unstructured.documents.elements import Table
from io import StringIO
import tempfile
import shutil

from src.config import (
    EMBEDDING_MODEL,
    LANCEDB_FOLDER,
    DATA_FOLDER,
    TEXT_SPLITTER_CHUNK_SIZE,
    TEXT_SPLITTER_CHUNK_OVERLAP,
)

# Load env variables
load_dotenv()

# --- Configuration ---
TABLE_NAME = "rag_table"
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
BATCH_SIZE = 100


# --- Helper Functions ---
def calculate_file_hash(file_path):
    """Calculates the SHA256 hash of a local file's content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_content_hash(content: str):
    """Calculates the SHA256 hash of a string content."""
    hasher = hashlib.sha256()
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()


def describe_image(image_data: bytes, api_key: str) -> str:
    """Uses a multimodal model to describe an image."""
    try:
        llm = ChatAnthropic(model="claude-3-opus-20240229", api_key=api_key)
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
            {"type": "text", "text": "Describe the diagram, chart, or image in detail. If it is a table, try to represent it in Markdown format."}
        ])
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f" Error describing image: {e}")
        return ""


def parse_docx(file_path: str) -> str:
    """Parses a .docx file and returns its content as Markdown."""
    elements = partition_docx(filename=file_path, include_page_breaks=False)
    content = []
    for el in elements:
        if isinstance(el, Table):
            try:
                df = pd.read_html(StringIO(el.metadata.text_as_html))[0]
                content.append(df.to_markdown(index=False))
            except Exception:
                content.append(el.text)
        else:
            content.append(el.text)
    return "\n\n".join(content)


# --- Core Logic ---
def handle_compaction(db):
    """Handles the compaction of the LanceDB table."""
    print("Starting compaction process...")
    try:
        if TABLE_NAME not in db.table_names():
            return print(f"Table '{TABLE_NAME}' not found. Nothing to compact.")
        table = db.open_table(TABLE_NAME)
        print("Optimizing table (compacting files)... This may take a while.")
        table.optimize()
        print("Optimization complete.")
    except Exception as e:
        print(f"An error occurred during compaction: {e}")


def get_existing_source_hashes(db):
    """Retrieves existing source hashes from the database."""
    source_hashes = {}
    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        if 'source' in table.schema.names and 'hash' in table.schema.names:
            try:
                df = table.to_pandas()
                if not df.empty:
                    source_hashes = df.drop_duplicates('source').set_index('source')['hash'].to_dict()
                print(f"Found {len(source_hashes)} existing sources in the database.")
            except Exception as e:
                print(f"Could not read existing hashes. Error: {e}")
        else:
            print("Warning: Table exists but is missing required columns. Dropping and recreating.")
            db.drop_table(TABLE_NAME)
    return source_hashes

def scan_local_files():
    """Scans the local data folder for files."""
    current_files = {}
    if os.path.exists(DATA_FOLDER):
        print(f"Scanning local folder: {DATA_FOLDER}")
        for file_name in os.listdir(DATA_FOLDER):
            file_path = os.path.join(DATA_FOLDER, file_name)
            if os.path.isfile(file_path):
                current_files[file_path] = {"type": "local", "hash": calculate_file_hash(file_path)}
    return current_files

def scan_google_drive(temp_dir):
    """Scans a Google Drive folder for files."""
    current_files = {}
    if not (GOOGLE_DRIVE_FOLDER_ID and SERVICE_ACCOUNT_PATH and os.path.exists(SERVICE_ACCOUNT_PATH)):
        return current_files

    print(f"DEBUG: Scanning Google Drive folder: {GOOGLE_DRIVE_FOLDER_ID}")
    try:
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH, scopes=["https://www.googleapis.com/auth/drive.readonly"])
        drive_service = build('drive', 'v3', credentials=creds)
        
        results = drive_service.files().list(
            q=f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents",
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        items = results.get('files', [])
        
        print(f"DEBUG: Found {len(items)} files in Google Drive folder.")

        for item in items:
            file_id = item['id']
            file_name = item['name']
            mime_type = item['mimeType']
            source_id = f"https://docs.google.com/document/d/{file_id}"
            doc = None

            if mime_type == 'application/vnd.google-apps.document':
                try:
                    request = drive_service.files().export_media(fileId=file_id, mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
                    temp_docx_path = os.path.join(temp_dir, f"temp_{file_id}.docx")
                    with open(temp_docx_path, 'wb') as f:
                        f.write(request.execute())
                    page_content = parse_docx(temp_docx_path)
                    doc = Document(page_content=page_content, metadata={'source': source_id, 'title': file_name})
                except Exception as e:
                    print(f" Could not process Google Doc {file_name}: {e}")

            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                try:
                    source_id = f"https://docs.google.com/spreadsheets/d/{file_id}"
                    request = drive_service.files().export_media(fileId=file_id, mimeType='text/csv')
                    csv_content = request.execute().decode('utf-8')
                    df = pd.read_csv(StringIO(csv_content))
                    page_content = df.to_markdown(index=False)
                    doc = Document(page_content=page_content, metadata={'source': source_id, 'title': file_name})
                except Exception as e:
                    print(f" Could not process Google Sheet {file_name}: {e}")
            
            if doc:
                content_hash = calculate_content_hash(doc.page_content)
                print(f"DEBUG: Google Doc '{file_name}' has hash: {content_hash}")
                current_files[source_id] = {"type": "drive", "hash": content_hash, "doc": doc}

    except Exception as e:
        print(f"Error scanning Google Drive: {e}")
    
    return current_files

def delete_stale_data(db, files_to_delete, files_to_update):
    """Deletes stale data from the database."""
    if TABLE_NAME in db.table_names() and (files_to_delete or files_to_update):
        table = db.open_table(TABLE_NAME)
        to_delete_from_db = list(files_to_delete) + list(files_to_update.keys())
        if to_delete_from_db:
            delete_query = "','".join(to_delete_from_db)
            print(f"DEBUG: Deleting records for {len(to_delete_from_db)} changed/deleted sources: {to_delete_from_db}")
            table.delete(f"source IN ('{delete_query}')")

def process_files(files_for_processing, api_key):
    """Loads and processes files to be ingested."""
    docs_to_process = []
    print(f"DEBUG: Processing {len(files_for_processing)} files.")
    for source, info in files_for_processing.items():
        print(f"DEBUG: Processing source: {source}")
        if info["type"] == "local":
            try:
                file_path = source
                file_name = os.path.basename(file_path)
                loaded_docs = []
                
                unstructured_extensions = [".pdf", ".doc", ".txt", ".md"]
                image_extensions = [".png", ".jpg", ".jpeg"]
                file_ext = os.path.splitext(file_name)[1].lower()

                if file_ext == ".docx":
                    try:
                        page_content = parse_docx(file_path)
                        loaded_docs.append(Document(page_content=page_content))
                    except Exception as e:
                        print(f" Error processing DOCX {file_path}: {e}")
                elif file_ext in unstructured_extensions:
                    loader = UnstructuredFileLoader(file_path, mode="single", strategy="fast")
                    loaded_docs = loader.load()
                elif file_ext == ".csv":
                    try:
                        df = pd.read_csv(file_path)
                        markdown_table = df.to_markdown(index=False)
                        loaded_docs.append(Document(page_content=markdown_table))
                    except Exception as e:
                        print(f" Error processing CSV {file_path}: {e}")
                elif file_ext in image_extensions:
                    with open(file_path, "rb") as image_file:
                        image_data = image_file.read()
                    description = describe_image(image_data, api_key)
                    if description:
                        loaded_docs.append(Document(page_content=description))
                else: 
                    print(f"Skipping unsupported local file: {file_name}")
                
                for doc in loaded_docs:
                    doc.metadata.update({'hash': info['hash'], 'source': source})
                docs_to_process.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading {source}: {e}")
        elif info["type"] == "drive":
            doc = info["doc"]
            doc.metadata.update({'hash': info['hash'], 'source': source})
            docs_to_process.append(doc)
    return docs_to_process

def ingest_data(db, docs_to_process, embeddings):
    """Splits and ingests documents into the database."""
    if not docs_to_process:
        return print("DEBUG: No documents were successfully loaded for ingestion.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP
    )
    splits = splitter.split_documents(docs_to_process)
    print(f"DEBUG: Split into {len(splits)} chunks.")

    if TABLE_NAME not in db.table_names():
        print("DEBUG: Creating new table with explicit schema.")
        try:
            embedding_dim = len(embeddings.embed_query("test"))
        except Exception:
            embedding_dim = 1024  # Fallback for bge-large-en-v1.5
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), list_size=embedding_dim)),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("hash", pa.string())
        ])
        db.create_table(TABLE_NAME, schema=schema)

    table = db.open_table(TABLE_NAME)
    
    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i:i + BATCH_SIZE]
        data_to_add = []
        for doc in batch:
            data_to_add.append({
                "vector": embeddings.embed_query(doc.page_content),
                "text": doc.page_content,
                "source": doc.metadata['source'],
                "hash": doc.metadata['hash']
            })
        if data_to_add:
            table.add(data_to_add)
            print(f"DEBUG: Successfully indexed batch of {len(data_to_add)} chunks.")

def main():
    """Main function to run the ingestion process."""
    # --- Cleanup old temp files ---
    if os.path.exists(DATA_FOLDER):
        for file_name in os.listdir(DATA_FOLDER):
            if file_name.startswith("temp_") and file_name.endswith(".docx"):
                try:
                    os.remove(os.path.join(DATA_FOLDER, file_name))
                    print(f"Removed orphaned temp file: {file_name}")
                except Exception as e:
                    print(f"Could not remove orphaned temp file {file_name}: {e}")

    db = lancedb.connect(LANCEDB_FOLDER)

    # --- Compaction ---
    if "--compact" in sys.argv:
        handle_compaction(db)
        return

    # --- Ingestion ---
    print("DEBUG: Starting smart ingestion process...")
    
    source_hashes = get_existing_source_hashes(db)
    print(f"DEBUG: Existing source hashes: {source_hashes}")
    
    temp_dir = tempfile.mkdtemp()
    try:
        local_files = scan_local_files()
        drive_files = scan_google_drive(temp_dir)
        current_files = {**local_files, **drive_files}
        print(f"DEBUG: Current files found: {current_files.keys()}")

        files_to_add = {k: v for k, v in current_files.items() if k not in source_hashes}
        files_to_update = {k: v for k, v in current_files.items() if k in source_hashes and source_hashes[k] != v["hash"]}
        files_to_delete = set(source_hashes.keys()) - set(current_files.keys())
        
        print(f"DEBUG: Files to add: {files_to_add.keys()}")
        print(f"DEBUG: Files to update: {files_to_update.keys()}")
        print(f"DEBUG: Files to delete: {files_to_delete}")

        delete_stale_data(db, files_to_delete, files_to_update)

        files_for_processing = {**files_to_add, **files_to_update}
        if not files_for_processing:
            return print("DEBUG: No new or updated documents to process. Ingestion complete.")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        docs_to_process = process_files(files_for_processing, api_key)
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        ingest_data(db, docs_to_process, embeddings)

        print("DEBUG: Ingestion process finished.")
        print("DEBUG: Creating LanceDB index...")
        try:
            table = db.open_table(TABLE_NAME)
            # This will create an index if one does not exist.
            # The parameters are chosen to be low to work with small datasets.
            table.create_index(num_partitions=1, num_sub_vectors=2, replace=True)
            print("DEBUG: LanceDB index created successfully.")
        except Exception as e:
            print(f"DEBUG: Error creating LanceDB index: {e}")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
