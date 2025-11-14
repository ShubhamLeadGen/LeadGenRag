import time
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.ingest import main as run_ingestion
from src.config import DATA_FOLDER

class IngestionTrigger(FileSystemEventHandler):
    def __init__(self, session_state):
        self.session_state = session_state

    def on_any_event(self, event):
        # We only care about events in the data folder
        print(f"DEBUG: File watcher detected a local file system event: {event.event_type} on {event.src_path}")
        if event.src_path.startswith(os.path.join(os.getcwd(), DATA_FOLDER)):
            print(f"DEBUG: Change detected in monitored folder '{DATA_FOLDER}'. Triggering ingestion.")
            try:
                run_ingestion()
                # Signal to the main app that data has changed
                self.session_state.data_changed = True
                print("DEBUG: Ingestion finished. Flag 'data_changed' set to True.")
            except Exception as e:
                print(f"Error during ingestion: {e}")

def start_file_watcher(session_state):
    if not hasattr(session_state, 'watcher_started'):
        session_state.watcher_started = True
        print("Starting file watcher...")

        event_handler = IngestionTrigger(session_state)
        observer = Observer()
        observer.schedule(event_handler, path=DATA_FOLDER, recursive=True)
        observer.start()
        print(f"File watcher started on directory: {DATA_FOLDER}")

        # Keep the thread alive if needed, but for Streamlit,
        # the main process will keep running.
        # You might want to store the observer object in the session state
        # if you need to stop it gracefully later.
        session_state.observer = observer
