import time
import os
import sys
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from scripts.ingest import main as run_ingestion
from src.config import DATA_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IngestionTrigger(FileSystemEventHandler):
    def __init__(self, session_state):
        self.session_state = session_state

    def on_any_event(self, event):
        # We only care about events in the data folder
        logging.debug(f"File watcher detected a local file system event: {event.event_type} on {event.src_path}")
        if event.src_path.startswith(os.path.join(os.getcwd(), DATA_FOLDER)):
            logging.info(f"Change detected in monitored folder '{DATA_FOLDER}'. Triggering ingestion.")
            try:
                run_ingestion()
                # Signal to the main app that data has changed
                self.session_state.data_changed = True
                logging.info("Ingestion finished. Flag 'data_changed' set to True.")
            except Exception as e:
                logging.error(f"Error during ingestion: {e}")

def start_file_watcher(session_state):
    if not hasattr(session_state, 'watcher_started'):
        session_state.watcher_started = True
        logging.info("Starting file watcher...")

        event_handler = IngestionTrigger(session_state)
        observer = Observer()
        observer.schedule(event_handler, path=DATA_FOLDER, recursive=True)
        observer.start()
        logging.info(f"File watcher started on directory: {DATA_FOLDER}")

        # Keep the thread alive if needed, but for Streamlit,
        # the main process will keep running.
        # You might want to store the observer object in the session state
        # if you need to stop it gracefully later.
        session_state.observer = observer
