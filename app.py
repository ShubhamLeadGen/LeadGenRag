import os
import sys
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path before modification: {sys.path}")
sys.path.insert(0, 'C:\\RAGHF')
print(f"sys.path after modification: {sys.path}")
import json
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.backend.qa_chain import build_qa_chain
from src.backend.caching import load_cache
from src.ui.sidebar import sidebar
from src.ui.chat import chat_interface
from src.backend.file_watcher import start_file_watcher

def main():
    st.set_page_config(page_title="CAPX", page_icon="Img/Gemini_Generated_Image_2d6csh2d6csh2d6c.png", layout="wide")
    st.markdown('''
        <style>
            .block-container {
                padding-top: 0rem;
            }
            [data-testid="stSidebar"] > div:first-child {
                padding-top: 0rem;
            }
        </style>
        ''',
        unsafe_allow_html=True
    )


    # Start the file watcher in a background thread
    start_file_watcher(st.session_state)

    # Check if data has changed and rerun if necessary
    if st.session_state.get('data_changed', False):
        st.cache_resource.clear()
        st.session_state.data_changed = False
        st.rerun()

    if 'app_initialized' not in st.session_state:
        # CSS for pulsing animation
        pulsing_css = '''
        <style>
        @keyframes pulse {
          0% { transform: scale(0.95); }
          50% { transform: scale(1.05); }
          100% { transform: scale(0.95); }
        }
        div[data-testid="stImage"] > img {
          animation: pulse 2s ease-in-out infinite;
        }
        </style>
        '''
        st.markdown(pulsing_css, unsafe_allow_html=True)

        # Center the image using columns
        image_path = os.path.join("Img", "Gemini_Generated_Image_2d6csh2d6csh2d6c.png")
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.image(image_path, width='stretch')

        # Don't block on heavy initialization (LLM / vectorstore). Defer QA chain
        # building until the first user query to improve initial load time.
        st.session_state.app_initialized = True
        st.rerun()

    # --- Main App Logic ---

    # This component call triggers the JS to load data from the browser.
    # Its return value is stored in st.session_state.load_cache by Streamlit on a subsequent script run.
    load_cache()

    # Check if the session state has been initialized.
    if 'sessions' not in st.session_state:
        # Get the data returned by the component. It will be None on the first run.
        cached_data = st.session_state.get('load_cache')

        # If there's no data yet, show a spinner and wait for the component to return data and trigger a rerun.
        if cached_data is None:
            st.spinner("Loading sessions...")
            return

        # If we have data, process it and initialize the session state.
        try:
            st.session_state.sessions = json.loads(cached_data) if cached_data and cached_data != "CACHE_EMPTY" else {}
        except (json.JSONDecodeError, TypeError):
            st.session_state.sessions = {}

    # Ensure an active session is always selected.
    if "active_session_id" not in st.session_state or st.session_state.active_session_id not in st.session_state.sessions:
        if st.session_state.sessions:
            # On load, select the most recent session. Sorting ensures we get the latest one.
            st.session_state.active_session_id = sorted(st.session_state.sessions.keys())[-1]
        else:
            # If there are no sessions at all, create a new one.
            new_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state.sessions[new_id] = []
            st.session_state.active_session_id = new_id

    # Ensure verbosity is initialized in session state
    if "verbosity" not in st.session_state:
        st.session_state.verbosity = "Normal"

    # Render sidebar first to get verbosity setting
    sidebar()

    # Render the chat interface. The UI will lazily build the QA chain on first use
    # to avoid long blocking times during initial page load.
    chat_interface(None, None)


if __name__ == "__main__":
    main()