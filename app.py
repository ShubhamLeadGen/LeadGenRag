import os
import sys

# Add the project root directory to the Python path.
# This ensures that the `src` module can be found when the app is run,
# for example, by Streamlit.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
from src.ui.session_manager import initialize_session
from src.backend.file_watcher import start_file_watcher

def load_css(file_name):
    """Loads an external CSS file and applies it to the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def _find_logo():
    """Find the logo image file, searching in likely directories."""
    preferred_path = "Img/Gemini_Generated_Image_2d6csh2d6csh2d6c.png"
    if os.path.exists(preferred_path):
        return preferred_path
    
    # Fallback search in case of different casing or other images
    for dir_name in ("Img", "img"):
        if os.path.isdir(dir_name):
            for fname in os.listdir(dir_name):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
                    return os.path.join(dir_name, fname)
    return None

def main():
    st.set_page_config(page_title="CAPX", page_icon=_find_logo(), layout="wide")
    load_css("src/ui/styles.css")

    # Start the file watcher in a background thread
    start_file_watcher(st.session_state)

    # Check if data has changed and rerun if necessary
    if st.session_state.get('data_changed', False):
        st.cache_resource.clear()
        st.session_state.data_changed = False
        st.rerun()

    if 'app_initialized' not in st.session_state:
        # Center the image using columns
        image_path = _find_logo()
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            if image_path and os.path.exists(image_path):
                st.image(image_path, width='stretch')
            else:
                st.markdown("<h3>CAPX</h3>", unsafe_allow_html=True)

        # Don't block on heavy initialization (LLM / vectorstore). Defer QA chain
        # building until the first user query to improve initial load time.
        st.session_state.app_initialized = True
        st.rerun()

    # --- Main App Logic ---
    initialize_session()

    # Ensure verbosity is initialized in session state
    if "verbosity" not in st.session_state:
        st.session_state.verbosity = "Normal"

    # Render sidebar first to get verbosity setting
    sidebar()

    # Render the chat interface. The UI will lazily build the QA chain on first use
    # to avoid long blocking times during initial page load.

    chat_interface()


if __name__ == "__main__":
    main()