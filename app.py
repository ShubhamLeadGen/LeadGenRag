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
        def _find_logo(preferred=("Img", "Gemini_Generated_Image_2d6csh2d6csh2d6c.png")):
            preferred_dir, preferred_name = preferred
            preferred_path = os.path.join(preferred_dir, preferred_name)
            if os.path.exists(preferred_path):
                return preferred_path
            # Try alternate dir casing
            alt_dir = "img" if preferred_dir.lower() == "img" else "Img"
            alt_path = os.path.join(alt_dir, preferred_name)
            if os.path.exists(alt_path):
                return alt_path
            # Otherwise search for any image in Img or img
            for d in ("Img", "img"):
                if os.path.isdir(d):
                    for fname in os.listdir(d):
                        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
                            return os.path.join(d, fname)
            return None

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

    # This component call attempts to load data from the browser via JS.
    # If the JS isn't ready (e.g., first run), we fall back to an empty sessions dict
    # so the UI (chat interface) can render immediately instead of blocking forever.
    try:
        cached_data = load_cache()
    except Exception:
        cached_data = None

    # Check if the session state has been initialized.
    # Merge Local Storage sessions into session_state once, preserving non-empty histories.
    if not st.session_state.get("sessions_auto_merged", False):
        if cached_data and cached_data != "CACHE_EMPTY":
            try:
                parsed = json.loads(cached_data)
                if isinstance(parsed, dict):
                    # Ensure sessions dict exists
                    if 'sessions' not in st.session_state:
                        st.session_state.sessions = {}
                    for k, v in parsed.items():
                        try:
                            inc_len = len(v) if isinstance(v, list) else 0
                        except Exception:
                            inc_len = 0
                        exist = st.session_state.sessions.get(k)
                        try:
                            exist_len = len(exist) if isinstance(exist, list) else 0
                        except Exception:
                            exist_len = 0

                        # If incoming is empty but existing has messages, keep existing
                        if inc_len == 0 and exist_len > 0:
                            continue
                        st.session_state.sessions[k] = v
                    st.session_state.sessions_auto_merged = True
            except Exception:
                st.session_state.sessions_auto_merged = True

    if 'sessions' not in st.session_state:
        # If the JS hasn't populated `cached_data` yet, proceed with an empty sessions dict
        # so the chat UI is visible. Provide a visible button to allow the user to
        # explicitly load saved sessions from browser localStorage when available.
        if cached_data is None:
            st.session_state.sessions = {}
            st.sidebar.info("Saved sessions not loaded from browser. Click 'Load saved sessions' to restore previous chats.")
            if st.sidebar.button("Load saved sessions"):
                # Try to load again; this will execute JS in the browser and return any stored sessions.
                try:
                    new_data = load_cache()
                except Exception:
                    new_data = None

                if new_data and new_data != "CACHE_EMPTY":
                    try:
                        st.session_state.sessions = json.loads(new_data)
                        st.experimental_rerun()
                    except Exception:
                        st.sidebar.error("Failed to parse saved sessions from browser storage.")
                else:
                    st.sidebar.warning("No saved sessions found in browser localStorage.")
        else:
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