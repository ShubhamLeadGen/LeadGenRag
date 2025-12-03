import json
from datetime import datetime
import streamlit as st
from src.backend.caching import load_cache

def initialize_session():
    """
    Initializes the Streamlit session state, loading and merging chat sessions
    from the browser's local storage.
    """
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
