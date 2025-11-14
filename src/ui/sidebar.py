import streamlit as st
from datetime import datetime
from src.backend.caching import save_cache

def sidebar():

    st.sidebar.header("Chat Sessions")

    # Verbosity setting
    st.session_state.verbosity = st.sidebar.selectbox(
        "Response Verbosity",
        options=["Concise", "Normal", "Detailed"],
        index=["Concise", "Normal", "Detailed"].index(st.session_state.get("verbosity", "Normal")),
        help="Choose the level of detail for the AI's answers."
    )

    st.sidebar.divider()

    session_names = list(st.session_state.sessions.keys())

    if not session_names:
        st.sidebar.write("No sessions yet.")
        return

    active_session_index = 0
    if st.session_state.active_session_id in session_names:
        active_session_index = session_names.index(st.session_state.active_session_id)

    active_session = st.sidebar.selectbox(
        "Select Session",
        options=session_names,
        index=active_session_index,
        format_func=lambda sid: f"{sid} ({len(st.session_state.sessions.get(sid, []))} msgs)"
    )

    if active_session != st.session_state.active_session_id:
        st.session_state.active_session_id = active_session
        st.rerun()

    if st.sidebar.button("+ New Session"):
        new_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.session_state.active_session_id = new_id
        st.session_state.sessions[new_id] = []
        save_cache(st.session_state.sessions)
        st.rerun()


    if st.sidebar.button("🗑️ Delete Current Session"):
        sid = st.session_state.active_session_id
        if sid in st.session_state.sessions:
            del st.session_state.sessions[sid]
            if st.session_state.sessions:
                st.session_state.active_session_id = sorted(st.session_state.sessions.keys())[0]
            else:
                new_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state.sessions[new_id] = []
                st.session_state.active_session_id = new_id
            save_cache(st.session_state.sessions)
            st.rerun()
