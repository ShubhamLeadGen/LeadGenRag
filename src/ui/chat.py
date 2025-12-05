import os
import streamlit as st
import concurrent.futures
import logging
import traceback
from src.backend.caching import save_cache
from src.backend.agentic_rag import decompose_query, synthesize_answer
from src.backend.context_storage import add_to_context_storage, find_similar_response
from src.backend.remember_storage import add_to_remember_user_data_storage
from src.utils import beautify_response
from src.backend.qa_chain import build_qa_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _handle_remember_command(prompt):
    """
    Handles the 'remember this' command, extracting the content to be remembered
    and adding it to the user data storage.
    """
    # Using a more robust check for the command
    if prompt.strip().lower().startswith("remember this"):
        content_to_remember = prompt.strip()[len("remember this"):
        ].strip()
        if content_to_remember.startswith(':'):
            content_to_remember = content_to_remember[1:].strip()

        if content_to_remember:
            add_to_remember_user_data_storage(content_to_remember, content_to_remember)
            st.toast("Your response was remembered, thank you!", icon="👍")
            return "I will remember that."
        else:
            return "What should I remember? Please provide the information after 'remember this:'."
    return None

def _process_rag_query(prompt, chat_container):
    """
    Processes a user query through the RAG pipeline, including query decomposition,
    parallel execution of sub-questions, and synthesis of the final answer.
    The QA chain now uses a MergerRetriever to search all data sources at once.
    """
    try:
        # Build the QA chain on each query to ensure settings are respected.
        # The build_qa_chain function is cached, so it will only rebuild
        # if the verbosity or strict_mode settings have changed.
        try:
            chain, model = build_qa_chain(
                verbosity=st.session_state.get("verbosity", "Normal"),
                strict_mode=st.session_state.get("strict_mode", False)
            )
            st.session_state["qa_chain"] = chain
            st.session_state["llm"] = model
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            st.error(f"Could not build the QA chain: {e}")
            return f"Error initializing models: {e}"
        
        chain = st.session_state["qa_chain"]

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking... (searching all sources)"):
                    sub_questions = decompose_query(prompt)
                    logging.info(f"Decomposed sub-questions: {sub_questions}")

                    if len(sub_questions) > 1:
                        intermediate_answers = ["" for _ in sub_questions]
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future_to_index = {executor.submit(chain.invoke, {"query": q}): i for i, q in
                                               enumerate(sub_questions)}
                            for future in concurrent.futures.as_completed(future_to_index):
                                index = future_to_index[future]
                                try:
                                    result = future.result()
                                    logging.info(f"Retrieved documents for sub-question '{sub_questions[index]}': {result['source_documents']}")
                                    if result['source_documents']:
                                        intermediate_answers[index] = result["result"]
                                    else:
                                        intermediate_answers[index] = "I don't have any information about that in my knowledge base."
                                except Exception as exc:
                                    tb_str = traceback.format_exc()
                                    logging.error(f"Sub-question at index {index} generated an exception:\n{tb_str}")
                                    intermediate_answers[
                                        index] = f"An error occurred during sub-question processing: {tb_str}"
                        final_answer = synthesize_answer(prompt, intermediate_answers)
                    elif sub_questions:
                        result = chain.invoke({"query": sub_questions[0]})
                        logging.info(f"Retrieved documents: {result['source_documents']}")
                        if result['source_documents']:
                            final_answer = result["result"]
                        else:
                            final_answer = "I don't have any information about that in my knowledge base."
                    else:
                        final_answer = "I'm not sure how to answer that. Could you rephrase?"

                    response = beautify_response(final_answer)
    except Exception as e:
        tb_str = traceback.format_exc()
        logging.error(f"An error occurred during RAG query processing: {e}\n{tb_str}")
        response = "Sorry, I encountered an error while processing your request. Please check the logs for details."

    return response

def chat_interface():
    st.title("CAPX")

    messages = st.session_state.sessions.get(st.session_state.active_session_id, [])

    st.subheader(f"Session: {st.session_state.active_session_id} ({len(messages)} messages)")
    chat_container = st.container()
    for idx, msg in enumerate(messages):
        with chat_container:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                    
                if msg["role"] == "assistant":
                    # Create a unique key for each like button
                    like_button_key = f"like_button_{st.session_state.active_session_id}_{idx}"
                    
                    if msg.get("liked", False):
                        st.button("❤️ Liked", key=like_button_key, disabled=True)
                    else:
                        if st.button("👍 Like", key=like_button_key):
                            if find_similar_response(msg["query"]):
                                st.toast("A similar response has already been liked.", icon="❤️")
                            else:
                                add_to_context_storage(msg["query"], msg["content"])
                                st.toast("Response liked and saved to context!", icon="👍")
                            
                            st.session_state.sessions[st.session_state.active_session_id][idx]["liked"] = True
                            save_cache(st.session_state.sessions)
                            st.rerun()

    if prompt := st.chat_input("Ask me something..."):
        logging.info(f"Appending user message: {prompt}")
        st.session_state.sessions[st.session_state.active_session_id].append(
            {"role": "user", "content": prompt}
        )
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # 1. Check for "remember this" command first.
        response = _handle_remember_command(prompt)
        if response:
            st.session_state.sessions[st.session_state.active_session_id].append(
                {"role": "assistant", "content": response, "query": prompt}
            )
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
        else:
            # 2. If not a command, process as a normal RAG query.
            # The new QA chain now handles searching all data sources.
            response = _process_rag_query(prompt, chat_container)
            logging.info(f"Appending QA chain assistant response: {response}")
            st.session_state.sessions[st.session_state.active_session_id].append(
                {"role": "assistant", "content": response, "query": prompt}
            )
        
        save_cache(st.session_state.sessions)
        # Rerun to display the new message and update button states