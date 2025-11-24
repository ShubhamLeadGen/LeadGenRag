import os
import streamlit as st
import concurrent.futures
from src.backend.caching import save_cache
from src.backend.agentic_rag import decompose_query, synthesize_answer
from src.backend.context_storage import add_to_context_storage, find_similar_response
from src.backend.remember_storage import add_to_remember_user_data_storage, find_in_remember_user_data_storage
from src.utils import beautify_response

def chat_interface(qa_chain, llm):
    st.title("Saarthi")

    messages = st.session_state.sessions.get(st.session_state.active_session_id, [])

    st.subheader(f"Session: {st.session_state.active_session_id} ({len(messages)} messages)")
    chat_container = st.container()
    for idx, msg in enumerate(messages):
        with chat_container:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and msg.get("from_context_storage"):
                    st.markdown(f'{msg["content"]} *(response gained from context)*')
                else:
                    st.markdown(msg["content"])
                    
                if msg["role"] == "assistant" and not msg.get("from_context_storage", False):
                    # Create a unique key for each like button
                    like_button_key = f"like_button_{st.session_state.active_session_id}_{idx}"
                    
                    # Check if the message has been liked already
                    if msg.get("liked", False):
                        st.button("❤️ Liked", key=like_button_key, disabled=True)
                    else:
                        if st.button("👍 Like", key=like_button_key):
                            # Check if a similar response already exists
                            if find_similar_response(msg["query"]):
                                st.toast("Heart’s taken", icon="❤️")
                            else:
                                add_to_context_storage(msg["query"], msg["content"])
                                st.toast("Response liked and saved to context!", icon="👍")
                            
                            # Mark the message as liked and rerun to update the button
                            st.session_state.sessions[st.session_state.active_session_id][idx]["liked"] = True
                            st.rerun()

    if prompt := st.chat_input("Ask me something..."):
        print(f"DEBUG: Appending user message: {prompt}")
        st.session_state.sessions[st.session_state.active_session_id].append(
            {"role": "user", "content": prompt}
        )
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Handle "remember this" command
        processed_prompt = "".join(e for e in prompt.lower() if e.isalnum())
        if processed_prompt.startswith("rememberthis") or processed_prompt.startswith("remberthis") or processed_prompt.startswith("Remember this"):
            # Find the start of the actual content
            try:
                start_index = prompt.lower().find("remember this")
                if start_index == -1:
                    start_index = prompt.lower().find("rember this")

                # Adjust start_index to be after the trigger phrase
                if "remember this" in prompt.lower():
                    start_index += len("remember this")
                else:
                    start_index += len("rember this")


                if start_index < len(prompt) and prompt[start_index] == ":":
                    start_index += 1
                
                content_to_remember = prompt[start_index:].strip()


                if content_to_remember:
                    add_to_remember_user_data_storage(content_to_remember, content_to_remember)
                    st.toast("Your response was remembered, thank you!", icon="👍")
                    
                    response = "I will remember that."
                else:
                    response = "What should I remember? Please provide the information after the trigger phrase."

            except Exception as e:
                response = f"An error occurred: {e}"

            st.session_state.sessions[st.session_state.active_session_id].append(
                {"role": "assistant", "content": response, "query": prompt}
            )
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
            save_cache(st.session_state.sessions)


        else:
            # Check remember storage first
            remembered_response = find_in_remember_user_data_storage(prompt)
            if remembered_response:
                with st.spinner("Retrieving remembered information..."):
                    # Use LLM to synthesize a concise answer from the remembered response
                    final_answer = synthesize_answer(prompt, [remembered_response])
                    response = beautify_response(final_answer)
                    print(f"DEBUG: Appending remembered assistant response: {response}")
                    st.session_state.sessions[st.session_state.active_session_id].append(
                        {"role": "assistant", "content": response, "query": prompt, "from_remember_storage": True}
                    )
                    save_cache(st.session_state.sessions)
                st.rerun()
            else:
                # Check context storage
                with st.spinner("Checking context storage..."):
                    stored_response = find_similar_response(prompt)
                if stored_response:
                    response = stored_response
                    print(f"DEBUG: Appending stored assistant response: {response}")
                    st.session_state.sessions[st.session_state.active_session_id].append(
                        {"role": "assistant", "content": response, "query": prompt, "from_context_storage": True}
                    )
                    save_cache(st.session_state.sessions)
                    st.rerun()
                else:
                    # Ensure QA chain is available; build lazily if not provided.
                    chain = qa_chain
                    model = llm
                    if chain is None:
                        with st.spinner("Warming up models and vectorstore (first use)..."):
                            from src.backend.qa_chain import build_qa_chain

                            try:
                                chain, model = build_qa_chain(st.session_state.get("verbosity", "Normal"))
                                # Cache in session for this Streamlit session to avoid rebuilding
                                st.session_state["qa_chain"] = chain
                                st.session_state["llm"] = model
                            except Exception as e:
                                response = f"Error initializing models: {e}"
                                st.session_state.sessions[st.session_state.active_session_id].append(
                                    {"role": "assistant", "content": response, "query": prompt}
                                )
                                save_cache(st.session_state.sessions)
                                st.rerun()

                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                sub_questions = decompose_query(prompt)

                                if len(sub_questions) > 1:
                                    intermediate_answers = ["" for _ in sub_questions]
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future_to_index = {executor.submit(chain.invoke, {"query": q}): i for i, q in enumerate(sub_questions)}
                                        for future in concurrent.futures.as_completed(future_to_index):
                                            index = future_to_index[future]
                                            try:
                                                result = future.result()
                                                intermediate_answers[index] = result["result"]
                                            except Exception as exc:
                                                print(f"Sub-question at index {index} generated an exception: {exc}")
                                                intermediate_answers[index] = "Error processing sub-question."

                                    final_answer = synthesize_answer(prompt, intermediate_answers)
                                elif sub_questions:
                                    result = chain.invoke({"query": sub_questions[0]})
                                    final_answer = result["result"]
                                else:
                                    final_answer = "I'm not sure how to answer that. Could you rephrase?"

                                response = beautify_response(final_answer)
                print(f"DEBUG: Appending QA chain assistant response: {response}")
                st.session_state.sessions[st.session_state.active_session_id].append(
                    {"role": "assistant", "content": response, "query": prompt}
                )
                save_cache(st.session_state.sessions)
