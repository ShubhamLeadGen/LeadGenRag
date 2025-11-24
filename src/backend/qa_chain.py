import os
import streamlit as st
import time
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.config import LLM_MODEL, RETRIEVER_K
from src.backend.vectorstore import load_vectorstore
from src.backend.agentic_rag import set_llm

@st.cache_resource
def build_qa_chain(verbosity="Normal"):
    start_total = time.perf_counter()
    print("[timing] build_qa_chain: starting QA chain build")

    t0 = time.perf_counter()
    vs = load_vectorstore()
    t1 = time.perf_counter()
    print(f"[timing] build_qa_chain: load_vectorstore took {(t1-t0):.2f}s")

    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVER_K})

    prompts = {
        "Concise": "You are a helpful assistant. Your goal is to provide concise and accurate answers. Keep your answers very brief, ideally 1-2 sentences.",
        "Normal": "You are a helpful assistant. Your goal is to provide concise and accurate answers. Tailor the length of your response to the user's question. For simple questions, provide a brief, 2-4 sentence answer. For more complex questions that require a brief explanation, provide a 4-6 sentence summary. Avoid lengthy responses unless the user specifically asks for a detailed explanation.",
        "Detailed": "You are a helpful assistant. Your goal is to provide comprehensive and detailed answers. Use the provided context thoroughly. Explain your reasoning and provide as much detail as possible."
    }
    system_prompt = prompts.get(verbosity, prompts["Normal"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    t2 = time.perf_counter()
    llm = ChatOpenAI(
        model="anthropic/claude-3-5-haiku-20241022",
        temperature=0,
        api_key=os.getenv("FASTROUTER_API_KEY"),
        openai_api_key=None,
        base_url=os.getenv("FASTROUTER_BASE_URL")
    )
    t3 = time.perf_counter()
    print(f"[timing] build_qa_chain: ChatOpenAI init took {(t3-t2):.2f}s")

    set_llm(llm)  # Set the LLM for the agentic chains

    t4 = time.perf_counter()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    t5 = time.perf_counter()
    print(f"[timing] build_qa_chain: RetrievalQA.from_chain_type took {(t5-t4):.2f}s")
    total = time.perf_counter() - start_total
    print(f"[timing] build_qa_chain: total build time {(total):.2f}s")
    return chain, llm
