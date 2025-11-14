import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from src.config import LLM_MODEL, RETRIEVER_K
from src.backend.vectorstore import load_vectorstore
from src.backend.agentic_rag import set_llm

@st.cache_resource
def build_qa_chain(verbosity="Normal"):
    vs = load_vectorstore()
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

    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    set_llm(llm)  # Set the LLM for the agentic chains
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    ), llm
