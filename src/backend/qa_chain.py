import os
import time
import streamlit as st
import lancedb
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.retrievers import MergerRetriever

from src.config import LLM_MODEL, RETRIEVER_K, LANCEDB_FOLDER, EMBEDDING_MODEL, SIMILARITY_THRESHOLD
from src.backend.agentic_rag import set_llm
from src.backend.remember_storage import REMEMBER_USER_DATA_TABLE_NAME
from src.backend.context_storage import CONTEXT_TABLE_NAME

@st.cache_resource
def build_qa_chain(verbosity="Normal"):
    """
    Builds a RetrievalQA chain that sources documents from three different vector
    stores: the main RAG table, the user's "remembered" data, and the user's
    "liked" conversation context.
    """
    start_total = time.perf_counter()
    print("[timing] build_qa_chain: starting QA chain build with MergerRetriever")

    # 1. Initialize Embeddings and DB Connection
    t0 = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = lancedb.connect(LANCEDB_FOLDER)
    table_names = db.table_names()
    t1 = time.perf_counter()
    print(f"[timing] build_qa_chain: Embeddings and DB connection took {(t1-t0):.2f}s")

    retrievers = []

    # 2. Create Retriever for the main `rag_table`
    if "rag_table" in table_names:
        rag_vectorstore = LanceDB(connection=db, embedding=embeddings, table_name="rag_table")
        retrievers.append(rag_vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVER_K, "score_threshold": SIMILARITY_THRESHOLD}
        ))
        print("[info] build_qa_chain: Added `rag_table` to retrievers.")
    else:
        print("[warning] build_qa_chain: Main `rag_table` not found.")

    # 3. Create Retriever for the `rememberUserData` table
    if REMEMBER_USER_DATA_TABLE_NAME in table_names:
        remember_vectorstore = LanceDB(
            connection=db,
            embedding=embeddings,
            table_name=REMEMBER_USER_DATA_TABLE_NAME
        )
        retrievers.append(remember_vectorstore.as_retriever(
            search_kwargs={"k": 3, "score_threshold": SIMILARITY_THRESHOLD}
        ))
        print(f"[info] build_qa_chain: Added `{REMEMBER_USER_DATA_TABLE_NAME}` to retrievers.")
    else:
        print(f"[info] build_qa_chain: Table `{REMEMBER_USER_DATA_TABLE_NAME}` not found, skipping.")

    # 4. Create Retriever for the `rag_context_table`
    if CONTEXT_TABLE_NAME in table_names:
        context_vectorstore = LanceDB(
            connection=db,
            embedding=embeddings,
            table_name=CONTEXT_TABLE_NAME
        )
        retrievers.append(context_vectorstore.as_retriever(
            search_kwargs={"k": 3, "score_threshold": SIMILARITY_THRESHOLD}
        ))
        print(f"[info] build_qa_chain: Added `{CONTEXT_TABLE_NAME}` to retrievers.")
    else:
        print(f"[info] build_qa_chain: Table `{CONTEXT_TABLE_NAME}` not found, skipping.")

    if not retrievers:
        raise ValueError("No vector stores found! Cannot build QA chain. Please run ingestion and use the app.")

    # 5. Create the MergerRetriever
    lotr = MergerRetriever(retrievers=retrievers)
    print(f"[info] build_qa_chain: MergerRetriever created with {len(retrievers)} retrievers.")

    # 6. Create the LLM and Prompt
    prompts = {
        "Concise": "You are a helpful assistant. Your goal is to provide concise and accurate answers. Keep your answers very brief, ideally 1-2 sentences. Use the provided context to answer.",
        "Normal": "You are a helpful assistant. Your goal is to provide concise and accurate answers based on the provided context. Tailor the length of your response to the user's question. For simple questions, provide a brief, 2-4 sentence answer. For more complex questions, provide a 4-6 sentence summary.",
        "Detailed": "You are a helpful assistant. Your goal is to provide comprehensive and detailed answers. Use the provided context thoroughly. Explain your reasoning and provide as much detail as possible."
    }
    system_prompt = prompts.get(verbosity, prompts["Normal"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context from multiple sources:\n{context}\n\nQuestion:\n{question}")
    ])

    t2 = time.perf_counter()
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("FASTROUTER_API_KEY"),
        openai_api_key=None,
        base_url=os.getenv("FASTROUTER_BASE_URL")
    )
    t3 = time.perf_counter()
    print(f"[timing] build_qa_chain: ChatOpenAI init took {(t3-t2):.2f}s")

    set_llm(llm)  # Set the LLM for the agentic chains

    # 7. Create the final RetrievalQA chain
    t4 = time.perf_counter()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=lotr, # Use the MergerRetriever here
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    t5 = time.perf_counter()
    print(f"[timing] build_qa_chain: RetrievalQA.from_chain_type took {(t5-t4):.2f}s")
    total = time.perf_counter() - start_total
    print(f"[timing] build_qa_chain: total build time {(total):.2f}s")
    return chain, llm
