import os
import re
import time
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
import lancedb
from langchain.retrievers import MergerRetriever
from src.config import (
    LANCEDB_FOLDER,
    EMBEDDING_MODEL,
    LLM_MODEL,
    SIMILARITY_THRESHOLD,
    RETRIEVER_K,
    MAX_CHARS_FOR_TEXT_EXTRACTION,
)
from src.utils import is_gibberish, polite_fallback, beautify_response, extract_clean_text
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from src.backend.remember_storage import add_to_remember_user_data_storage, get_remember_user_data_table

# -----------------
# LLM and Chains (initialized once)
# -----------------
print("[timing] agentic_rag: initializing module-level LLM...")
t0 = time.perf_counter()
llm = ChatOpenAI(
    model="anthropic/claude-3-5-haiku-20241022",
    temperature=0,
    api_key=os.getenv("FASTROUTER_API_KEY"),
    openai_api_key=None,
    base_url=os.getenv("FASTROUTER_BASE_URL")
)
t1 = time.perf_counter()
print(f"[timing] agentic_rag: module-level ChatOpenAI init took {(t1-t0):.2f}s")

# Decomposition Chain
_decompose_prompt = ChatPromptTemplate.from_template(
    """You are a query analyzer. Your task is to decompose a complex user query into a series of simpler, answerable sub-questions. If the query is simple, return it as a single sub-question.

    Query: {query}
    Sub-questions:"""
)
_decompose_chain = _decompose_prompt | llm

# Synthesis Chain
_synthesize_prompt = ChatPromptTemplate.from_template(
    """You are an expert synthesizer. You have been provided with a user's query and a set of intermediate answers. Your task is to synthesize these into a single, coherent, and concise final answer. Do not repeat the intermediate answers verbatim. Provide a direct and brief answer to the original query.

    Query: {query}
    Intermediate Answers:
    {intermediate_answers}

    Final Answer:"""
)
_synthesize_chain = _synthesize_prompt | llm

def set_llm(new_llm):
    global llm, _decompose_chain, _synthesize_chain
    llm = new_llm
    _decompose_chain = _decompose_prompt | llm
    _synthesize_chain = _synthesize_prompt | llm

from src.utils import is_gibberish, polite_fallback, beautify_response, extract_clean_text


# -----------------
# Agentic Logic
# -----------------
def decompose_query(query: str) -> list[str]:
    response = _decompose_chain.invoke({"query": query})
    sub_questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return sub_questions

def synthesize_answer(query: str, intermediate_answers: list[str]) -> str:
    response = _synthesize_chain.invoke({"query": query, "intermediate_answers": "\n".join(intermediate_answers)})
    return response.content


# -----------------
# Setup
# -----------------



def setup_retriever():
    print(f"Loading LanceDB index from: {LANCEDB_FOLDER}")
    t0 = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    t1 = time.perf_counter()
    print(f"[timing] setup_retriever: HuggingFaceEmbeddings init took {(t1-t0):.2f}s")

    t2 = time.perf_counter()
    db = lancedb.connect(LANCEDB_FOLDER)
    t3 = time.perf_counter()
    print(f"[timing] setup_retriever: lancedb.connect took {(t3-t2):.2f}s")

    table_names = db.table_names()
    if "rag_table" not in table_names:
        raise ValueError(f"LanceDB table 'rag_table' not found in {LANCEDB_FOLDER}. Please ingest data first.")

    t4 = time.perf_counter()
    vectorstore = LanceDB(connection=db, embedding=embeddings, table_name="rag_table")
    t5 = time.perf_counter()
    print(f"[timing] setup_retriever: LanceDB table open/create took {(t5-t4):.2f}s")
    remember_user_data_table = get_remember_user_data_table()
    remember_user_data_vectorstore = LanceDB(connection=db, embedding=embeddings, table_name=remember_user_data_table.name)

    retriever1 = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K, "score_threshold": SIMILARITY_THRESHOLD})
    retriever2 = remember_user_data_vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K, "score_threshold": SIMILARITY_THRESHOLD})

    return MergerRetriever(retrievers=[retriever1, retriever2])

# -----------------
# Main Chat Loop
# -----------------
def main():
    load_dotenv()


    
    retriever = setup_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={
            "prompt": ChatPromptTemplate.from_template(
                """You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {question}
Answer:"""
            )
        },
        return_source_documents=True
    )

    print("\n🚀 Agentic RAG Chat ready! Type your question (or 'exit' to quit).\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if question.lower().startswith("remember this"):
            remember_data = question[len("remember this:"):].strip()
            add_to_remember_user_data_storage(remember_data, remember_data)
            print("Bot: I will remember that.")
            continue
        if not question.strip():
            continue
        if is_gibberish(question):
            print("Bot:", polite_fallback(), "\n")
            continue

        try:
            sub_questions = decompose_query(question)
            print(f"--- Decomposed Sub-questions: {sub_questions} ---")
            intermediate_answers = []
            all_source_documents = []

            for sub_question in sub_questions:
                result = qa_chain.invoke({"query": sub_question})
                intermediate_answers.append(result["result"])
                all_source_documents.extend(result["source_documents"])

                print(f"--- Retrieved Documents for '{sub_question}' ---")
                for doc in result["source_documents"]:
                    print(f"Content: {doc.page_content[:100]}...") # Print first 100 chars
                    print(f"Metadata: {doc.metadata}")
                print("-------------------------------------------------")

            if len(intermediate_answers) > 1:
                final_answer = synthesize_answer(question, intermediate_answers)
            else:
                final_answer = intermediate_answers[0]

            if not final_answer:
                print("Bot:", polite_fallback(), "\n")
                continue

            print("Bot:", beautify_response(final_answer), "\n")

            # Optional: show sources
            print("Sources:")
            unique_sources = {doc.metadata.get('source', 'unknown') for doc in all_source_documents}
            for source in unique_sources:
                print(f"- {source}")

        except Exception as e:
            print(f" Error: {e}\n")


if __name__ == "__main__":
    main()