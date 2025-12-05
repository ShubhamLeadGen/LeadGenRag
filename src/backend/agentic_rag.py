# Updated Agentic RAG Code with Fixes
# Includes: simplified sub-questions, safe filtering, improved synthesis, threshold note

import os
import re
import time
import logging
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chat_models.fake import FakeListChatModel
import lancedb
from langchain.retrievers import MergerRetriever
from src.config import (
    LANCEDB_FOLDER,
    EMBEDDING_MODEL,
    LLM_MODEL,
    SIMILARITY_THRESHOLD,
    RETRIEVER_K,
    MAX_CHARS_FOR_TEXT_EXTRACTION,
    ALLOW_EXTERNAL_LLMS,
)
from src.utils import is_gibberish, polite_fallback, beautify_response
from langchain.chains import RetrievalQA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# LLM & Chains
# -----------------------------
llm = None
_decompose_chain = None
_synthesize_chain = None
_should_search_chain = None
_llm_initialized = False

# ----------- PROMPTS -----------

def _clean_lines(text):
    # Removes unwanted lines like "For the query"
    return [t for t in text.split("\n") if t.strip() and not t.lower().startswith("for the query")]

_decompose_prompt = ChatPromptTemplate.from_template(
    """You are a query decomposer for marketing and lead generation.
Break the user's query into simple, atomic sub-questions.
ONLY include sub-questions related to: lead generation, marketing, funnels, ads, campaigns, user acquisition.
Do NOT add explanations. Return only hyphenated bullet points.

Query: {query}
Sub-questions:"""
)

_synthesize_prompt = ChatPromptTemplate.from_template(
    """You are an expert synthesizer in lead generation.
Combine ONLY the intermediate answers into one clear final answer.
Do NOT hallucinate.
Do NOT add anything that is NOT inside the provided answers.

Query: {query}
Intermediate Answers:
{intermediate_answers}

Final Answer:"""
)

_should_search_prompt = ChatPromptTemplate.from_template(
    """Should a web search be used to answer this query?
Say only: Yes or No.

Query: {query}
Decision:"""
)

# ----------- INITIALIZATION -----------

# Exportable helper
def set_llm(new_llm):
    global llm, _decompose_chain, _synthesize_chain, _should_search_chain
    llm = new_llm
    _decompose_chain = _decompose_prompt | llm
    _synthesize_chain = _synthesize_prompt | llm
    _should_search_chain = _should_search_prompt | llm

def init_llm_and_chains():
    global llm, _decompose_chain, _synthesize_chain, _should_search_chain, _llm_initialized
    if _llm_initialized:
        return

    logging.info("[timing] agentic_rag: initializing module-level LLM...")
    t0 = time.perf_counter()

    if not ALLOW_EXTERNAL_LLMS:
        llm = FakeListChatModel(responses=["The answer is not in the context."])
    else:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=os.getenv("FASTROUTER_API_KEY"),
            openai_api_key=None,
            base_url=os.getenv("FASTROUTER_BASE_URL")
        )

    logging.info(f"ChatOpenAI init took {(time.perf_counter()-t0):.2f}s")

    _decompose_chain = _decompose_prompt | llm
    _synthesize_chain = _synthesize_prompt | llm
    _should_search_chain = _should_search_prompt | llm
    _llm_initialized = True


def decompose_query(query: str) -> list[str]:
    init_llm_and_chains()

    response = _decompose_chain.invoke({"query": query})
    lines = _clean_lines(response.content)

    sub_questions = [
        line.replace("-", "").strip() for line in lines if line.strip().startswith("-")
    ]

    logging.info(f"Decomposed sub-questions: {sub_questions}")
    return sub_questions


def synthesize_answer(query: str, intermediate_answers: list[str]) -> str:
    logging.info(f"Synthesizing answer for {query}")

    # Remove ONLY truly useless answers
    stop_phrases = [
        "i don't know", "cannot find", "no information", "not in the context",
        "apologize", "sorry"
    ]

    filtered = [a for a in intermediate_answers if not any(p in a.lower() for p in stop_phrases)]

    if not filtered:
        return "I couldn't find a relevant answer in the provided documents."

    response = _synthesize_chain.invoke({"query": query, "intermediate_answers": "\n".join(filtered)})
    return response.content


def should_search_web(query: str) -> bool:
    init_llm_and_chains()
    if not ALLOW_EXTERNAL_LLMS:
        return False
    result = _should_search_chain.invoke({"query": query}).content.strip().lower()
    return result == "yes"


# -----------------------------
# Embeddings & Retriever
# -----------------------------
_embeddings = None
_embeddings_initialized = False


def init_embeddings():
    global _embeddings, _embeddings_initialized
    if _embeddings_initialized:
        return
    t0 = time.perf_counter()
    _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logging.info(f"Embeddings initialized in {(time.perf_counter()-t0):.2f}s")
    _embeddings_initialized = True


def setup_retriever() -> MergerRetriever:
    logging.info(f"Loading LanceDB index from: {LANCEDB_FOLDER}")
    init_embeddings()

    db = lancedb.connect(LANCEDB_FOLDER)

    if "rag_table" not in db.table_names():
        raise ValueError("rag_table not found in LanceDB.")

    primary_vs = LanceDB(connection=db, embedding=_embeddings, table_name="rag_table")

    from src.backend.remember_storage import get_remember_user_data_table
    remember_table = get_remember_user_data_table()
    remember_vs = LanceDB(connection=db, embedding=_embeddings, table_name=remember_table.name)

    r1 = primary_vs.as_retriever(search_kwargs={"k": RETRIEVER_K, "score_threshold": SIMILARITY_THRESHOLD})
    r2 = remember_vs.as_retriever(search_kwargs={"k": RETRIEVER_K, "score_threshold": SIMILARITY_THRESHOLD})

    return MergerRetriever(retrievers=[r1, r2])


# -----------------------------
# MAIN LOOP
# -----------------------------

def main():
    load_dotenv()

    init_llm_and_chains()
    retriever = setup_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={
            "prompt": ChatPromptTemplate.from_template(
                """Use ONLY the context to answer.
If answer is not in context, say: I don't know.

Context:
{context}
Question: {question}
Answer:"""
            )
        },
        return_source_documents=True
    )

    print("\n🚀 Agentic RAG Chat ready! Type your question (or 'exit')\n")

    while True:
        q = input("You: ").strip()

        if q.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if is_gibberish(q):
            print("Bot:", polite_fallback())
            continue

        # ------- MEMORY -------
        if q.lower().startswith("remember this"):
            text = q.replace("remember this", "").strip()
            from src.backend.remember_storage import add_to_remember_user_data_storage
            add_to_remember_user_data_storage(text, text)
            print("Bot: I will remember that.")
            continue

        try:
            subqs = decompose_query(q)
            intermediate = []
            all_docs = []

            for sq in subqs:
                docs = retriever.get_relevant_documents(sq)
                all_docs.extend(docs)

                res = qa_chain.invoke({"query": sq})
                intermediate.append(res["result"])

            if len(intermediate) > 1:
                final = synthesize_answer(q, intermediate)
            else:
                final = intermediate[0]

            print("Bot:", beautify_response(final))

            if all_docs:
                print("Sources:", {d.metadata.get("source", "unknown") for d in all_docs})

        except Exception as e:
            logging.error(f"Error: {e}")
            print("An error occurred. Please check logs.")


if __name__ == "__main__":
    main()
