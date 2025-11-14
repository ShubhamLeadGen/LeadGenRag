import os
import re
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
import lancedb
from langchain.retrievers import MergerRetriever
from src.config import LANCEDB_FOLDER, EMBEDDING_MODEL, LLM_MODEL, SIMILARITY_THRESHOLD
from langchain_core.documents import Document
from src.backend.remember_storage import add_to_remember_user_data_storage, get_remember_user_data_table

from src.utils import is_gibberish, polite_fallback, beautify_response, extract_clean_text

# -----------------
# Setup
# -----------------
def setup_llm(api_key: str):
    return ChatAnthropic(
        model=LLM_MODEL,
        temperature=0,
        max_tokens_to_sample=2000,
        anthropic_api_key=api_key
    )

def setup_retriever():
    print(f"Loading LanceDB index from: {LANCEDB_FOLDER}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = lancedb.connect(LANCEDB_FOLDER)

    # Ensure table exists
    try:
        vectorstore = LanceDB(connection=db, embedding=embeddings, table_name="rag_table")
    except AttributeError:
        print("⚠️ LanceDB table missing. Creating new table...")
        vectorstore = LanceDB(connection=db, embedding=embeddings, table_name="rag_table", create_if_missing=True)

    remember_user_data_table = get_remember_user_data_table()
    remember_user_data_vectorstore = LanceDB(connection=db, embedding=embeddings, table_name=remember_user_data_table.name)

    retriever1 = vectorstore.as_retriever(search_kwargs={"k": 5, "score_threshold": SIMILARITY_THRESHOLD})
    retriever2 = remember_user_data_vectorstore.as_retriever(search_kwargs={"k": 5, "score_threshold": SIMILARITY_THRESHOLD})

    return MergerRetriever(retrievers=[retriever1, retriever2])

# -----------------
# Main Chat Loop
# -----------------
def main():
    load_dotenv()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file!")

    llm = setup_llm(anthropic_api_key)
    retriever = setup_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={
            "prompt": ChatPromptTemplate.from_template(
                """You are a helpful assistant. Use the following context to answer the user's question. If the user asks you to create something, like test cases, you can do so based on your understanding of the context.

Context:
{context}

Question: {question}
Answer:"""
            )
        },
        return_source_documents=True
    )

    print("\n🚀 RAG Chat ready! Type your question (or 'exit' to quit).\n")

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
            # Use new recommended method
            result = qa_chain.invoke({"query": question})

            if not result["result"]:
                print("Bot:", polite_fallback(), "\n")
                continue

            print("Bot:", beautify_response(result["result"]), "\n")

            # Optional: show sources
            print("Sources:")
            for doc in result["source_documents"]:
                print(f"- {doc.metadata.get('source', 'unknown')}")

        except Exception as e:
            print(f" Error: {e}\n")

if __name__ == "__main__":
    main()
