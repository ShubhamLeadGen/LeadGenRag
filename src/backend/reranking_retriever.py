from langchain.retrievers import MergerRetriever
from langchain.schema import BaseRetriever, Document
from typing import List, Any
from sentence_transformers import CrossEncoder
import numpy as np

class ReRankingRetriever(BaseRetriever):
    """
    A retriever that re-ranks the results of a base retriever using a CrossEncoder model.
    """
    base_retriever: MergerRetriever
    cross_encoder_model: Any
    k: int = 4  # The final number of documents to return

    def __init__(self, *, base_retriever: MergerRetriever, k: int, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', **kwargs: Any):
        cross_encoder_model = CrossEncoder(model_name)
        super().__init__(base_retriever=base_retriever, cross_encoder_model=cross_encoder_model, k=k, **kwargs)
        
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves and re-ranks documents for a given query.
        """
        # 1. Get initial documents from the base retriever (MergerRetriever)
        initial_docs = self.base_retriever.get_relevant_documents(query)

        if not initial_docs:
            return []

        # 2. Prepare sentence pairs for the CrossEncoder
        sentence_pairs = [[query, doc.page_content] for doc in initial_docs]

        # 3. Get scores from the CrossEncoder
        scores = self.cross_encoder_model.predict(sentence_pairs)

        # 4. Combine documents with their scores
        scored_docs = list(zip(initial_docs, scores))

        # 5. Sort documents by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 6. Return the top-k documents
        top_k_docs = [doc for doc, score in scored_docs[:self.k]]

        print(f"[info] ReRankingRetriever: Re-ranked {len(initial_docs)} docs to {len(top_k_docs)} for query: '{query[:30]}...'")

        return top_k_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Asynchronous version of get_relevant_documents.
        This is a simplified implementation for demonstration.
        """
        # This is not a true async implementation, but it's required by the interface.
        return self.get_relevant_documents(query)
