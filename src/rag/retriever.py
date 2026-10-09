"""
retriever.py — Semantic Retrieval
==================================
Responsibility: Given a user question, find the most relevant document chunks
from ChromaDB using cosine similarity between the question embedding and the
stored chunk embeddings.

WHY TOP-K?
We don't pass the whole database to the LLM. We retrieve only the k most
relevant chunks (default 4) and pass those as context. This keeps prompts
short and focused.
"""

import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.rag.embedder import get_vector_store

logger = logging.getLogger(__name__)

DEFAULT_K = 4  # number of chunks to retrieve per query


def get_retriever(k: int = DEFAULT_K) -> VectorStoreRetriever:
    """
    Return a LangChain retriever that performs similarity search over ChromaDB.

    Args:
        k: Number of top chunks to retrieve for each query.

    Returns:
        A VectorStoreRetriever you can call .invoke("your question") on.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    logger.info("Retriever ready (k=%d)", k)
    return retriever


def retrieve(query: str, k: int = DEFAULT_K) -> list[Document]:
    """
    Convenience function: retrieve top-k chunks for a query string.

    Args:
        query: The user's question in plain text.
        k: Number of chunks to return.

    Returns:
        List of the most relevant Document chunks.
    """
    retriever = get_retriever(k=k)
    results = retriever.invoke(query)
    logger.info("Retrieved %d chunks for query: %r", len(results), query[:60])
    return results
