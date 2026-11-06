"""
embedder.py — Embedding & Vector Store Ingestion
=================================================
Responsibility: Take document chunks, convert them to vector embeddings
using OpenAI, and store them in a local ChromaDB collection.

WHY EMBEDDINGS?
An embedding is a list of numbers (a vector) that captures the *meaning*
of a piece of text. Texts with similar meaning have vectors that are close
together in space. This lets us find relevant chunks for a question without
needing exact keyword matches.

WHY CHROMADB (LOCAL)?
During development, we want zero infrastructure. ChromaDB can run entirely
in-process and persist data to disk — no Docker, no cloud account needed.
When you go to production you can swap it for Pinecone, Weaviate, etc.
"""

import hashlib
import logging
import threading
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = "data/chroma_db"   # where ChromaDB saves its files
COLLECTION_NAME = "insurance_policies"  # logical name for this set of documents


# lru_cache does NOT stop several threads from each running the function on a cold
# cache. The first call can come from many threads at once (an agent issuing
# parallel searches; the coverage and exclusions agents running side by side), and
# ChromaDB's client setup is not safe to run concurrently: a live run crashed with
# "RustBindingsAPI object has no attribute 'bindings'" / "Could not connect to
# tenant default_tenant". The lock serialises construction; once built, taking it
# costs microseconds and never covers the search itself.
_build_lock = threading.Lock()


@lru_cache(maxsize=1)
def _build_vector_store() -> Chroma:
    """Construct the store. Call through get_vector_store(), never directly."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # cheap, fast, good quality
        # OpenAI API key is read from os.environ["OPENAI_API_KEY"] automatically
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_vector_store() -> Chroma:
    """
    Load (or create) the ChromaDB vector store backed by OpenAI embeddings.

    Cached and thread-safe: the embeddings client and Chroma handle are built once,
    on the first call (lazy, so importing this module needs no OPENAI_API_KEY), and
    reused. Concurrent first calls wait for the one build instead of racing.

    Returns:
        A LangChain Chroma object you can call .add_documents() or
        .as_retriever() on.

    Note:
        OPENAI_API_KEY must be set in your environment (via .env + python-dotenv).
    """
    with _build_lock:
        return _build_vector_store()


get_vector_store.cache_clear = _build_vector_store.cache_clear  # type: ignore[attr-defined]


def _chunk_id(source: str, index: int) -> str:
    """Deterministic ID from filename + chunk index, so re-ingests upsert."""
    return hashlib.sha256(f"{source}:{index}".encode()).hexdigest()


def ingest_documents(chunks: list[Document]) -> int:
    """
    Embed and store document chunks in ChromaDB.

    Idempotent: chunk IDs are a hash of (source filename, chunk index), and any
    existing chunks for the same source are deleted first, so re-ingesting a
    PDF replaces it instead of duplicating it (even if the new version has
    fewer chunks).

    Args:
        chunks: List of Document objects (output of loader.load_pdf).

    Returns:
        Number of chunks successfully added.
    """
    if not chunks:
        logger.warning("No chunks provided — nothing to ingest.")
        return 0

    logger.info("Ingesting %d chunks into ChromaDB...", len(chunks))
    vector_store = get_vector_store()

    ids = [
        _chunk_id(chunk.metadata.get("source", "unknown"), i)
        for i, chunk in enumerate(chunks)
    ]
    for source in {chunk.metadata.get("source", "unknown") for chunk in chunks}:
        vector_store.delete(where={"source": source})
    vector_store.add_documents(chunks, ids=ids)
    logger.info("✅ Ingested %d chunks into collection '%s'", len(chunks), COLLECTION_NAME)
    return len(chunks)
