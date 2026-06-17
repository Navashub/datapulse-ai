"""
embeddings.py — Text embeddings using sentence-transformers.

We switched from Ollama to sentence-transformers for deployment because:
  - Ollama requires 4GB+ RAM — too heavy for cloud free tiers
  - sentence-transformers runs in ~200MB RAM — perfect for Render
  - Same quality embeddings, fully offline, no API key needed

The model we use (all-MiniLM-L6-v2) produces 384-dimensional vectors.
NOTE: This changes the vector dimension from 768 (nomic-embed-text)
to 384. If you have existing data, clear your LanceDB folder and
re-ingest after this change.
"""

import logging
from functools import lru_cache
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it.

    lru_cache means the model is loaded on first call and reused
    for every subsequent call — no reloading on each request.
    Loading takes ~2 seconds the first time.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """
    Convert a single string into a vector embedding.

    Args:
        text: Text to embed.

    Returns:
        List of 384 floats representing the text meaning.

    Raises:
        ValueError: If text is empty.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks efficiently.

    sentence-transformers supports true batch encoding —
    much faster than calling embed_text() in a loop.

    Args:
        texts: List of text chunks to embed.

    Returns:
        List of embedding vectors in the same order as input.
    """
    if not texts:
        raise ValueError("Cannot embed empty list")

    model = get_embedding_model()
    logger.info(f"Embedding {len(texts)} chunks...")

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    logger.info(f"Embedding complete — {len(embeddings)} vectors created")

    return [e.tolist() for e in embeddings]


def embed_query(question: str) -> list[float]:
    """
    Embed a user query for similarity search.

    Kept separate from embed_text() for clarity in the RAG pipeline.
    """
    logger.debug(f"Embedding query: {question[:60]}...")
    return embed_text(question)