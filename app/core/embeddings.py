"""
embeddings.py — Text embeddings using FastEmbed.

FastEmbed is a lightweight embedding library by Qdrant.
It uses ONNX runtime instead of PyTorch — this means:
  - ~50MB RAM usage vs ~1.5GB for sentence-transformers
  - Faster startup (no PyTorch initialization)
  - Same quality embeddings for RAG tasks
  - Works on Render free tier (512MB RAM limit)

Model: BAAI/bge-small-en-v1.5
  - 384 dimensions
  - Excellent retrieval quality
  - Only 130MB download on first run
"""

import logging
from functools import lru_cache
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """
    Load the FastEmbed model once and cache it for reuse.

    lru_cache ensures we only load the model once per server lifetime.
    First call takes ~5 seconds to download and initialise.
    All subsequent calls return the cached model instantly.
    """
    logger.info(f"Loading FastEmbed model: {EMBEDDING_MODEL_NAME}")
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded successfully")
    return model


def embed_text(text: str) -> list[float]:
    """
    Convert a single string into a vector embedding.

    Args:
        text: Text to embed. Should be a single chunk.

    Returns:
        List of 384 floats representing the text meaning.

    Raises:
        ValueError: If text is empty.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    model = get_embedding_model()

    # FastEmbed returns a generator — we take the first (only) result
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks efficiently.

    FastEmbed processes all texts in one optimised ONNX pass —
    much faster than calling embed_text() in a loop.

    Args:
        texts: List of text chunks to embed.

    Returns:
        List of embedding vectors in the same order as input.

    Raises:
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("Cannot embed empty list")

    model = get_embedding_model()
    logger.info(f"Embedding {len(texts)} chunks...")

    embeddings = list(model.embed(texts))
    logger.info(f"Embedding complete — {len(embeddings)} vectors created")

    return [e.tolist() for e in embeddings]


def embed_query(question: str) -> list[float]:
    """
    Embed a user query for similarity search.

    Kept separate from embed_text() for clarity in the RAG pipeline
    and to make it easy to add query-specific prefixes later.

    Args:
        question: The user's natural language question.

    Returns:
        Embedding vector for the question.
    """
    logger.debug(f"Embedding query: {question[:60]}...")
    return embed_text(question)