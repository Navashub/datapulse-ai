"""
embeddings.py — Converts text into vector embeddings using Ollama.

WHAT IS AN EMBEDDING?
  An embedding is a list of numbers (a vector) that represents the
  MEANING of a piece of text. Texts with similar meanings produce
  vectors that are mathematically close to each other.

  Example:
    "The dog ran fast"   → [0.23, -0.41, 0.87, ...]
    "The puppy sprinted" → [0.25, -0.39, 0.85, ...]  ← very close!
    "The stock market"   → [-0.91, 0.12, -0.33, ...] ← far away

  This is what makes semantic search work — we don't match keywords,
  we match MEANING.

WHY nomic-embed-text?
  It's a dedicated embedding model — trained specifically to produce
  high-quality vectors for search and retrieval tasks.
  Using a chat model (like llama3.2) for embeddings would work but
  produce lower quality results. Right tool for the right job.
"""

import logging
from ollama import Client, ResponseError
from app.config import get_settings

logger = logging.getLogger(__name__)


def get_ollama_client() -> Client:
    """
    Create and return an Ollama client pointed at the configured base URL.

    We create a new client per call rather than a module-level singleton
    so that config changes (e.g. in tests) are always picked up.
    """
    settings = get_settings()
    return Client(host=settings.ollama_base_url)


def embed_text(text: str) -> list[float]:
    """
    Convert a single string into a vector embedding via Ollama.

    Args:
        text: The text to embed. Should be a single chunk (not a full document).

    Returns:
        A list of floats representing the embedding vector.
        The length depends on the model (nomic-embed-text produces 768 dimensions).

    Raises:
        RuntimeError: If Ollama is unreachable or the model is not pulled.
        ValueError:   If the input text is empty.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    settings = get_settings()
    client = get_ollama_client()

    try:
        response = client.embeddings(
            model=settings.ollama_embedding_model,
            prompt=text,
        )
        # response.embedding is a list of floats — the vector representation
        return response["embedding"]

    except ResponseError as e:
        # ResponseError means Ollama responded but reported an error —
        # most likely the model hasn't been pulled yet
        raise RuntimeError(
            f"Ollama embedding failed. "
            f"Make sure you have run: ollama pull {settings.ollama_embedding_model}\n"
            f"Error: {e}"
        ) from e

    except Exception as e:
        # Catch connection errors — Ollama server not running
        raise RuntimeError(
            f"Could not connect to Ollama at {settings.ollama_base_url}. "
            f"Make sure Ollama is running: ollama serve\n"
            f"Error: {e}"
        ) from e


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks, returning a list of vectors.

    We embed one at a time (not in parallel) to avoid overwhelming
    the local Ollama server. For a production system you would
    batch these with asyncio.gather() — a good stretch goal for students.

    Args:
        texts: List of text chunks to embed.

    Returns:
        List of embedding vectors, in the same order as the input texts.

    Raises:
        RuntimeError: If any single embedding fails.
        ValueError:   If the texts list is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts")

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = []

    for i, text in enumerate(texts):
        embedding = embed_text(text)
        embeddings.append(embedding)

        # Log progress every 10 chunks so long ingestions feel responsive
        if (i + 1) % 10 == 0:
            logger.info(f"  Embedded {i + 1}/{len(texts)} chunks")

    logger.info(f"Embedding complete — {len(embeddings)} vectors created")
    return embeddings


def embed_query(question: str) -> list[float]:
    """
    Embed a user's query question for similarity search.

    This is functionally identical to embed_text(), but having a
    separate function makes the RAG pipeline code read more clearly:
      - embed_batch()  → used during ingestion (documents)
      - embed_query()  → used during retrieval (questions)

    Some embedding models support separate instruction prefixes for
    queries vs documents — having this function separate makes that
    upgrade easy to add later.

    Args:
        question: The user's natural language question.

    Returns:
        Embedding vector for the question.
    """
    logger.debug(f"Embedding query: {question[:60]}...")
    return embed_text(question)