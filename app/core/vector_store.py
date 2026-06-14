"""
vector_store.py — ChromaDB interface for storing and retrieving vectors.

WHAT IS A VECTOR STORE?
  A regular database stores rows and lets you search by exact values
  (WHERE name = 'John'). A vector store stores embeddings and lets you
  search by SIMILARITY — "find me the 4 chunks most similar in meaning
  to this question."

  ChromaDB is our vector store. It runs locally (no Docker, no cloud)
  and persists data to disk — so ingested documents survive restarts.

HOW IT FITS IN THE RAG PIPELINE:
  Ingest:  chunk → embed → store in ChromaDB (with metadata)
  Query:   embed question → search ChromaDB → get top-K chunks back
                                                        ↓
                                             send to LLM as context
"""

import logging
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Create a persistent ChromaDB client.

    PersistentClient saves vectors to disk at chroma_persist_directory.
    This means ingested documents survive server restarts — critical
    for a real application.

    Returns:
        A ChromaDB client instance.
    """
    settings = get_settings()

    client = chromadb.PersistentClient(
        path=settings.chroma_persist_directory,
        settings=ChromaSettings(
            # anonymized_telemetry=False stops ChromaDB from sending
            # usage data to their servers — good practice for privacy
            anonymized_telemetry=False,
        ),
    )

    logger.debug(f"ChromaDB client created at: {settings.chroma_persist_directory}")
    return client


def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """
    Get the ChromaDB collection (or create it if it doesn't exist yet).

    A ChromaDB collection is like a table — it holds all our vectors
    together with their metadata and original text.

    We use get_or_create_collection() (not get_collection) so the app
    starts cleanly on first run without manual setup.

    Args:
        client: A ChromaDB client instance.

    Returns:
        The ChromaDB collection for DataPulse documents.
    """
    settings = get_settings()

    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        # cosine distance measures the ANGLE between vectors —
        # better than euclidean distance for text similarity because
        # it's not affected by the length of the text
        metadata={"hnsw:space": "cosine"},
    )

    logger.debug(f"Using ChromaDB collection: {settings.chroma_collection_name}")
    return collection


def store_chunks(
    collection: chromadb.Collection,
    *,
    chunks: list[str],
    embeddings: list[list[float]],
    document_id: str,
    filename: str,
) -> int:
    """
    Store text chunks and their embeddings in ChromaDB.

    Each chunk is stored with:
      - A unique ID          : "doc_id::chunk_0", "doc_id::chunk_1", etc.
      - Its embedding vector : the numeric representation of its meaning
      - Its original text    : so we can return it in query responses
      - Metadata             : document_id and filename for filtering

    Args:
        collection:   The ChromaDB collection to store into.
        chunks:       List of text strings (from chunker.py).
        embeddings:   List of vectors (from embeddings.py), same order as chunks.
        document_id:  UUID of the parent Document in PostgreSQL.
        filename:     Original filename — stored as metadata for display.

    Returns:
        Number of chunks stored.

    Raises:
        ValueError: If chunks and embeddings lists have different lengths.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
        )

    if not chunks:
        logger.warning(f"No chunks to store for document {document_id}")
        return 0

    # Build parallel lists — ChromaDB's add() takes lists, not dicts
    ids = [f"{document_id}::chunk_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,       # "documents" here means the raw text — ChromaDB's terminology
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(chunks)} chunks in ChromaDB for document {document_id}")
    return len(chunks)


def search_similar_chunks(
    collection: chromadb.Collection,
    *,
    query_embedding: list[float],
    top_k: int = 4,
    document_id: str | None = None,
) -> list[dict]:
    """
    Find the top-K most semantically similar chunks to a query.

    This is the RETRIEVAL step in RAG (Retrieval Augmented Generation).
    We convert the question into a vector, then ask ChromaDB:
    "Which stored vectors are closest in meaning to this one?"

    Args:
        collection:      The ChromaDB collection to search.
        query_embedding: The embedded question vector.
        top_k:           How many chunks to return.
        document_id:     If provided, search only within this document.
                         If None, search across all documents.

    Returns:
        List of dicts, each with keys:
          - text         : the chunk's original text
          - document_id  : which document it came from
          - filename     : original filename for display
          - chunk_index  : position in the original document
          - distance     : similarity score (lower = more similar for cosine)
    """
    # Build an optional filter — ChromaDB calls these "where" clauses
    # If document_id is provided, only return chunks from that document
    where_filter = {"document_id": document_id} if document_id else None

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    # Only add the where clause if we're filtering — passing where=None errors
    if where_filter:
        query_params["where"] = where_filter

    results = collection.query(**query_params)

    # ChromaDB returns parallel lists — zip them into a readable structure
    chunks_out = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, distance in zip(documents, metadatas, distances):
        chunks_out.append({
            "text": text,
            "document_id": meta.get("document_id", ""),
            "filename": meta.get("filename", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": round(distance, 4),
        })

    logger.info(
        f"ChromaDB search returned {len(chunks_out)} chunks "
        f"(top_k={top_k}, filtered_by_doc={document_id is not None})"
    )
    return chunks_out


def delete_document_chunks(collection: chromadb.Collection, document_id: str) -> None:
    """
    Delete all chunks belonging to a specific document from ChromaDB.

    Called when a document is deleted via the API — we must clean up
    both PostgreSQL (the metadata) AND ChromaDB (the vectors).
    Leaving orphaned vectors wastes space and pollutes search results.

    Args:
        collection:  The ChromaDB collection to delete from.
        document_id: The document whose chunks should be removed.
    """
    collection.delete(where={"document_id": document_id})
    logger.info(f"Deleted all ChromaDB chunks for document {document_id}")