"""
vector_store.py — LanceDB interface for storing and retrieving vectors.

We use LanceDB instead of ChromaDB because LanceDB is pure Python —
no C++ compilation required, works on all platforms including Windows.

The concepts are identical to ChromaDB:
  - A "table" in LanceDB = a "collection" in ChromaDB
  - We store vectors + text + metadata
  - We search by vector similarity (cosine distance)

HOW IT FITS IN THE RAG PIPELINE:
  Ingest:  chunk → embed → store in LanceDB (with metadata)
  Query:   embed question → search LanceDB → get top-K chunks back
                                                      ↓
                                           send to LLM as context
"""

import logging
import os
import pyarrow as pa
import lancedb

from app.config import get_settings

logger = logging.getLogger(__name__)

# LanceDB table schema — defines the columns in our vector table
# Every stored chunk has: a vector, the text, and metadata fields
def get_schema() -> pa.Schema:
    """
    Define the LanceDB table schema using PyArrow.

    The vector dimension (768) matches nomic-embed-text output.
    If you switch embedding models, update this dimension to match.
    """
    settings = get_settings()
    return pa.schema([
        pa.field("vector",      pa.list_(pa.float32(), 384)),  # embedding vector
        pa.field("text",        pa.utf8()),                     # original chunk text
        pa.field("document_id", pa.utf8()),                     # parent document UUID
        pa.field("filename",    pa.utf8()),                     # original filename
        pa.field("chunk_index", pa.int32()),                    # position in document
        pa.field("chunk_id",    pa.utf8()),                     # unique chunk identifier
    ])


def get_lancedb_table() -> lancedb.table.Table:
    """
    Connect to LanceDB and get (or create) the documents table.

    LanceDB stores data as files on disk at chroma_persist_directory
    (we reuse the same config key to avoid adding a new env var).
    The directory is created automatically if it doesn't exist.

    Returns:
        A LanceDB Table object ready for reads and writes.
    """
    settings = get_settings()
    db_path = settings.chroma_persist_directory  # reusing config key
    table_name = settings.chroma_collection_name

    # Connect to local LanceDB (creates folder if needed)
    db = lancedb.connect(db_path)

    # Get existing table or create a new one with our schema
    if table_name in db.table_names():
        table = db.open_table(table_name)
        logger.debug(f"Opened existing LanceDB table: {table_name}")
    else:
        table = db.create_table(table_name, schema=get_schema())
        logger.debug(f"Created new LanceDB table: {table_name}")

    return table


def store_chunks(
    table: lancedb.table.Table,
    *,
    chunks: list[str],
    embeddings: list[list[float]],
    document_id: str,
    filename: str,
) -> int:
    """
    Store text chunks and their embeddings in LanceDB.

    Each chunk is stored as one row with:
      - vector      : the embedding (list of 768 floats)
      - text        : the original chunk text
      - document_id : links back to the PostgreSQL Document record
      - filename    : shown in query responses as the source
      - chunk_index : position within the document
      - chunk_id    : unique ID for this specific chunk

    Args:
        table:       LanceDB table to write into.
        chunks:      List of text strings from chunker.py.
        embeddings:  List of vectors from embeddings.py (same order).
        document_id: UUID of the parent Document in PostgreSQL.
        filename:    Original filename for display in responses.

    Returns:
        Number of chunks stored.

    Raises:
        ValueError: If chunks and embeddings have different lengths.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
        )

    if not chunks:
        logger.warning(f"No chunks to store for document {document_id}")
        return 0

    # Build list of row dicts — LanceDB accepts a list of dicts
    rows = [
        {
            "vector":      [float(v) for v in embeddings[i]],
            "text":        chunks[i],
            "document_id": document_id,
            "filename":    filename,
            "chunk_index": i,
            "chunk_id":    f"{document_id}::chunk_{i}",
        }
        for i in range(len(chunks))
    ]

    table.add(rows)

    logger.info(f"Stored {len(chunks)} chunks in LanceDB for document {document_id}")
    return len(chunks)


def search_similar_chunks(
    table: lancedb.table.Table,
    *,
    query_embedding: list[float],
    top_k: int = 4,
    document_id: str | None = None,
) -> list[dict]:
    """
    Find the top-K most semantically similar chunks to a query vector.

    This is the RETRIEVAL step in RAG. We convert the question into
    a vector, then ask LanceDB: 'Which stored vectors are closest
    in meaning to this one?'

    Args:
        table:           LanceDB table to search.
        query_embedding: The embedded question vector.
        top_k:           How many chunks to return.
        document_id:     If set, filter results to this document only.

    Returns:
        List of dicts with keys: text, document_id, filename,
        chunk_index, distance.
    """
    query_vector = [float(v) for v in query_embedding]

    # Build the search query
    search = (
        table.search(query_vector)
             .limit(top_k)
             .select(["text", "document_id", "filename", "chunk_index"])
    )

    # Apply document filter if provided
    if document_id:
        search = search.where(f"document_id = '{document_id}'")

    results = search.to_list()

    chunks_out = [
        {
            "text":        row["text"],
            "document_id": row["document_id"],
            "filename":    row["filename"],
            "chunk_index": row["chunk_index"],
            "distance":    round(row.get("_distance", 0.0), 4),
        }
        for row in results
    ]

    logger.info(
        f"LanceDB search returned {len(chunks_out)} chunks "
        f"(top_k={top_k}, filtered_by_doc={document_id is not None})"
    )
    return chunks_out


def delete_document_chunks(
    table: lancedb.table.Table,
    document_id: str,
) -> None:
    """
    Delete all chunks belonging to a specific document.

    Called when a document is deleted via DELETE /documents/{id}.
    We must clean up vectors here to prevent orphaned data polluting
    future search results.

    Args:
        table:       LanceDB table to delete from.
        document_id: The document whose chunks should be removed.
    """
    table.delete(f"document_id = '{document_id}'")
    logger.info(f"Deleted all LanceDB chunks for document {document_id}")