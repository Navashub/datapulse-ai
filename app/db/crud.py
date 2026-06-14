"""
crud.py — Create, Read, Update, Delete operations for PostgreSQL.

All database interactions live here — routes never write raw SQL or
call SQLAlchemy directly. This separation means:
  - Routes stay thin and readable
  - DB logic is testable in isolation
  - Swapping the database later only requires changes in this file
"""

import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.models import Document, QueryHistory

logger = logging.getLogger(__name__)


# ── Document Operations ───────────────────────────────────────────────────────

def create_document(
    db: Session,
    *,
    doc_id: str,
    filename: str,
    file_type: str,
    char_count: int,
    chunk_count: int,
    chroma_collection: str,
) -> Document:
    """
    Insert a new Document record into PostgreSQL.

    We use keyword-only arguments (the * forces this) to prevent
    accidental argument ordering bugs — with 6 similar parameters,
    positional args are a silent bug waiting to happen.

    Returns the newly created Document ORM object.
    """
    document = Document(
        id=doc_id,
        filename=filename,
        file_type=file_type,
        char_count=char_count,
        chunk_count=chunk_count,
        chroma_collection=chroma_collection,
    )
    db.add(document)
    db.commit()
    db.refresh(document)  # Reload from DB to populate server-set fields (created_at)

    logger.info(f"Document created: id={doc_id}, filename={filename}, chunks={chunk_count}")
    return document


def get_document_by_id(db: Session, doc_id: str) -> Document | None:
    """
    Fetch a single document by its UUID.

    Returns None if no document exists with that ID.
    Routes should handle the None case and return a 404.
    """
    return db.query(Document).filter(Document.id == doc_id).first()


def get_all_documents(db: Session, limit: int = 100) -> list[Document]:
    """
    Return all documents, ordered newest first.

    limit=100 is a safety cap — without it, a large database would
    return thousands of rows and potentially crash the API response.
    """
    return (
        db.query(Document)
        .order_by(desc(Document.created_at))
        .limit(limit)
        .all()
    )


def delete_document(db: Session, doc_id: str) -> bool:
    """
    Delete a document record by ID.

    Returns True if deleted, False if the document was not found.
    The cascade="all, delete-orphan" on the relationship means
    related QueryHistory rows are automatically deleted too.
    """
    document = get_document_by_id(db, doc_id)
    if not document:
        return False

    db.delete(document)
    db.commit()
    logger.info(f"Document deleted: id={doc_id}")
    return True


# ── QueryHistory Operations ───────────────────────────────────────────────────

def create_query_history(
    db: Session,
    *,
    query_id: str,
    question: str,
    answer: str,
    document_id: str | None,
    chunks_retrieved: int,
    response_time_seconds: float,
) -> QueryHistory:
    """
    Log a completed RAG query to the database.

    Every successful /query call should produce one QueryHistory row.
    This gives us a full audit trail of what was asked and answered.
    """
    entry = QueryHistory(
        id=query_id,
        question=question,
        answer=answer,
        document_id=document_id,
        chunks_retrieved=chunks_retrieved,
        response_time_seconds=round(response_time_seconds, 3),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    logger.info(
        f"Query logged: id={query_id}, chunks={chunks_retrieved}, "
        f"time={response_time_seconds:.2f}s"
    )
    return entry


def get_recent_queries(db: Session, limit: int = 20) -> list[QueryHistory]:
    """
    Return the most recent queries, newest first.

    Useful for a dashboard or debugging view.
    Default limit of 20 keeps responses snappy.
    """
    return (
        db.query(QueryHistory)
        .order_by(desc(QueryHistory.created_at))
        .limit(limit)
        .all()
    )