"""
documents.py — GET /documents endpoint.

Lists all documents that have been ingested into DataPulse AI.
Simple but important — users need to know what's in the system
before they can query it intelligently.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.crud import get_all_documents, get_document_by_id, delete_document
from app.db.database import get_db
from app.core.vector_store import get_lancedb_table, delete_document_chunks
from app.schemas.schemas import DocumentListResponse, DocumentResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
    description="Returns metadata for every document currently in the knowledge base.",
)
async def list_documents(
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """
    Return all ingested documents ordered by most recently added.

    Does NOT return the document text — only metadata.
    This keeps the response fast even with hundreds of documents.

    Args:
        db: PostgreSQL session.

    Returns:
        DocumentListResponse with total count and list of documents.
    """
    documents = get_all_documents(db)

    logger.info(f"Listed {len(documents)} documents")

    return DocumentListResponse(
        total=len(documents),
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Get a single document by ID",
    description="Returns metadata for a specific ingested document.",
)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """
    Fetch a single document's metadata by its UUID.

    Useful for confirming a specific document was ingested correctly,
    or for getting the document_id to scope a /query call.

    Args:
        document_id: The UUID of the document to fetch.
        db:          PostgreSQL session.

    Returns:
        DocumentResponse for the matching document.

    Raises:
        404: No document found with the given ID.
    """
    document = get_document_by_id(db, document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document found with id '{document_id}'",
        )

    return DocumentResponse.model_validate(document)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Removes a document from both PostgreSQL and ChromaDB.",
)
async def remove_document(
    document_id: str,
    db: Session = Depends(get_db),
) -> None:
    """
    Delete a document and ALL its associated data.

    This removes:
      - The Document row from PostgreSQL
      - All QueryHistory rows linked to this document (cascade delete)
      - All vector chunks from ChromaDB

    This two-phase delete (ChromaDB first, then PostgreSQL) is
    intentional — if ChromaDB deletion fails, we still have the
    PostgreSQL record and can retry. If we deleted PostgreSQL first
    and ChromaDB failed, we'd have orphaned vectors with no metadata.

    Args:
        document_id: UUID of the document to delete.
        db:          PostgreSQL session.

    Raises:
        404: No document found with that ID.
        500: ChromaDB deletion failed.
    """
    # Confirm document exists before attempting deletion
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document found with id '{document_id}'",
        )

    # Phase 1 — delete vectors from ChromaDB
    try:
        table = get_lancedb_table()
        delete_document_chunks(table, document_id)
    except Exception as e:
        logger.error(f"ChromaDB deletion failed for {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document vectors: {e}",
        )

    # Phase 2 — delete metadata from PostgreSQL
    delete_document(db, document_id)

    logger.info(f"Document fully deleted: {document_id}")
    # 204 No Content — return nothing on successful delete