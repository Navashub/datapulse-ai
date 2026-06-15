"""
ingest.py — POST /ingest endpoint.

This route handles the full document ingestion pipeline:
  1. Accept an uploaded file (text or PDF)
  2. Extract the raw text
  3. Split into chunks
  4. Embed each chunk via Ollama
  5. Store vectors in ChromaDB
  6. Store metadata in PostgreSQL
  7. Return a summary to the caller

This is the WRITE side of DataPulse AI.
Nothing can be queried until something is ingested.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.config import get_settings, Settings
from app.core.chunker import chunk_text, extract_text_from_file
from app.core.embeddings import embed_batch
from app.core.vector_store import (
    get_lancedb_table,
    store_chunks,
)
from app.db.crud import create_document
from app.db.database import get_db
from app.schemas.schemas import IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# File size limit — 10MB. Large files are fine for chunking but
# can cause memory pressure during embedding. Adjust as needed.
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description="Upload a .txt or .pdf file to extract, chunk, embed, and store.",
)
async def ingest_document(
    file: UploadFile = File(..., description="A .txt or .pdf file to ingest"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """
    Ingest an uploaded document into the DataPulse AI knowledge base.

    Accepts .txt and .pdf files. The file is processed synchronously —
    for very large files (50+ pages) this may take 30–60 seconds due
    to the embedding step. A production system would handle this
    with a background task queue (Celery, ARQ) — good stretch goal.

    Args:
        file:     The uploaded file from the multipart form request.
        db:       PostgreSQL session (injected by FastAPI).
        settings: App config (injected by FastAPI).

    Returns:
        IngestResponse with document ID, chunk count, and confirmation.

    Raises:
        400: Unsupported file type or empty file.
        413: File exceeds the size limit.
        500: Embedding or storage failure.
    """
    # ── VALIDATE FILE TYPE ────────────────────────────────────────────────────
    filename = file.filename or "unknown"
    logger.info(f"Ingest request received: {filename}")

    if filename.endswith(".txt") or filename.endswith(".md"):
        file_type = "text"
    elif filename.endswith(".pdf"):
        file_type = "pdf"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type: '{filename}'. "
                "Only .txt, .md, and .pdf files are accepted."
            ),
        )

    # ── READ FILE CONTENT ─────────────────────────────────────────────────────
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is 10MB.",
        )

    # ── EXTRACT TEXT ──────────────────────────────────────────────────────────
    try:
        text = extract_text_from_file(content, file_type)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    logger.info(f"Extracted {len(text):,} characters from {filename}")

    # ── CHUNK TEXT ────────────────────────────────────────────────────────────
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document produced no text chunks. The file may be empty or unreadable.",
        )

    logger.info(f"Split into {len(chunks)} chunks")

    # ── EMBED CHUNKS ──────────────────────────────────────────────────────────
    # This is the slowest step — each chunk makes a call to Ollama
    try:
        embeddings = embed_batch(chunks)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # ── STORE IN CHROMADB ─────────────────────────────────────────────────────
    document_id = str(uuid.uuid4())

    try:
        table = get_lancedb_table()
        store_chunks(
            table,
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            filename=filename,
        )
    except Exception as e:
        logger.error(f"ChromaDB storage failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store document vectors: {e}",
        )

    # ── STORE METADATA IN POSTGRESQL ──────────────────────────────────────────
    try:
        document = create_document(
            db,
            doc_id=document_id,
            filename=filename,
            file_type=file_type,
            char_count=len(text),
            chunk_count=len(chunks),
            chroma_collection=settings.chroma_collection_name,
        )
    except Exception as e:
        logger.error(f"PostgreSQL storage failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save document metadata: {e}",
        )

    logger.info(
        f"Ingestion complete: doc_id={document_id}, "
        f"chunks={len(chunks)}, chars={len(text):,}"
    )

    return IngestResponse(
        document_id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        char_count=document.char_count,
        chunk_count=document.chunk_count,
        message=(
            f"Successfully ingested '{filename}' — "
            f"{len(chunks)} chunks ready for querying."
        ),
    )