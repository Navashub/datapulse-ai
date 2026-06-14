"""
schemas.py — Pydantic schemas for request validation and response serialization.

These are DIFFERENT from SQLAlchemy models (db/models.py):
  - SQLAlchemy models   : define the DATABASE table structure
  - Pydantic schemas    : define the API request/response shape

Why separate? Because what the API accepts/returns is often different
from what the database stores. For example, the API never exposes
internal IDs in requests, and never returns raw DB objects directly.
"""

from pydantic import BaseModel, Field
from datetime import datetime


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response schema for GET /health"""
    status: str
    app_name: str
    version: str
    message: str


# ── Document Ingest ───────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """
    Response returned after successfully ingesting a document.

    Tells the caller everything that happened during ingestion:
    how many chunks were created, which document ID to use for
    future queries, etc.
    """
    document_id: str
    filename: str
    file_type: str
    char_count: int
    chunk_count: int
    message: str

    model_config = {"from_attributes": True}  # Allows building from ORM objects


# ── Document Listing ──────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    """
    Represents a single document in the GET /documents response.

    We deliberately exclude the full text — returning it would make
    the response enormous for large documents.
    """
    id: str
    filename: str
    file_type: str
    char_count: int
    chunk_count: int
    chroma_collection: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Wrapper for the list of documents — makes it easy to add pagination later."""
    total: int
    documents: list[DocumentResponse]


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Request body for POST /query.

    question    : the natural language question to answer
    document_id : optional — if provided, search only within that document.
                  If omitted, search across ALL ingested documents.
    top_k       : how many chunks to retrieve — overrides the server default.
    """
    question: str = Field(
        ...,                          # ... means required — no default
        min_length=3,
        max_length=1000,
        description="The question to ask about your documents",
        examples=["What are the key findings in this report?"],
    )
    document_id: str | None = Field(
        default=None,
        description="Scope the search to a specific document ID (optional)",
    )
    top_k: int = Field(
        default=4,
        ge=1,                         # ge = greater than or equal to
        le=10,                        # le = less than or equal to
        description="Number of document chunks to retrieve (1–10)",
    )


class SourceChunk(BaseModel):
    """
    A single retrieved chunk shown in the query response.

    Showing sources is critical for RAG — it lets users verify
    the answer came from real content, not LLM hallucination.
    """
    chunk_index: int
    text: str
    document_id: str
    filename: str


class QueryResponse(BaseModel):
    """
    Full response from POST /query.

    Includes the answer AND the source chunks that produced it.
    This transparency is what makes RAG trustworthy vs. a black-box LLM.
    """
    query_id: str
    question: str
    answer: str
    sources: list[SourceChunk]        # The chunks the LLM used to answer
    chunks_retrieved: int
    response_time_seconds: float
    model_used: str                   # e.g. "llama3.2" — useful for debugging


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """
    Standard error shape returned by all error handlers.

    Consistent error responses mean the frontend/client always knows
    exactly where to find the error message — no guessing the shape.
    """
    error: str
    detail: str | None = None