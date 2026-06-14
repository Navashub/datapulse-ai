"""
models.py — SQLAlchemy ORM models for DataPulse AI.

These classes define the shape of our PostgreSQL tables.
SQLAlchemy reads these at startup and Alembic uses them to
generate migration files — so the database schema always
stays in sync with what the code expects.

Two tables:
  - Document       : metadata about every file we ingest
  - QueryHistory   : a log of every question asked + the answer given
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Text,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """
    Base class that all ORM models inherit from.

    DeclarativeBase (SQLAlchemy 2.0 style) gives us:
      - The metadata registry that Alembic reads for migrations
      - The __tablename__ convention
      - Type-safe mapped_column() support
    """
    pass


class Document(Base):
    """
    Represents a document that has been ingested into DataPulse AI.

    When a user uploads a file (text or PDF), we:
      1. Store the file's metadata here in PostgreSQL
      2. Split the text into chunks and store the vectors in ChromaDB

    The `chroma_collection` field links this record to its vectors —
    so if we ever need to delete a document, we know which ChromaDB
    collection to clean up.
    """

    __tablename__ = "documents"

    # UUID primary key — better than auto-increment integers for APIs
    # because IDs are unpredictable (no enumeration attacks) and can be
    # generated client-side without a database round-trip.
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Original filename as uploaded — shown back to users in /documents
    filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # "text" or "pdf" — lets us handle display and processing differently
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)

    # How many characters the full extracted text contains.
    # Useful for understanding document size at a glance.
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # How many chunks we split this document into.
    # Stored so we can show "ingested 12 chunks" in the API response.
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # The ChromaDB collection where this document's vectors live.
    # Currently always the same collection, but storing it here makes
    # it easy to support per-user or per-project collections later.
    chroma_collection: Mapped[str] = mapped_column(String(255), nullable=False)

    # Timestamp set automatically by the database when the row is created.
    # We use func.now() so the DB sets it — not the application — which
    # avoids timezone inconsistencies across machines.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # One document can have many queries made against it.
    # back_populates keeps both sides of the relationship in sync.
    queries: Mapped[list["QueryHistory"]] = relationship(
        "QueryHistory",
        back_populates="document",
        cascade="all, delete-orphan",  # deleting a document deletes its query history too
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id!r} filename={self.filename!r} chunks={self.chunk_count}>"


class QueryHistory(Base):
    """
    Logs every question asked through the /query endpoint.

    Storing query history lets us:
      - Show users what has been asked before
      - Debug RAG pipeline quality (compare question vs answer)
      - Eventually build analytics on popular topics
    """

    __tablename__ = "query_history"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # The exact question the user asked
    question: Mapped[str] = mapped_column(Text, nullable=False)

    # The answer the LLM generated — stored as Text (unlimited length)
    answer: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional: which document this query was scoped to.
    # NULL means the query searched across ALL ingested documents.
    document_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    # How many ChromaDB chunks were retrieved and sent to the LLM.
    # Useful for debugging — if chunks_retrieved is 0, the answer is a hallucination.
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # How long the full RAG pipeline took in seconds.
    # Latency tracking is the first step toward performance optimisation.
    response_time_seconds: Mapped[float] = mapped_column(Float, nullable=True)

    # Auto-set timestamp on insert
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Many-to-one: each query belongs to at most one document
    document: Mapped["Document | None"] = relationship(
        "Document",
        back_populates="queries",
    )

    def __repr__(self) -> str:
        return f"<QueryHistory id={self.id!r} question={self.question[:40]!r}>"