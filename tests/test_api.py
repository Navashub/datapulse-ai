"""
test_api.py — Pytest test suite for DataPulse AI.

TESTING STRATEGY:
  We use FastAPI's TestClient which runs the app in-process —
  no need to spin up a real server. This makes tests fast and
  reliable in CI environments.

  Test categories:
    1. Health check          — always works, no dependencies
    2. Ingest (text file)    — happy path
    3. Ingest validation     — bad inputs return correct errors
    4. Query validation      — bad inputs caught before pipeline runs
    5. Documents listing     — reflects ingested state
    6. Full RAG flow         — ingest then query end-to-end

  We use pytest fixtures to share setup across tests and
  unittest.mock to isolate external dependencies (Ollama, ChromaDB)
  so tests pass without those services running.
"""

import io
import uuid
import pytest

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

# ── Test Client ───────────────────────────────────────────────────────────────
# TestClient wraps the FastAPI app — no real HTTP, no real server.
# Requests go directly through the ASGI interface.
client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_text_file() -> tuple[str, bytes, str]:
    """
    A minimal in-memory text file for upload tests.

    Returns (filename, content_bytes, content_type).
    Using io.BytesIO means no real files on disk during testing.
    """
    content = (
        "DataPulse AI is an AI-powered document intelligence platform. "
        "It allows users to ingest documents and ask questions about them. "
        "The system uses RAG (Retrieval Augmented Generation) to ground answers "
        "in the actual document content rather than LLM training data. "
        "ChromaDB stores the vector embeddings. Ollama runs the local AI models. "
        "PostgreSQL stores document metadata and query history."
    )
    return ("test_document.txt", content.encode("utf-8"), "text/plain")


@pytest.fixture
def mock_db_session():
    """
    Mock SQLAlchemy session — prevents real DB calls during unit tests.

    For integration tests against a real DB you would use a test
    database and real sessions. For unit tests, mocking is faster
    and avoids environment dependencies.
    """
    mock_session = MagicMock()
    return mock_session


@pytest.fixture
def mock_document():
    """A fake Document ORM object for use in mock return values."""
    doc = MagicMock()
    doc.id = str(uuid.uuid4())
    doc.filename = "test_document.txt"
    doc.file_type = "text"
    doc.char_count = 500
    doc.chunk_count = 3
    doc.chroma_collection = "datapulse_documents"
    return doc


# ── Test 1: Health Check ──────────────────────────────────────────────────────

class TestHealthEndpoint:
    """
    Health check tests — these should ALWAYS pass regardless of
    whether Ollama, ChromaDB, or PostgreSQL are running.
    """

    def test_health_returns_200(self):
        """GET /health must return 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_shape(self):
        """Response must include all required fields."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "status" in data
        assert "app_name" in data
        assert "version" in data
        assert "message" in data

    def test_health_status_is_ok(self):
        """Status field must be 'ok' — not 'error' or anything else."""
        response = client.get("/api/v1/health")
        assert response.json()["status"] == "ok"

    def test_health_app_name(self):
        """App name must match config."""
        response = client.get("/api/v1/health")
        assert response.json()["app_name"] == "DataPulse AI"


# ── Test 2: Ingest Validation ─────────────────────────────────────────────────

class TestIngestValidation:
    """
    Test that bad inputs are rejected BEFORE hitting Ollama or ChromaDB.
    These tests do not need any external services running.
    """

    def test_ingest_rejects_unsupported_file_type(self):
        """
        .csv files should return 400 Bad Request.
        We validate file type before any expensive processing.
        """
        fake_csv = io.BytesIO(b"col1,col2\nval1,val2")
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("data.csv", fake_csv, "text/csv")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_ingest_rejects_empty_file(self):
        """
        An empty file should return 400 Bad Request.
        No point chunking or embedding nothing.
        """
        empty_file = io.BytesIO(b"")
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("empty.txt", empty_file, "text/plain")},
        )
        assert response.status_code == 400

    def test_ingest_requires_file(self):
        """
        POST /ingest with no file should return 422 Unprocessable Entity.
        FastAPI/Pydantic handles this automatically.
        """
        response = client.post("/api/v1/ingest")
        assert response.status_code == 422


# ── Test 3: Ingest Happy Path (mocked) ───────────────────────────────────────

class TestIngestHappyPath:
    """
    Test successful ingestion with all external calls mocked out.

    We mock:
      - embed_batch      → skips Ollama embedding calls
      - store_chunks     → skips ChromaDB writes
      - create_document  → skips PostgreSQL writes
      - get_db           → skips real DB session
    """

    def test_ingest_text_file_returns_201(self, sample_text_file, mock_document):
        """Successful ingestion must return HTTP 201 Created."""
        filename, content, content_type = sample_text_file

        with (
            patch("app.api.routes.ingest.embed_batch", return_value=[[0.1] * 768] * 3),
            patch("app.api.routes.ingest.store_chunks", return_value=3),
            patch("app.api.routes.ingest.get_chroma_client", return_value=MagicMock()),
            patch("app.api.routes.ingest.get_or_create_collection", return_value=MagicMock()),
            patch("app.api.routes.ingest.create_document", return_value=mock_document),
            patch("app.db.database.get_db", return_value=MagicMock()),
        ):
            response = client.post(
                "/api/v1/ingest",
                files={"file": (filename, io.BytesIO(content), content_type)},
            )

        assert response.status_code == 201

    def test_ingest_response_contains_document_id(self, sample_text_file, mock_document):
        """Response must include a document_id for future queries."""
        filename, content, content_type = sample_text_file

        with (
            patch("app.api.routes.ingest.embed_batch", return_value=[[0.1] * 768] * 3),
            patch("app.api.routes.ingest.store_chunks", return_value=3),
            patch("app.api.routes.ingest.get_chroma_client", return_value=MagicMock()),
            patch("app.api.routes.ingest.get_or_create_collection", return_value=MagicMock()),
            patch("app.api.routes.ingest.create_document", return_value=mock_document),
            patch("app.db.database.get_db", return_value=MagicMock()),
        ):
            response = client.post(
                "/api/v1/ingest",
                files={"file": (filename, io.BytesIO(content), content_type)},
            )

        data = response.json()
        assert "document_id" in data
        assert "chunk_count" in data
        assert "message" in data


# ── Test 4: Query Validation ──────────────────────────────────────────────────

class TestQueryValidation:
    """
    Test that invalid query requests are rejected before hitting the pipeline.
    Pydantic handles most of this — we verify the behaviour here.
    """

    def test_query_rejects_empty_question(self):
        """
        A question shorter than 3 characters should return 422.
        Defined by min_length=3 in QueryRequest schema.
        """
        response = client.post(
            "/api/v1/query",
            json={"question": "Hi"},
        )
        assert response.status_code == 422

    def test_query_rejects_missing_question(self):
        """POST /query with no body should return 422."""
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_query_rejects_top_k_out_of_range(self):
        """
        top_k must be between 1 and 10.
        Values outside this range return 422.
        """
        response = client.post(
            "/api/v1/query",
            json={"question": "What is DataPulse AI?", "top_k": 99},
        )
        assert response.status_code == 422

    def test_query_accepts_valid_request_shape(self):
        """
        A well-formed request should NOT be rejected at validation.
        We mock the pipeline so this tests the route layer only,
        not the actual RAG execution.
        """
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "query_id": str(uuid.uuid4()),
            "question": "What is DataPulse AI?",
            "answer": "DataPulse AI is a document intelligence platform.",
            "sources": [],
            "chunks_retrieved": 2,
            "response_time_seconds": 1.23,
            "model_used": "llama3.2",
        }

        with patch("app.api.routes.query.run_rag_pipeline", return_value=mock_response):
            response = client.post(
                "/api/v1/query",
                json={"question": "What is DataPulse AI?"},
            )

        # Should not be a validation error — the pipeline mock handles the rest
        assert response.status_code != 422


# ── Test 5: Chunker Unit Tests ────────────────────────────────────────────────

class TestChunker:
    """
    Unit tests for the text chunking utility.
    Pure Python — no mocking, no external services needed.
    """

    def test_chunk_text_basic_split(self):
        """A long string must be split into multiple chunks."""
        from app.core.chunker import chunk_text

        text = "word " * 300  # 1500 characters
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)
        assert all(len(c) > 0 for c in chunks)

    def test_chunk_text_empty_input(self):
        """Empty string must return an empty list, not raise an error."""
        from app.core.chunker import chunk_text

        chunks = chunk_text("", chunk_size=500, chunk_overlap=50)
        assert chunks == []

    def test_chunk_text_short_input(self):
        """Text shorter than chunk_size must return exactly one chunk."""
        from app.core.chunker import chunk_text

        text = "This is a short document."
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_overlap_invalid(self):
        """chunk_overlap >= chunk_size must raise ValueError."""
        from app.core.chunker import chunk_text

        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("some text", chunk_size=100, chunk_overlap=100)

    def test_extract_text_from_txt(self):
        """UTF-8 text bytes must be decoded correctly."""
        from app.core.chunker import extract_text_from_file

        content = "Hello, DataPulse AI!".encode("utf-8")
        result = extract_text_from_file(content, "text")

        assert result == "Hello, DataPulse AI!"

    def test_extract_text_unsupported_type(self):
        """Unsupported file type must raise ValueError."""
        from app.core.chunker import extract_text_from_file

        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text_from_file(b"data", "csv")


# ── Test 6: Documents Endpoint ────────────────────────────────────────────────

class TestDocumentsEndpoint:
    """Tests for GET /documents — verifies the listing endpoint shape."""

    def test_documents_returns_200(self):
        """GET /documents must always return 200, even if list is empty."""
        with patch("app.api.routes.documents.get_all_documents", return_value=[]):
            response = client.get("/api/v1/documents")

        assert response.status_code == 200

    def test_documents_response_shape(self):
        """Response must include 'total' and 'documents' keys."""
        with patch("app.api.routes.documents.get_all_documents", return_value=[]):
            response = client.get("/api/v1/documents")

        data = response.json()
        assert "total" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_documents_total_matches_list_length(self):
        """'total' field must equal the length of 'documents' list."""
        with patch("app.api.routes.documents.get_all_documents", return_value=[]):
            response = client.get("/api/v1/documents")

        data = response.json()
        assert data["total"] == len(data["documents"])

    def test_get_document_by_invalid_id_returns_404(self):
        """Requesting a non-existent document ID must return 404."""
        fake_id = str(uuid.uuid4())
        with patch("app.api.routes.documents.get_document_by_id", return_value=None):
            response = client.get(f"/api/v1/documents/{fake_id}")

        assert response.status_code == 404