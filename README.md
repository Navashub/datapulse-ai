# DataPulse AI 

A local document intelligence API built with FastAPI, PostgreSQL, LanceDB, and Ollama.

Upload `.txt`, `.md`, or `.pdf` documents, ingest them into a vector store, and ask questions with grounded answers and source chunks.

---

## What This Project Does

- Ingests plain text and PDF documents
- Extracts and chunks document text
- Converts chunks into embeddings via Ollama
- Stores vectors in a local LanceDB store
- Persists metadata and query history in PostgreSQL
- Answers questions with a Retrieval-Augmented Generation (RAG) pipeline
- Returns answer text plus source chunks for transparency

---

## Prerequisites

Install the following before running the project:

- Python 3.11+
- PostgreSQL 14+ (or NeonDB)
- Ollama
- Git

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/datapulse-ai.git
cd datapulse-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama and pull models

Run Ollama in a separate terminal:

```bash
ollama serve
```

Then pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Verify the models:

```bash
ollama list
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your `DATABASE_URL`.

Example local PostgreSQL:

```bash
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/datapulse_db
```

Example NeonDB:

```bash
DATABASE_URL=postgresql://user:password@ep-xxxx.aws.neon.tech/neondb?sslmode=require
```

Other environment variables can remain at their defaults for local development.

### 6. Create the PostgreSQL database (local only)

```bash
psql -U postgres -c "CREATE DATABASE datapulse_db;"
```

If you use NeonDB or another hosted database, skip this step.

### 7. Run database migrations

```bash
alembic upgrade head
```

If `alembic` is not available:

```bash
python -m alembic upgrade head
```

### 8. Start the API server

```bash
uvicorn app.main:app --reload
```

Open the API docs at:

```text
http://localhost:8000/docs
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Service health check |
| `/api/v1/ingest` | POST | Upload and ingest a document |
| `/api/v1/query` | POST | Ask a question against ingested documents |
| `/api/v1/documents` | GET | List all ingested documents |
| `/api/v1/documents/{document_id}` | GET | Get document metadata |
| `/api/v1/documents/{document_id}` | DELETE | Delete a document and its vectors |

---

## Usage Examples

### Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@your_document.txt"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```

### List ingested documents

```bash
curl http://localhost:8000/api/v1/documents
```

### Query a specific document

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about data privacy?", "document_id": "<DOCUMENT_ID>"}'
```

---

## Project Structure

```
datapulse-ai/
├── alembic/                 # Database migration config and versions
├── app/
│   ├── api/routes/          # FastAPI route handlers
│   ├── config.py            # Environment-driven settings
│   ├── core/                # RAG pipeline, chunking, embeddings, store
│   ├── db/                  # SQLAlchemy models and CRUD
│   ├── main.py              # FastAPI application entrypoint
│   └── schemas/schemas.py   # Request/response models
├── tests/                   # Pytest test suite
├── .env.example             # Example environment file
├── requirements.txt
└── README.md
```

---

## Architecture Overview

This app uses a Retrieval-Augmented Generation flow:

1. Ingested documents are converted to plain text.
2. Text is split into overlapping chunks.
3. Chunks are embedded using Ollama.
4. Embeddings are stored in a local LanceDB vector store.
5. Queries are embedded and matched against stored chunks.
6. The best chunks are included in a prompt to Ollama.
7. Ollama returns an answer grounded in the retrieved content.

This makes responses more reliable because the model is forced to use document content.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | PostgreSQL connection string |
| `OLLAMA_BASE_URL` | ❌ | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_CHAT_MODEL` | ❌ | `llama3.2` | Model for answer generation |
| `OLLAMA_EMBEDDING_MODEL` | ❌ | `nomic-embed-text` | Model for embeddings |
| `CHROMA_PERSIST_DIRECTORY` | ❌ | `./chroma_db` | Local vector store directory |
| `CHROMA_COLLECTION_NAME` | ❌ | `datapulse_documents` | Vector collection name |
| `RAG_TOP_K` | ❌ | `4` | Chunks retrieved per query |
| `CHUNK_SIZE` | ❌ | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | ❌ | `50` | Overlap between chunks |
| `DEBUG` | ❌ | `false` | Enable debug logging |

---

## Troubleshooting

### Ollama connection problems

- Verify Ollama is running:
  ```bash
  ollama serve
  ```
- Confirm required models are pulled:
  ```bash
  ollama list
  ```
- If a model is missing:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

### PostgreSQL connection errors

- Ensure `DATABASE_URL` is correct.
- For NeonDB, include `?sslmode=require`.
- For local Postgres, confirm the database exists and credentials are valid.

### Port 8000 already in use

```bash
uvicorn app.main:app --reload --port 8001
```

### Alembic command not found

```bash
python -m alembic upgrade head
```

---

## Run Tests

```bash
pytest tests/ -v
```

The test suite verifies API routes, validation, and basic behavior. Many external dependencies are mocked so the tests should run without Ollama or PostgreSQL.

You should see:
============================================================

DataPulse AI v1.0.0

Ollama URL    : http://localhost:11434

Chat model    : llama3.2

Embed model   : nomic-embed-text

ChromaDB path : ./chroma_db
DataPulse AI is ready. Visit http://localhost:8000/docs

Open your browser at **http://localhost:8000/docs** to see the
interactive Swagger UI.

---

## Using the API

### Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@your_document.txt"
```

Response:
```json
{
  "document_id": "3f7a1b2c-...",
  "filename": "your_document.txt",
  "file_type": "text",
  "char_count": 4821,
  "chunk_count": 12,
  "message": "Successfully ingested 'your_document.txt' — 12 chunks ready for querying."
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics in this document?"}'
```

Response:
```json
{
  "query_id": "9a2b3c4d-...",
  "question": "What are the main topics in this document?",
  "answer": "Based on the provided documents, the main topics are...",
  "sources": [
    {
      "chunk_index": 2,
      "text": "The document covers...",
      "document_id": "3f7a1b2c-...",
      "filename": "your_document.txt"
    }
  ],
  "chunks_retrieved": 4,
  "response_time_seconds": 3.241,
  "model_used": "llama3.2"
}
```

### List ingested documents

```bash
curl http://localhost:8000/api/v1/documents
```

---

## Run Tests

```bash
pytest tests/ -v
```

Expected: 18 tests, all passing.

---

## Project Structure
datapulse-ai/

├── app/

│   ├── main.py              # FastAPI app, router registration, logging

│   ├── config.py            # All config via environment variables

│   ├── api/routes/

│   │   ├── health.py        # GET  /api/v1/health

│   │   ├── ingest.py        # POST /api/v1/ingest

│   │   ├── query.py         # POST /api/v1/query

│   │   └── documents.py     # GET  /api/v1/documents

│   ├── core/

│   │   ├── chunker.py       # Text splitting + PDF extraction

│   │   ├── embeddings.py    # Ollama embedding calls

│   │   ├── vector_store.py  # ChromaDB interface

│   │   └── rag_pipeline.py  # The full RAG flow

│   ├── db/

│   │   ├── database.py      # Engine, session, get_db()

│   │   ├── models.py        # Document + QueryHistory ORM models

│   │   └── crud.py          # All database operations

│   └── schemas/

│       └── schemas.py       # Pydantic request/response models

├── tests/

│   └── test_api.py          # 18 pytest tests

├── alembic/                 # Database migrations

├── .env.example             # Config template

├── requirements.txt

└── README.md

---

## How RAG Works (for students)

**The problem:** An LLM only knows what it was trained on.
It cannot answer questions about your specific documents.

**The solution — three steps:**

RETRIEVE  Your question is converted to a vector (embedding).

ChromaDB finds the chunks from your documents that are

most similar in meaning to your question.
AUGMENT   Those chunks are injected into the prompt as context:

"Answer this question using ONLY this information: ..."
GENERATE  The LLM reads the context and writes an answer.

It's not searching — it's reading what we give it.


**Analogy:** Open-book exam. We find the right pages (retrieval).
The LLM reads those pages and writes the answer (generation).

---

## Troubleshooting

### `chromadb` fails to install on Windows

```bash
# Option 1 — use a pre-built version
pip install chromadb==0.4.24 --only-binary=:all:

# Option 2 — install C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select "Desktop development with C++" then retry
```

### `alembic: command not found`

```bash
# Use python -m prefix on Windows
python -m alembic upgrade head
```

### `Could not connect to Ollama`

```bash
# Make sure Ollama is running in a separate terminal
ollama serve

# Check models are pulled
ollama list
```

### NeonDB connection timeout

Make sure your `DATABASE_URL` ends with `?sslmode=require`.
NeonDB requires SSL and will silently timeout without it.

### Port 8000 already in use

```bash
uvicorn app.main:app --reload --port 8001
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | PostgreSQL connection string |
| `OLLAMA_BASE_URL` | ❌ | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_CHAT_MODEL` | ❌ | `llama3.2` | Model for answer generation |
| `OLLAMA_EMBEDDING_MODEL` | ❌ | `nomic-embed-text` | Model for embeddings |
| `CHROMA_PERSIST_DIRECTORY` | ❌ | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | ❌ | `datapulse_documents` | ChromaDB collection name |
| `RAG_TOP_K` | ❌ | `4` | Chunks retrieved per query |
| `CHUNK_SIZE` | ❌ | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | ❌ | `50` | Overlap between chunks |
| `DEBUG` | ❌ | `false` | Enables SQL query logging |