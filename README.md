# DataPulse AI 🔍

An AI-powered document intelligence API. Upload documents, ask questions,
get answers grounded in your actual content — not LLM guesswork.

Built with FastAPI, PostgreSQL, ChromaDB, and local Ollama models.

---

## What It Does

| Endpoint | What happens |
|---|---|
| `POST /api/v1/ingest` | Upload a `.txt` or `.pdf` → chunked, embedded, stored |
| `POST /api/v1/query`  | Ask a question → RAG pipeline → grounded answer + sources |
| `GET  /api/v1/documents` | List all ingested documents |
| `GET  /api/v1/health` | Confirm the API is running |

---

## Architecture

User Question

│

▼

FastAPI Route ──► embed question (Ollama)

│

▼

ChromaDB search ──► top-K relevant chunks

│

▼

Build prompt (question + chunks)

│

▼

Ollama LLM ──► answer

│

▼

Log to PostgreSQL

│

▼

Return answer + sources

---

## Prerequisites

Install these before starting:

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | https://python.org |
| PostgreSQL | 14+ | https://postgresql.org **or** use NeonDB (cloud) |
| Ollama | latest | https://ollama.com |
| Git | any | https://git-scm.com |

---

## Setup — Step by Step

### 1. Clone the repo

```bash
git clone https://github.com/your-username/datapulse-ai.git
cd datapulse-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows (Git Bash)
source venv/Scripts/activate

# Windows (Command Prompt)
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** If `chromadb` fails to install, see the
> Troubleshooting section at the bottom of this README.

### 4. Pull Ollama models

Make sure Ollama is installed and running, then pull the two models
DataPulse AI needs:

```bash
# Start Ollama (leave this running in a separate terminal)
ollama serve

# Pull the chat model (used to generate answers)
ollama pull llama3.2

# Pull the embedding model (used to convert text to vectors)
ollama pull nomic-embed-text
```

Verify both are available:
```bash
ollama list
```

You should see both `llama3.2` and `nomic-embed-text` listed.

### 5. Configure environment variables

```bash
# Copy the example config
cp .env.example .env
```

Open `.env` and fill in your values:

```bash
# If using local PostgreSQL:
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/datapulse_db

# If using NeonDB (recommended for quick setup):
DATABASE_URL=postgresql://user:password@ep-xxxx.aws.neon.tech/neondb?sslmode=require
```

Everything else in `.env` can stay as the defaults for local development.

### 6. Create the database

**Local PostgreSQL only** — create the database first:
```bash
psql -U postgres -c "CREATE DATABASE datapulse_db;"
```

**NeonDB** — your database already exists, skip this step.

### 7. Run database migrations

```bash
# Generate the migration file from our models
alembic revision --autogenerate -m "create documents and query_history tables"

# Apply the migration — creates the tables in PostgreSQL
alembic upgrade head
```

> **Windows users:** If `alembic` is not found, use:
> `python -m alembic revision --autogenerate -m "create tables"`
> `python -m alembic upgrade head`

### 8. Start the API

```bash
uvicorn app.main:app --reload
```

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