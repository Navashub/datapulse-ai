"""
rag_pipeline.py — The centrepiece of DataPulse AI.

WHAT IS RAG? (Retrieval Augmented Generation)
  RAG solves a fundamental problem with LLMs: they only know what
  they were trained on. They cannot answer questions about YOUR documents.

  RAG fixes this in three steps:

  1. RETRIEVE  — embed the question, find the most relevant chunks
                 from ChromaDB (your documents, not the internet)

  2. AUGMENT   — build a prompt that includes those chunks as context,
                 so the LLM has the information it needs to answer

  3. GENERATE  — send the augmented prompt to the LLM and return
                 its answer to the user

  The LLM is not searching — it's READING. We do the searching.
  The LLM just reads what we give it and summarises an answer.

ANALOGY:
  Imagine an open-book exam. The student (LLM) can't remember everything,
  but we find the right pages of the textbook (retrieval) and put them
  in front of the student. They read those pages and write the answer.
  RAG is the person finding the right pages.
"""

import logging
import time
import uuid

from ollama import Client, ResponseError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.embeddings import embed_query
from app.core.vector_store import (
    get_lancedb_table,
    search_similar_chunks,
)
from app.db import crud
from app.schemas.schemas import QueryResponse, SourceChunk

logger = logging.getLogger(__name__)


def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build the prompt we send to the LLM.

    This is where RAG happens — we inject the retrieved chunks into
    the prompt as context, so the LLM answers from OUR documents,
    not from its training data.

    The prompt engineering here follows three principles:
      1. Tell the LLM exactly what role it plays
      2. Give it the context clearly labelled
      3. Tell it what to do if the context doesn't contain the answer
         (this prevents hallucination)

    Args:
        question:       The user's original question.
        context_chunks: Retrieved chunks from ChromaDB.

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    # Format each chunk with its source filename so the LLM can
    # reference where information came from
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_parts.append(
            f"[Source {i} — {chunk['filename']}]\n{chunk['text']}"
        )

    context_text = "\n\n".join(context_parts)

    # The system-style instruction is baked into the user prompt here
    # because some Ollama models respond better to single-turn prompts
    prompt = f"""You are a helpful document assistant. Answer the user's question 
using ONLY the context provided below. Do not use outside knowledge.

If the context does not contain enough information to answer the question, 
say: "I could not find a clear answer in the provided documents."

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

    return prompt


def run_rag_pipeline(
    question: str,
    db: Session,
    document_id: str | None = None,
    top_k: int | None = None,
) -> QueryResponse:
    """
    Execute the full RAG pipeline: retrieve → augment → generate → log.

    This is the core function of DataPulse AI. Every /query request
    flows through here.

    Pipeline steps:
      1. Embed the question
      2. Search ChromaDB for similar chunks
      3. Build a prompt with the retrieved context
      4. Send to Ollama LLM and get an answer
      5. Log the query to PostgreSQL
      6. Return a structured response with sources

    Args:
        question:    The user's natural language question.
        db:          SQLAlchemy session (injected by FastAPI).
        document_id: Optional — scope search to one document.
        top_k:       Number of chunks to retrieve (falls back to config default).

    Returns:
        QueryResponse with answer, sources, and metadata.

    Raises:
        RuntimeError: If Ollama is unreachable or the model is not available.
        ValueError:   If no chunks are found (empty vector store).
    """
    settings = get_settings()
    top_k = top_k or settings.rag_top_k
    start_time = time.perf_counter()  # High-precision timer for latency tracking

    # ── STEP 1: EMBED THE QUESTION ────────────────────────────────────────────
    # Turn the question into a vector so we can compare it to stored chunk vectors
    logger.info(f"RAG pipeline started | question='{question[:60]}...'")
    query_vector = embed_query(question)

    # ── STEP 2: RETRIEVE RELEVANT CHUNKS ─────────────────────────────────────
    # Ask ChromaDB: "which stored chunks are most similar to this question?"
    table = get_lancedb_table()
    retrieved_chunks = search_similar_chunks(
        table,
        query_embedding=query_vector,
        top_k=top_k,
        document_id=document_id,
    )

    if not retrieved_chunks:
        # No chunks means either the vector store is empty or the
        # document_id filter returned nothing — tell the user clearly
        raise ValueError(
            "No relevant content found. "
            "Make sure documents have been ingested before querying."
        )

    logger.info(f"Retrieved {len(retrieved_chunks)} chunks from ChromaDB")

    # ── STEP 3: BUILD THE AUGMENTED PROMPT ───────────────────────────────────
    # Inject the retrieved chunks into the prompt as context
    prompt = build_rag_prompt(question, retrieved_chunks)

    # ── STEP 4: GENERATE AN ANSWER ────────────────────────────────────────────
    # Send the augmented prompt to the local Ollama LLM
    ollama_client = Client(host=settings.ollama_base_url)

    try:
        logger.info(f"Sending prompt to Ollama model: {settings.ollama_chat_model}")
        response = ollama_client.chat(
            model=settings.ollama_chat_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        answer = response["message"]["content"].strip()

    except ResponseError as e:
        raise RuntimeError(
            f"Ollama chat failed. "
            f"Make sure you have run: ollama pull {settings.ollama_chat_model}\n"
            f"Error: {e}"
        ) from e

    except Exception as e:
        raise RuntimeError(
            f"Could not connect to Ollama at {settings.ollama_base_url}. "
            f"Make sure Ollama is running: ollama serve\n"
            f"Error: {e}"
        ) from e

    # ── STEP 5: CALCULATE RESPONSE TIME ──────────────────────────────────────
    elapsed = time.perf_counter() - start_time
    query_id = str(uuid.uuid4())

    logger.info(f"RAG pipeline complete | time={elapsed:.2f}s | query_id={query_id}")

    # ── STEP 6: LOG TO POSTGRESQL ─────────────────────────────────────────────
    # Store every query for auditability and future analytics
    crud.create_query_history(
        db,
        query_id=query_id,
        question=question,
        answer=answer,
        document_id=document_id,
        chunks_retrieved=len(retrieved_chunks),
        response_time_seconds=elapsed,
    )

    # ── STEP 7: BUILD AND RETURN STRUCTURED RESPONSE ──────────────────────────
    # Map retrieved chunks to the SourceChunk schema so callers can see
    # exactly which parts of which documents produced the answer
    sources = [
        SourceChunk(
            chunk_index=chunk["chunk_index"],
            text=chunk["text"],
            document_id=chunk["document_id"],
            filename=chunk["filename"],
        )
        for chunk in retrieved_chunks
    ]

    return QueryResponse(
        query_id=query_id,
        question=question,
        answer=answer,
        sources=sources,
        chunks_retrieved=len(retrieved_chunks),
        response_time_seconds=round(elapsed, 3),
        model_used=settings.ollama_chat_model,
    )