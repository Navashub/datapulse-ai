"""
query.py — POST /query endpoint.

This route receives a natural language question, runs it through
the RAG pipeline, and returns an answer grounded in the ingested documents.

This is the READ side of DataPulse AI — and the most impressive
part to demo. The user asks a question in plain English and gets
a cited, document-grounded answer back in seconds.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.rag_pipeline import run_rag_pipeline
from app.db.database import get_db
from app.schemas.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query your documents",
    description=(
        "Ask a natural language question. The RAG pipeline retrieves "
        "relevant chunks from your ingested documents and generates "
        "a grounded answer using a local Ollama model."
    ),
)
async def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
) -> QueryResponse:
    """
    Run the RAG pipeline for a user question.

    The heavy lifting is all in rag_pipeline.py — this route just:
      1. Validates the incoming request (Pydantic handles this)
      2. Calls the pipeline
      3. Handles any errors with clear HTTP status codes
      4. Returns the structured response

    Keeping routes thin like this makes them easy to read and test.

    Args:
        request: Validated QueryRequest (question, optional document_id, top_k).
        db:      PostgreSQL session for logging query history.

    Returns:
        QueryResponse with answer, sources, and performance metadata.

    Raises:
        404: No relevant content found (empty vector store or bad document_id).
        500: Ollama unreachable or pipeline failure.
    """
    logger.info(
        f"Query received: '{request.question[:60]}' "
        f"| doc_id={request.document_id} | top_k={request.top_k}"
    )

    try:
        response = run_rag_pipeline(
            question=request.question,
            db=db,
            document_id=request.document_id,
            top_k=request.top_k,
        )
        return response

    except ValueError as e:
        # ValueError from the pipeline means no chunks found —
        # this is a user error (nothing ingested), not a server error
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except RuntimeError as e:
        # RuntimeError means Ollama is down or model not pulled —
        # a server-side infrastructure problem
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    except Exception as e:
        logger.error(f"Unexpected pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}",
        )