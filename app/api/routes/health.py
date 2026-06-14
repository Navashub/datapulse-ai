"""
health.py — GET /health endpoint.

A health check endpoint is standard practice in any production API.
It lets load balancers, monitoring tools, and developers instantly
verify the service is alive without triggering any business logic.
"""

import logging
from fastapi import APIRouter, Depends
from app.config import get_settings, Settings
from app.schemas.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current health status of the DataPulse AI API.",
)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """
    Confirm the API is running and return basic app info.

    This route has no DB or Ollama calls — it should always respond
    instantly. If this endpoint is slow, the problem is the server
    itself, not a dependency.
    """
    logger.debug("Health check called")
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        message="DataPulse AI is running. Ready to ingest and query documents.",
    )