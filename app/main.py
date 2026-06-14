"""
main.py — FastAPI application factory.

This is the entry point of DataPulse AI. It:
  1. Creates the FastAPI app instance
  2. Registers all routers (routes)
  3. Sets up logging
  4. Adds startup/shutdown lifecycle hooks
  5. Provides the ASGI app object that uvicorn serves

PATTERN — Application Factory:
  We define the app here but never import it in other modules.
  Other modules import from config, db, core — never from main.
  This keeps the dependency graph clean and makes testing easy.
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import health, ingest, query, documents

# ── Logging Setup ─────────────────────────────────────────────────────────────
# Configure logging once at startup — all modules use logging.getLogger(__name__)
# which flows up to this root configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to console
    ],
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ── App Instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered document intelligence API. "
        "Ingest documents, then ask questions about them in plain English. "
        "Powered by local Ollama models and ChromaDB vector search."
    ),
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc UI at /redoc
)


# ── CORS Middleware ───────────────────────────────────────────────────────────
# CORS allows browser-based frontends to call this API.
# allow_origins=["*"] is fine for local development — restrict
# this to specific domains in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lifecycle Events ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    """
    Runs once when the server starts.

    Good place to:
      - Validate that required services are reachable (Ollama, DB)
      - Pre-warm connections
      - Log startup confirmation
    """
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info(f"  Ollama URL    : {settings.ollama_base_url}")
    logger.info(f"  Chat model    : {settings.ollama_chat_model}")
    logger.info(f"  Embed model   : {settings.ollama_embedding_model}")
    logger.info(f"  ChromaDB path : {settings.chroma_persist_directory}")
    logger.info(f"  Debug mode    : {settings.debug}")
    logger.info("=" * 60)
    logger.info("DataPulse AI is ready. Visit http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Runs once when the server shuts down — good place to flush logs."""
    logger.info("DataPulse AI shutting down. Goodbye.")


# ── Router Registration ───────────────────────────────────────────────────────
# All routes are prefixed with /api/v1 — versioning from day one
# means we can ship /api/v2 later without breaking existing clients.
API_PREFIX = "/api/v1"

app.include_router(health.router, prefix=API_PREFIX, tags=["Health"])
app.include_router(ingest.router, prefix=API_PREFIX, tags=["Ingest"])
app.include_router(query.router,  prefix=API_PREFIX, tags=["Query"])
app.include_router(documents.router, prefix=API_PREFIX, tags=["Documents"])


# ── Root Redirect ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the API docs — friendlier than a 404."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")