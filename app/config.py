"""
config.py — Centralised application configuration.

All settings are read from environment variables (or a .env file).
Using pydantic-settings means:
  1. Every setting is type-checked at startup — the app refuses to start
     with bad config rather than failing mysteriously at runtime.
  2. There is ONE place to look for every configurable value.
  3. Secrets never appear in source code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    pydantic-settings automatically reads from a .env file in the
    working directory, then from actual environment variables
    (environment variables take priority — important for production).
    """

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "DataPulse AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    # Full connection URL format:
    # postgresql://username:password@host:port/database_name
    database_url: str

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    # Directory where ChromaDB will persist its vector data on disk.
    # ChromaDB creates this folder automatically if it doesn't exist.
    chroma_persist_directory: str = "./chroma_db"

    # Name of the collection inside ChromaDB where we store our embeddings.
    # Think of it like a table name inside the vector database.
    chroma_collection_name: str = "datapulse_documents"

    # ── Ollama ────────────────────────────────────────────────────────────────
    # Ollama runs a local REST API server (default: http://localhost:11434).
    # The SDK uses this base URL to communicate with your local Ollama instance.
    ollama_base_url: str = "http://localhost:11434"

    # The Ollama model to use for generating answers (chat/generation).
    # Students can change this to any model they have pulled locally,
    # e.g. "llama3.1:8b" for higher quality, "phi3" for speed.
    ollama_chat_model: str = "llama3.2"

    # The Ollama model to use for creating embeddings (turning text into vectors).
    # nomic-embed-text is purpose-built for embeddings — much better than using
    # a chat model for this task.
    ollama_embedding_model: str = "nomic-embed-text"

    # ── RAG Pipeline Tuning ───────────────────────────────────────────────────
    # How many text chunks to retrieve from ChromaDB before sending to the LLM.
    # Higher = more context, but also more tokens for the LLM to process.
    rag_top_k: int = 4

    # Maximum number of characters per text chunk when splitting documents.
    # Smaller chunks = more precise retrieval. Larger chunks = more context per result.
    chunk_size: int = 500

    # How many characters consecutive chunks share.
    # Overlap prevents answers being cut off at chunk boundaries.
    chunk_overlap: int = 50

    # Tell pydantic-settings to load from a .env file automatically.
    # extra="ignore" means unknown variables in .env don't cause errors —
    # useful when students add their own variables for experimentation.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    @lru_cache ensures Settings() is only instantiated ONCE, no matter
    how many times get_settings() is called across the app. This is
    important because each instantiation reads and validates the .env file.

    FastAPI's dependency injection system calls this function, so every
    route that needs config just declares: settings = Depends(get_settings)
    """
    return Settings()