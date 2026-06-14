"""
database.py — SQLAlchemy engine and session factory.

This module is the single source of truth for database connectivity.
Everything that needs a DB session imports get_db() from here.

Key concepts for students:
  - Engine    : the connection pool to PostgreSQL (created once at startup)
  - Session   : a single unit-of-work transaction (created per request)
  - Dependency: FastAPI calls get_db() for every route that needs the DB,
                and guarantees the session is closed when the request ends.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from app.config import get_settings

# ── Engine ────────────────────────────────────────────────────────────────────
# The engine manages a connection pool to PostgreSQL.
# pool_pre_ping=True tests connections before using them — this silently
# recovers from situations where PostgreSQL restarted while our app was running.
settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,          # Detects stale connections — critical for Neon's auto-suspend
    pool_recycle=300,            # Recycle connections every 5 min — Neon times out idle connections
    connect_args={"sslmode": "require"},  # Neon requires SSL
    echo=settings.debug,
)

# ── Session Factory ───────────────────────────────────────────────────────────
# SessionLocal is a factory — calling SessionLocal() gives us a new session.
# autocommit=False means we control when transactions are committed.
# autoflush=False means SQLAlchemy won't write to the DB until we say so.
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per request.

    Usage in a route:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...

    The try/finally pattern guarantees the session is always closed,
    even if an exception is raised inside the route handler.
    This prevents connection leaks that would eventually exhaust the pool.
    """
    db = SessionLocal()
    try:
        yield db          # FastAPI injects this session into the route
    finally:
        db.close()        # Always runs — even if the route raised an exception