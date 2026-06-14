"""
alembic/env.py — Alembic migration environment configuration.

Alembic uses this file to know:
  1. Which database to connect to (from our app settings)
  2. Which models to compare against (our SQLAlchemy Base)

The key addition over the default file is importing our models and
settings so Alembic can auto-generate migrations from model changes.
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import our app's config and models so Alembic knows what to migrate
from app.config import get_settings
from app.db.models import Base   # noqa: F401 — import triggers model registration
# If you add new model files, import them here too so Alembic sees them.

settings = get_settings()

# Alembic Config object — gives access to values in alembic.ini
config = context.config

# Override the sqlalchemy.url in alembic.ini with our app's setting.
# This means we only maintain ONE database URL (in .env), not two.
config.set_main_option("sqlalchemy.url", settings.database_url)

# Set up Python logging from alembic.ini config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the MetaData object Alembic inspects to detect model changes.
# It must include all models — which it does, because they all inherit Base.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL script)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No pooling for migrations — each run is one-shot
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()