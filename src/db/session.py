# src/db/session.py

"""
SQLAlchemy engine + session management (Phase 14).

The engine is built lazily from the resolved database URL:

* ``$DATABASE_URL`` (Render/Heroku style) takes precedence — its ``postgres://``
  scheme is normalized to ``postgresql+psycopg2://`` for SQLAlchemy 2.0.
* otherwise ``settings.database.url`` (default: a local SQLite file).

SQLite gets ``check_same_thread=False`` so FastAPI's threadpool can share the
connection. ``reset_engine()`` exists so tests can point at a temp database.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.db.base import Base
from src.logging_setup import get_logger

logger = get_logger(__name__)

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def resolve_database_url() -> str:
    raw = os.environ.get("DATABASE_URL") or get_settings().database.url
    if raw.startswith("postgres://"):
        raw = "postgresql+psycopg2://" + raw[len("postgres://"):]
    elif raw.startswith("postgresql://"):
        raw = "postgresql+psycopg2://" + raw[len("postgresql://"):]
    return raw


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        url = resolve_database_url()
        kwargs: dict = {"pool_pre_ping": True, "echo": get_settings().database.echo}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            # Ensure the parent dir for a file-based SQLite DB exists.
            if ":///" in url and not url.endswith(":memory:"):
                db_path = url.split(":///", 1)[1]
                if db_path and db_path != ":memory:":
                    Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(url, **kwargs)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
        logger.info("DB engine ready (%s)", url.split("@")[-1] if "@" in url else url)
    return _engine


def get_sessionmaker() -> sessionmaker:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def init_db() -> None:
    """Create tables if they don't exist. Idempotent."""
    from src.db import models  # noqa: F401 — ensure models register on Base

    Base.metadata.create_all(get_engine())
    logger.info("DB schema ensured (users, user_stats)")


def get_db() -> Iterator[Session]:
    """FastAPI dependency — yields a session, always closed."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for non-request code (scripts, OAuth callbacks)."""
    db = get_sessionmaker()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reset_engine() -> None:
    """Drop the cached engine — for tests that switch databases."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
