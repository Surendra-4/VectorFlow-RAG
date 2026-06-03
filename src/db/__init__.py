# src/db/__init__.py

"""
Relational database layer for users + per-user statistics (Phase 14).

Stores ONLY accounts and usage stats — never ingested documents/chunks/indexes.

Public surface:

* :class:`Base` — declarative base
* :class:`User` / :class:`UserStats` — ORM models
* :func:`get_db` / :func:`session_scope` / :func:`init_db` / :func:`reset_engine`
"""

from src.db.base import Base
from src.db.models import STAT_FIELDS, User, UserStats
from src.db.session import (
    get_db,
    get_engine,
    init_db,
    reset_engine,
    resolve_database_url,
    session_scope,
)

__all__ = [
    "Base",
    "STAT_FIELDS",
    "User",
    "UserStats",
    "get_db",
    "get_engine",
    "init_db",
    "reset_engine",
    "resolve_database_url",
    "session_scope",
]
