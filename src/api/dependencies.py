# src/api/dependencies.py

"""
Dependency-injection helpers for the API.

The pipeline is a **process-wide singleton** held on ``app.state.pipeline``.
``get_pipeline()`` is the FastAPI dependency endpoints use; in tests we
override it to inject a pre-ingested test pipeline.

Concurrency model:

* search/ask/status reads: lock-free (the pipeline is read-only after
  ingestion)
* ingestion writes: serialized by ``app.state.ingest_lock`` because
  ``RAGPipeline.ingest_*`` mutates the vector store, BM25 index, and
  corpus fingerprint
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

from fastapi import Depends, HTTPException, Request, status

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from src.db.models import User
    from src.rag_pipeline import RAGPipeline


def get_pipeline(request: Request) -> "RAGPipeline":
    """Return the process-wide RAGPipeline singleton.

    Raises 503 if the pipeline isn't initialized yet (cold start window).
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not yet initialized",
        )
    return pipeline


def get_ingest_lock(request: Request) -> threading.Lock:
    """Return the process-wide ingestion lock."""
    lock = getattr(request.app.state, "ingest_lock", None)
    if lock is None:
        # Defensive — should always be set by the lifespan handler.
        lock = threading.Lock()
        request.app.state.ingest_lock = lock
    return lock


def get_request_id(request: Request) -> str:
    """Correlation ID set by ``CorrelationIdMiddleware``."""
    return getattr(request.state, "request_id", "unknown")


def get_started_at(request: Request) -> float:
    """Monotonic start time for the app process — used for uptime reporting."""
    return getattr(request.app.state, "started_at", 0.0)


def get_runtime_config(request: Request):
    """Return the process-wide ``RuntimeConfigStore`` (Phase 12c).

    Raises 503 if the store isn't initialized (cold start / DI override path).
    """
    store = getattr(request.app.state, "runtime_config", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Runtime config not initialized",
        )
    return store


def get_index_manager(request: Request):
    """Return the process-wide ``IndexManager`` (Phase 12e/h)."""
    im = getattr(request.app.state, "index_manager", None)
    if im is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index manager not initialized",
        )
    return im


def get_job_registry(request: Request):
    """Return the process-wide ``JobRegistry`` (Phase 12h)."""
    jr = getattr(request.app.state, "job_registry", None)
    if jr is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job registry not initialized",
        )
    return jr


# --------------------------------------------------------------------------- #
# Auth (Phase 14)
# --------------------------------------------------------------------------- #


def get_db_session():
    """FastAPI dependency yielding a DB session (re-exported for routes)."""
    from src.db.session import get_db

    yield from get_db()


def get_optional_user(request: Request) -> "Optional[User]":
    """Return the authenticated user from a Bearer JWT, or None.

    Lazy by design: the anonymous hot path (search/ask with no token) never
    opens a DB session. When a token is present we load the user in a
    short-lived session; ``expire_on_commit=False`` keeps scalar attributes
    usable after it closes (the returned instance is detached — read its ``id``
    / ``to_public()``; routes that mutate re-fetch by id). Never raises.
    """
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    from src.auth.security import decode_token

    payload = decode_token(header[len("Bearer "):].strip())
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    try:
        from src.db.models import User
        from src.db.session import get_sessionmaker

        db = get_sessionmaker()()
        try:
            return db.get(User, user_id)
        finally:
            db.close()
    except Exception:  # pragma: no cover - auth lookup must never 500 a request
        return None


def get_current_user(user: "Optional[User]" = Depends(get_optional_user)) -> "User":
    """Require a valid authenticated user (401 otherwise)."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_user_if_enabled(
    user: "Optional[User]" = Depends(get_optional_user),
) -> "Optional[User]":
    """Enforce auth only when ``settings.auth.required`` is True (production).

    Local/dev/tests (required=False) stay open and simply attribute stats when a
    token is present. Production flips one flag to gate the data endpoints.
    """
    from src.config import get_settings

    if get_settings().auth.required and user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
