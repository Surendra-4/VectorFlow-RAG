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
from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request, status

if TYPE_CHECKING:
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
