# src/api/app.py

"""
FastAPI app factory for VectorFlow-RAG.

Usage::

    # Programmatic
    from src.api import create_app
    app = create_app()

    # Via uvicorn
    uvicorn src.api.app:app --host 127.0.0.1 --port 8000

The factory:

1. Instantiates a single :class:`RAGPipeline` on app.state at startup —
   models load once per process and are shared across all requests.
2. Registers CORS, correlation-ID, and timing middleware.
3. Registers structured-error handlers (see :mod:`src.api.errors`).
4. Mounts all route modules under the ``/api/v1/`` prefix.

For tests, callers can construct an app without auto-initializing a
pipeline (``init_pipeline=False``) and inject one via dependency override.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.errors import register_error_handlers
from src.api.middleware import CorrelationIdMiddleware, TimingMiddleware
from src.config import Settings, get_settings
from src.logging_setup import configure_from_settings, get_logger

logger = get_logger(__name__)


def _apply_runtime_config_to_pipeline(pipeline, runtime_config) -> None:
    """Apply persisted runtime overrides on boot.

    Idempotent and parity-preserving: when the persisted snapshot matches the
    env baseline (fresh installs, no ``var/runtime_config.json``), this is a
    no-op and behavior is byte-identical to pre-Phase-12.
    """
    live = runtime_config.live
    # Only swap providers if it would actually change anything — otherwise the
    # default-constructed OllamaClient stays in place for tightest parity.
    current_provider = getattr(pipeline.llm, "name", "ollama")
    current_model = getattr(pipeline.llm, "model", None)
    if (live.chat.provider != current_provider) or (live.chat.model != current_model):
        try:
            pipeline.set_chat_provider(live.chat)
        except Exception as exc:
            logger.warning(
                "Could not apply persisted chat provider (%s/%s): %s — keeping default.",
                live.chat.provider, live.chat.model, exc,
            )
    try:
        pipeline.apply_live_settings(live)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not apply persisted live settings: %s — keeping defaults.", exc)


def _build_lifespan(settings: Settings, init_pipeline: bool):
    """Return an async lifespan context manager bound to the chosen settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_from_settings(settings)
        app.state.started_at = time.monotonic()
        app.state.app_name = settings.app.name
        app.state.app_version = settings.app.version
        app.state.ingest_lock = threading.Lock()
        app.state.settings = settings

        # Index manager + job registry are cheap (no model load) and useful in
        # every mode — construct them unconditionally so the index/jobs API
        # works even when a pipeline isn't auto-initialized.
        from src.indexing import IndexManager
        from src.jobs import JobRegistry

        app.state.index_manager = IndexManager()
        app.state.job_registry = JobRegistry()
        _wire_job_metrics(app.state.job_registry)

        # Ensure the users + stats tables exist (Phase 14). Best-effort: a DB
        # outage must not block the rest of the service from starting.
        try:
            from src.db.session import init_db

            init_db()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("DB init skipped/failed: %s", exc)

        if init_pipeline:
            # Lazy import — keeps app construction cheap during test setup
            # where callers always override the pipeline via DI.
            from src.rag_pipeline import RAGPipeline
            from src.runtime_config import RuntimeConfigStore

            logger.info("Initializing RAGPipeline for HTTP service…")
            app.state.pipeline = RAGPipeline(settings=settings)
            app.state.runtime_config = RuntimeConfigStore(settings=settings)
            _apply_runtime_config_to_pipeline(app.state.pipeline, app.state.runtime_config)
            logger.info("Pipeline ready (backend=%s, cache=%s)",
                        settings.vector_store.backend,
                        settings.cache.backend)
        else:
            app.state.pipeline = None
            app.state.runtime_config = None

        try:
            yield
        finally:
            logger.info("FastAPI app shutting down")
            jr = getattr(app.state, "job_registry", None)
            if jr is not None:
                jr.shutdown(wait=False)

    return lifespan


def _wire_job_metrics(job_registry) -> None:
    """Record terminal job outcomes into the metrics registry (bounded labels)."""
    def _hook(job) -> None:
        try:
            from src.observability import get_metrics

            m = get_metrics()
            m.index_jobs_total.inc(job.type, job.status.value)
            # Per-recipe build metrics — index_type is a fixed recipe enum, so
            # cardinality stays bounded (never the unbounded index name).
            if job.type == "index_build" and isinstance(job.result, dict):
                itype = str(job.result.get("index_type", "unknown"))
                m.index_builds_total.inc(itype, job.status.value)
                if job.started_at and job.finished_at:
                    m.index_build_duration_ms.observe(
                        itype, value=(job.finished_at - job.started_at) * 1000
                    )
        except Exception:  # pragma: no cover - metrics must never break jobs
            pass

    job_registry.set_metrics_hook(_hook)


def create_app(
    settings: Optional[Settings] = None,
    init_pipeline: bool = True,
) -> FastAPI:
    """
    Construct and return the FastAPI app.

    Args:
        settings: pre-built Settings. Default = singleton from get_settings().
        init_pipeline: whether to construct a RAGPipeline on startup.
            Set to False in tests that inject their own via DI override.
    """
    resolved = settings or get_settings()

    app = FastAPI(
        title=resolved.app.name,
        version=resolved.app.version,
        description="Local-first hybrid retrieval RAG platform.",
        lifespan=_build_lifespan(resolved, init_pipeline),
    )

    # CORS — local-first default; configured to permit the planned Next.js dev server.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    # Order matters: correlation ID must run before timing so timing logs include it.
    app.add_middleware(TimingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    register_error_handlers(app)

    # Routes — local imports keep the public package surface minimal.
    from src.api.routes import (
        admin,
        ask,
        auth,
        cache,
        config as config_routes,
        documents,
        health,
        indexes,
        ingest,
        jobs,
        models,
        observability,
        search,
        status,
    )

    app.include_router(health.router)  # at root, not under /api/v1/
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(status.router, prefix="/api/v1")
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(ask.router, prefix="/api/v1")
    app.include_router(cache.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(observability.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(config_routes.router, prefix="/api/v1")
    app.include_router(indexes.router, prefix="/api/v1")
    app.include_router(jobs.router, prefix="/api/v1")

    return app


# NOTE: We deliberately do NOT expose a module-level ``app = create_app()``.
# Constructing the pipeline at import time would pay the model-load cost
# during test imports and other read-the-module-and-leave scenarios.
# For uvicorn, use the factory entry point::
#
#     uvicorn --factory src.api.app:create_app --host 127.0.0.1 --port 8000
