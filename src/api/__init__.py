# src/api/__init__.py

"""
FastAPI service layer for VectorFlow-RAG.

Public surface:

* :func:`src.api.app.create_app` — FastAPI application factory
* :mod:`src.api.schemas` — versioned Pydantic request/response models
* :mod:`src.api.dependencies` — DI helpers (pipeline singleton, request context)

All endpoints live under ``/api/v1/`` except ``/health``. Forward-compat
note: schema additions are non-breaking; schema removals or renames
require bumping to ``/api/v2/``.
"""

from src.api.app import create_app

__all__ = ["create_app"]
