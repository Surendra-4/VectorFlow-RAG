# src/loaders/__init__.py

"""
Multi-format document loaders for VectorFlow-RAG.

A *loader* takes a file path and produces a :class:`LoadedDocument` —
text plus page-level provenance. Loaders are pluggable and dispatch
to the right implementation by extension / MIME type via
:class:`LoaderRegistry`.

The pipeline (``src.rag_pipeline``) is responsible for everything
downstream of loading: identity, chunking, indexing. Loaders never
import retrieval code, and retrieval code only depends on the
``LoadedDocument`` data model — keeping ingestion isolated from
indexing so future streaming / async / queued ingestion can be added
without touching loader implementations.
"""

from src.loaders.base import (
    BaseLoader,
    LoadedDocument,
    LoadedPage,
    LoaderError,
    LoaderProtocol,
)
from src.loaders.registry import LoaderRegistry, default_registry

__all__ = [
    "BaseLoader",
    "LoadedDocument",
    "LoadedPage",
    "LoaderError",
    "LoaderProtocol",
    "LoaderRegistry",
    "default_registry",
]
