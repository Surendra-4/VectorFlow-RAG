# src/indexing/__init__.py

"""
Named index management for VectorFlow-RAG (Phase 12e+).

Turns vector indexes into first-class, named, persisted runtime entities you
can create, list, switch between, benchmark, export and import.

Public surface:

* :class:`IndexProfile` / :class:`CompatibilitySignature`
* :class:`IndexRegistry` / :func:`validate_index_name` / :class:`IndexRegistryError`
* :class:`IndexManager`
"""

from src.indexing.manager import IndexManager
from src.indexing.profile import CompatibilitySignature, IndexProfile
from src.indexing.registry import (
    IndexRegistry,
    IndexRegistryError,
    validate_index_name,
)

__all__ = [
    "CompatibilitySignature",
    "IndexManager",
    "IndexProfile",
    "IndexRegistry",
    "IndexRegistryError",
    "validate_index_name",
]
