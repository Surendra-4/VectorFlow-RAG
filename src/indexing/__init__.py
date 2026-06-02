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

from src.indexing.benchmark import (
    BenchmarkResult,
    benchmark_recipes,
    evaluate_store,
    exact_neighbors,
    persist_benchmark,
)
from src.indexing.compatibility import (
    Action,
    CompatibilityIssue,
    CompatibilityReport,
    IndexTargetConfig,
    Severity,
    check_compatibility,
    target_from_index_settings,
)
from src.indexing.manager import IndexManager
from src.indexing.profile import CompatibilitySignature, IndexProfile
from src.indexing.recipes import (
    RECIPES,
    RecipeError,
    RecipeSpec,
    build_factory_string,
    estimate,
    get_recipe,
    list_recipes,
    resolve_params,
    validate_recipe,
)
from src.indexing.registry import (
    IndexRegistry,
    IndexRegistryError,
    validate_index_name,
)

__all__ = [
    "Action",
    "BenchmarkResult",
    "CompatibilityIssue",
    "CompatibilityReport",
    "CompatibilitySignature",
    "benchmark_recipes",
    "evaluate_store",
    "exact_neighbors",
    "persist_benchmark",
    "IndexManager",
    "IndexProfile",
    "IndexRegistry",
    "IndexRegistryError",
    "IndexTargetConfig",
    "Severity",
    "check_compatibility",
    "target_from_index_settings",
    "RECIPES",
    "RecipeError",
    "RecipeSpec",
    "build_factory_string",
    "estimate",
    "get_recipe",
    "list_recipes",
    "resolve_params",
    "validate_index_name",
    "validate_recipe",
]
