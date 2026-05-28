# src/query_expansion/__init__.py

"""
Query expansion for VectorFlow-RAG.

Two strategies are shipped in Phase 5:

* :class:`MultiQueryExpander` — LLM rewrites the query into N variants;
  all variants are used as retrieval queries, fused via RRF.
* :class:`HyDEExpander` — LLM writes a hypothetical answer document;
  the document's embedding is used as an additional vector-side query.

Strategies share a single :class:`ExpansionStrategy` Protocol. The
:class:`QueryExpansionPipeline` composes any combination of strategies
into a single :class:`ExpandedQuery` object that the retrieval pipeline
consumes.

All strategies:

* are local-LLM only (Ollama)
* have a wall-clock timeout — slow LLMs never block retrieval
* contain failures (timeout / parse error / empty output) and produce
  empty :class:`ExpansionResult` with ``error`` populated, so the
  caller falls back to the original query cleanly
* sanitize input length before prompt construction (prompt-injection
  defense in depth)
"""

from src.query_expansion.base import (
    ExpandedQuery,
    ExpansionResult,
    ExpansionStrategy,
    dedupe_preserve_order,
    sanitize_query,
)
from src.query_expansion.hyde import HyDEExpander
from src.query_expansion.multi_query import MultiQueryExpander
from src.query_expansion.pipeline import QueryExpansionPipeline

__all__ = [
    "ExpandedQuery",
    "ExpansionResult",
    "ExpansionStrategy",
    "HyDEExpander",
    "MultiQueryExpander",
    "QueryExpansionPipeline",
    "dedupe_preserve_order",
    "sanitize_query",
]
