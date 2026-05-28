# src/query_expansion/pipeline.py

"""
Compose multiple expansion strategies into a single :class:`ExpandedQuery`.

The pipeline is intentionally dumb: it runs each registered strategy
serially, collects the results, and merges them. There's no clever
ordering or fall-through — that's the strategies' responsibility.

Strategy failures are non-fatal: the original query is always part of
``ExpandedQuery.queries``, so retrieval always has at least one usable
query string even if every strategy errored.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

from src.logging_setup import get_logger
from src.query_expansion.base import (
    ExpandedQuery,
    ExpansionResult,
    ExpansionStrategy,
    dedupe_preserve_order,
    sanitize_query,
)

logger = get_logger(__name__)


class QueryExpansionPipeline:
    """Run an ordered list of :class:`ExpansionStrategy` instances."""

    def __init__(
        self,
        strategies: Sequence[ExpansionStrategy],
        max_query_length: int = 500,
    ):
        self.strategies = list(strategies)
        self.max_query_length = max_query_length

    def expand(self, query: str) -> ExpandedQuery:
        sanitized = sanitize_query(query, max_length=self.max_query_length)
        # The sanitized original is always the first retrieval query, even
        # if every strategy fails — retrieval must remain functional.
        queries: List[str] = [sanitized] if sanitized else []
        hyde_docs: List[str] = []
        strategies_used: List[str] = []
        errors: List[str] = []

        start = time.perf_counter()
        for strategy in self.strategies:
            try:
                result: ExpansionResult = strategy.expand(sanitized)
            except Exception as exc:
                # Defensive: even if a strategy raises (shouldn't, but),
                # the pipeline must keep going.
                logger.exception("Expansion strategy %s raised", strategy.name)
                errors.append(f"{strategy.name}: {type(exc).__name__}")
                continue

            if result.error:
                errors.append(f"{result.strategy}: {result.error}")
                logger.debug("Strategy %s errored: %s", result.strategy, result.error)
                continue

            strategies_used.append(result.strategy)
            if result.queries:
                queries.extend(result.queries)
            if result.hyde_documents:
                hyde_docs.extend(result.hyde_documents)

        elapsed_ms = (time.perf_counter() - start) * 1000

        unique_queries = dedupe_preserve_order(queries)
        unique_hyde = dedupe_preserve_order(hyde_docs)

        return ExpandedQuery(
            original=sanitized,
            queries=tuple(unique_queries),
            hyde_documents=tuple(unique_hyde),
            strategies_used=tuple(strategies_used),
            errors=tuple(errors),
            expansion_latency_ms=elapsed_ms,
        )
