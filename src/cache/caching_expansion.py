# src/cache/caching_expansion.py

"""
Cache wrapper around a :class:`QueryExpansionPipeline`.

Cache key is keyed on:

* the LLM model name used for expansion
* the *sorted* list of strategy names (order-invariant)
* a stable hash of the strategy params (n_variants, n_docs, etc.)
* the sanitized query string

A cache hit returns the exact :class:`ExpandedQuery` object that was
produced on the first call — including its errors, latency, etc. This
is the desired UX for repeat queries: deterministic retrieval that
doesn't change between sessions.

A failed expansion (no variants, no HyDE docs) is still cached so we
don't re-issue the same failing LLM calls in a tight loop. Callers
that want to retry can clear the cache or use a different model.
"""

from __future__ import annotations

from typing import Optional

from src.cache.keys import CacheKeys
from src.cache.safe import SafeCache
from src.logging_setup import get_logger
from src.query_expansion.base import ExpandedQuery
from src.query_expansion.pipeline import QueryExpansionPipeline

logger = get_logger(__name__)


class CachingExpansionPipeline:
    """Drop-in replacement for ``QueryExpansionPipeline`` with caching."""

    def __init__(
        self,
        inner: QueryExpansionPipeline,
        cache: SafeCache,
        cache_key_model: str = "",
        cache_key_params: Optional[dict] = None,
        ttl_s: Optional[int] = None,
    ):
        self._inner = inner
        self._cache = cache
        self._cache_key_model = cache_key_model
        self._cache_key_params = dict(cache_key_params or {})
        self._ttl_s = ttl_s

        # Expose the same surface the pipeline uses upstream.
        self.strategies = inner.strategies

    def expand(self, query: str) -> ExpandedQuery:
        strategy_names = [getattr(s, "name", type(s).__name__) for s in self.strategies]
        key = CacheKeys.expansion(
            self._cache_key_model,
            strategy_names,
            self._cache_key_params,
            query,
        )

        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("CachingExpansionPipeline: cache hit for query=%r", query)
            return cached

        result = self._inner.expand(query)
        self._cache.set(key, result, ttl=self._ttl_s)
        return result
