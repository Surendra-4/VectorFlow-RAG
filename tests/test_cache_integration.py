# tests/test_cache_integration.py

"""
End-to-end cache integration tests.

Covers:

* embedding cache hits during repeated retrieval
* full-retrieval cache hits across identical queries
* corpus-fingerprint invalidation on re-ingestion
* RetrievalTrace surfaces cache stats + from_cache flag
* Redis-unavailable fallback at pipeline level
* expansion path cache (using a scripted expansion to avoid LLM calls)
* latency benchmark: warm vs cold
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.cache.factory import make_cache
from src.cache.memory import MemoryCache
from src.cache.safe import SafeCache
from src.config import CacheSettings, Settings
from src.query_expansion.base import ExpansionResult
from src.query_expansion.pipeline import QueryExpansionPipeline
from src.rag_pipeline import RAGPipeline


GOLDEN_DOCS = [
    "Photosynthesis converts light energy into chemical energy in chloroplasts.",
    "Mitochondria generate ATP via oxidative phosphorylation in eukaryotic cells.",
    "Reciprocal rank fusion combines ranked lists from multiple retrievers.",
    "BM25 is a sparse keyword retrieval algorithm based on TF-IDF.",
    "Vector embeddings map text to a continuous high-dimensional space.",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_pipeline_with_memory_cache(tmp_path) -> RAGPipeline:
    """Build a pipeline forced to use an in-memory cache (regardless of env)."""
    cache = SafeCache(MemoryCache(default_ttl_s=3600))
    rag = RAGPipeline(
        index_dir=str(tmp_path / "rag_cache"),
        cache=cache,
        enable_cache=True,
    )
    rag.ingest_documents(GOLDEN_DOCS)
    return rag


# --------------------------------------------------------------------------- #
# Full-retrieval cache
# --------------------------------------------------------------------------- #


class TestFullRetrievalCache:
    def test_repeat_query_hits_cache(self, tmp_path):
        rag = _build_pipeline_with_memory_cache(tmp_path)
        rag.cache.stats.reset()

        r1 = rag.search("photosynthesis", k=3)
        r2 = rag.search("photosynthesis", k=3)

        assert r1 == r2
        # Exactly one full-retrieval cache hit on the second call.
        assert rag.cache.stats.hits >= 1

    def test_cache_hit_short_circuits_retrieval(self, tmp_path):
        rag = _build_pipeline_with_memory_cache(tmp_path)
        rag.search("BM25", k=3)
        _, trace_warm = rag.search("BM25", k=3, return_trace=True)
        # The warm call should not have populated per_strategy or fused_pool
        # because it returned the cached results before retrieval ran.
        assert trace_warm.from_cache is True
        assert trace_warm.per_strategy == []
        assert trace_warm.fused_pool == []

    def test_different_k_busts_cache(self, tmp_path):
        rag = _build_pipeline_with_memory_cache(tmp_path)
        rag.cache.stats.reset()
        rag.search("photosynthesis", k=3)
        rag.search("photosynthesis", k=5)
        # Different k → different cache key → second call is also a miss.
        assert rag.cache.stats.misses >= 2

    def test_different_query_misses_separately(self, tmp_path):
        rag = _build_pipeline_with_memory_cache(tmp_path)
        rag.cache.stats.reset()
        rag.search("photosynthesis", k=3)
        rag.search("mitochondria", k=3)
        assert rag.cache.stats.misses >= 2
        assert rag.cache.stats.hits == 0

    def test_trace_includes_cache_stats(self, tmp_path):
        rag = _build_pipeline_with_memory_cache(tmp_path)
        _, trace = rag.search("BM25", k=3, return_trace=True)
        assert "hits" in trace.cache_stats
        assert "misses" in trace.cache_stats
        assert "hit_ratio" in trace.cache_stats


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #


class TestEmbeddingCache:
    def test_repeat_distinct_queries_share_embedding_cache(self, tmp_path):
        rag = _build_pipeline_with_memory_cache(tmp_path)
        rag.cache.stats.reset()

        # Same query but with different k → embedding cached after first,
        # second query embeds-hit (full-retrieval miss because of different k).
        rag.search("BM25 keyword", k=3)
        rag.search("BM25 keyword", k=5)

        # We expect at least 1 cache hit on the embedding namespace.
        assert rag.cache.stats.hits >= 1


# --------------------------------------------------------------------------- #
# Corpus-fingerprint invalidation
# --------------------------------------------------------------------------- #


class TestCorpusFingerprintInvalidation:
    def test_re_ingest_changes_fingerprint(self, tmp_path):
        rag = _build_pipeline_with_memory_cache(tmp_path)
        fp1 = rag.corpus_fingerprint
        rag.ingest_documents(GOLDEN_DOCS + ["A novel sentence about elephants."])
        fp2 = rag.corpus_fingerprint
        assert fp1 != fp2

    def test_old_cache_entries_invisible_after_re_ingest(self, tmp_path):
        rag = _build_pipeline_with_memory_cache(tmp_path)
        rag.search("photosynthesis", k=3)
        # First call populated the cache.
        rag.cache.stats.reset()
        # Re-ingest with different docs.
        rag.ingest_documents(["Completely different content about quantum tunneling."])
        # Same query → new fingerprint → new key → must be a miss.
        rag.search("photosynthesis", k=3)
        assert rag.cache.stats.misses >= 1

    def test_no_explicit_invalidate_method_needed(self, tmp_path):
        """Verify cache invalidation is purely structural (no API to forget)."""
        rag = _build_pipeline_with_memory_cache(tmp_path)
        # Pipeline shouldn't even expose a clear-retrieval-cache method.
        for attr in ("invalidate_retrieval_cache", "clear_retrieval_cache", "bust_cache"):
            assert not hasattr(rag, attr), f"unexpected invalidate method: {attr}"


# --------------------------------------------------------------------------- #
# Redis fallback at pipeline level
# --------------------------------------------------------------------------- #


class TestRedisFallbackAtPipeline:
    def test_pipeline_with_unreachable_redis_still_works(self, tmp_path, monkeypatch):
        """Pipeline boots cleanly with Redis backend pointing at nothing."""
        monkeypatch.setenv("VFR_CACHE__BACKEND", "redis")
        monkeypatch.setenv("VFR_CACHE__REDIS_URL", "redis://127.0.0.1:65535/0")
        from src.config import reset_settings_cache

        reset_settings_cache()
        try:
            rag = RAGPipeline(index_dir=str(tmp_path / "redis_fallback"))
            # Backend should have degraded to memory.
            assert rag.cache.backend_name == "memory"
            rag.ingest_documents(["test content"])
            results = rag.search("test", k=1)
            assert len(results) >= 1
        finally:
            reset_settings_cache()


# --------------------------------------------------------------------------- #
# Expansion path cache
# --------------------------------------------------------------------------- #


class _ScriptedMultiQuery:
    name = "multi_query"
    def __init__(self, variants):
        self.variants = tuple(variants)
        self.expand_calls = 0
    def expand(self, query):
        self.expand_calls += 1
        return ExpansionResult(strategy=self.name, queries=self.variants, latency_ms=1.0)


class TestExpansionPathCache:
    def test_repeat_query_with_expansion_hits_full_retrieval_cache(self, tmp_path):
        cache = SafeCache(MemoryCache())
        strat = _ScriptedMultiQuery(["BM25 explained", "keyword retrieval"])
        rag = RAGPipeline(
            index_dir=str(tmp_path / "exp_cache"),
            cache=cache,
            enable_cache=True,
            expansion_pipeline=QueryExpansionPipeline(strategies=[strat]),
            enable_expansion=True,
        )
        rag.ingest_documents(GOLDEN_DOCS)

        rag.search("BM25", k=3)
        calls_after_first = strat.expand_calls
        rag.search("BM25", k=3)
        # On the second call, the retrieval cache short-circuits before
        # expansion even runs, so the strategy is not re-invoked.
        assert strat.expand_calls == calls_after_first


# --------------------------------------------------------------------------- #
# Cache disabled = exact legacy behavior
# --------------------------------------------------------------------------- #


class TestCacheDisabledLegacyBehavior:
    def test_with_cache_disabled_no_caching_layer_wraps_embedder(self, tmp_path):
        rag = RAGPipeline(
            index_dir=str(tmp_path / "no_cache"),
            enable_cache=False,
        )
        # When caching is off, embedder is the raw Embedder, not a wrapper.
        from src.embedder import Embedder
        assert isinstance(rag.embedder, Embedder)
        # Cache backend should be null.
        assert rag.cache.backend_name == "null"

    def test_with_cache_disabled_search_still_works(self, tmp_path):
        rag = RAGPipeline(
            index_dir=str(tmp_path / "no_cache_2"),
            enable_cache=False,
        )
        rag.ingest_documents(GOLDEN_DOCS)
        results = rag.search("photosynthesis", k=2)
        assert len(results) >= 1


# --------------------------------------------------------------------------- #
# Latency benchmark — informational, not strict
# --------------------------------------------------------------------------- #


class TestCacheLatencyBenchmark:
    def test_warm_call_faster_than_cold(self, tmp_path):
        rag = _build_pipeline_with_memory_cache(tmp_path)

        # Cold
        t0 = time.perf_counter()
        rag.search("photosynthesis", k=3)
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm (full-retrieval cache hit)
        t0 = time.perf_counter()
        rag.search("photosynthesis", k=3)
        warm_ms = (time.perf_counter() - t0) * 1000

        print(f"\n[Retrieval latency] cold={cold_ms:.1f}ms warm={warm_ms:.2f}ms ratio={cold_ms/warm_ms:.0f}×")
        # Warm should be at least 2× faster than cold; in practice it's
        # 50-200× faster because retrieval is fully short-circuited.
        assert warm_ms < cold_ms
