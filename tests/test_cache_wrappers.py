# tests/test_cache_wrappers.py

"""Tests for the cache wrappers: CachingEmbedder and CachingExpansionPipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.cache.caching_embedder import CachingEmbedder
from src.cache.caching_expansion import CachingExpansionPipeline
from src.cache.factory import make_cache
from src.cache.memory import MemoryCache
from src.cache.safe import SafeCache
from src.config import CacheSettings
from src.query_expansion.base import ExpandedQuery, ExpansionResult
from src.query_expansion.pipeline import QueryExpansionPipeline


# --------------------------------------------------------------------------- #
# Mock embedder for fast tests
# --------------------------------------------------------------------------- #


class _MockEmbedder:
    """Deterministic, fast embedder. Records call count to detect cache hits."""

    model_name = "mock-embedder"
    dimension = 4

    def __init__(self):
        self.encode_calls = 0
        self.texts_seen: list[list[str]] = []

    def encode(self, texts, batch_size=32, show_progress=True, input_type=None):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        self.encode_calls += 1
        self.texts_seen.append(list(texts))
        self.last_input_type = input_type
        # Deterministic embedding: pad first 4 char codes / 1000.
        embs = []
        for t in texts:
            v = [ord(c) / 1000.0 for c in (t + "\x00\x00\x00\x00")[:4]]
            embs.append(v)
        return np.asarray(embs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# CachingEmbedder
# --------------------------------------------------------------------------- #


class TestCachingEmbedder:
    @pytest.fixture
    def cache(self):
        return SafeCache(MemoryCache())

    @pytest.fixture
    def setup(self, cache):
        inner = _MockEmbedder()
        return inner, cache, CachingEmbedder(inner, cache)

    def test_shape_preserved(self, setup):
        _, _, wrapped = setup
        out = wrapped.encode(["alpha", "beta", "gamma"], show_progress=False)
        assert out.shape == (3, 4)

    def test_first_call_misses_then_caches(self, setup):
        inner, cache, wrapped = setup
        wrapped.encode(["alpha"], show_progress=False)
        assert inner.encode_calls == 1
        assert cache.stats.misses == 1
        assert cache.stats.sets == 1

    def test_second_call_hits_cache(self, setup):
        inner, cache, wrapped = setup
        wrapped.encode(["alpha"], show_progress=False)
        wrapped.encode(["alpha"], show_progress=False)
        # Only the first call hit the inner embedder.
        assert inner.encode_calls == 1
        assert cache.stats.hits == 1

    def test_partial_hit_in_batch(self, setup):
        inner, cache, wrapped = setup
        wrapped.encode(["alpha"], show_progress=False)
        inner.encode_calls = 0  # reset

        # Mixed: alpha cached, beta and gamma new.
        wrapped.encode(["alpha", "beta", "gamma"], show_progress=False)
        # Inner called once with only the misses.
        assert inner.encode_calls == 1
        assert inner.texts_seen[-1] == ["beta", "gamma"]

    def test_results_in_original_order_after_partial_hit(self, setup):
        _, _, wrapped = setup
        # Warm cache with beta.
        wrapped.encode(["beta"], show_progress=False)
        # Now query in a different order.
        out = wrapped.encode(["alpha", "beta", "gamma"], show_progress=False)
        # The cached beta value matches what the inner would have produced.
        # Compute the expected embedding for "beta" directly.
        expected_beta = np.array(
            [ord(c) / 1000.0 for c in ("beta" + "\x00\x00\x00\x00")[:4]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(out[1], expected_beta)

    def test_duplicate_texts_in_batch_dedup_to_one_inner_call(self, setup):
        inner, _, wrapped = setup
        out = wrapped.encode(["x", "x", "x"], show_progress=False)
        assert out.shape == (3, 4)
        # Three positions in output, but only one unique text reached the embedder.
        assert inner.texts_seen[-1] == ["x"]

    def test_empty_input_returns_empty_array(self, setup):
        _, inner_cache, wrapped = setup
        out = wrapped.encode([], show_progress=False)
        assert out.shape == (0, 4)

    def test_string_input_treated_as_single_item(self, setup):
        _, _, wrapped = setup
        out = wrapped.encode("hello", show_progress=False)
        assert out.shape == (1, 4)

    def test_model_name_and_dimension_proxied(self, setup):
        inner, _, wrapped = setup
        assert wrapped.model_name == inner.model_name
        assert wrapped.dimension == inner.dimension

    def test_null_cache_means_no_cache_but_works(self):
        cache = make_cache(CacheSettings(backend="none"))
        inner = _MockEmbedder()
        wrapped = CachingEmbedder(inner, cache)
        wrapped.encode(["x"], show_progress=False)
        wrapped.encode(["x"], show_progress=False)
        # With NullCache, every call re-embeds.
        assert inner.encode_calls == 2


# --------------------------------------------------------------------------- #
# CachingExpansionPipeline
# --------------------------------------------------------------------------- #


class _CountingStrategy:
    name = "counting"

    def __init__(self, variant_text: str = "variant"):
        self.calls = 0
        self.variant_text = variant_text

    def expand(self, query):
        self.calls += 1
        return ExpansionResult(
            strategy=self.name, queries=(self.variant_text,), latency_ms=1.0
        )


class TestCachingExpansionPipeline:
    @pytest.fixture
    def setup(self):
        cache = SafeCache(MemoryCache())
        strat = _CountingStrategy()
        inner = QueryExpansionPipeline(strategies=[strat])
        wrapped = CachingExpansionPipeline(
            inner, cache, cache_key_model="mock-llm", cache_key_params={"n": 1}
        )
        return strat, cache, wrapped

    def test_first_call_misses_then_caches(self, setup):
        strat, cache, wrapped = setup
        wrapped.expand("query")
        assert strat.calls == 1
        assert cache.stats.misses == 1
        assert cache.stats.sets == 1

    def test_second_call_hits_cache(self, setup):
        strat, cache, wrapped = setup
        first = wrapped.expand("query")
        second = wrapped.expand("query")
        # Inner strategy only called once.
        assert strat.calls == 1
        # Cached object preserved.
        assert second.queries == first.queries
        assert cache.stats.hits == 1

    def test_different_queries_dont_share_cache(self, setup):
        strat, cache, wrapped = setup
        wrapped.expand("query a")
        wrapped.expand("query b")
        assert strat.calls == 2

    def test_different_params_dont_share_cache(self):
        cache = SafeCache(MemoryCache())
        strat = _CountingStrategy()
        inner = QueryExpansionPipeline(strategies=[strat])

        a = CachingExpansionPipeline(inner, cache, cache_key_model="m", cache_key_params={"n": 1})
        b = CachingExpansionPipeline(inner, cache, cache_key_model="m", cache_key_params={"n": 3})
        a.expand("query")
        b.expand("query")
        # Different params → different keys → both call the inner.
        assert strat.calls == 2

    def test_cached_result_is_full_expanded_query(self, setup):
        _, _, wrapped = setup
        result = wrapped.expand("query")
        assert isinstance(result, ExpandedQuery)
        assert "query" in result.queries  # original always present
        assert "variant" in result.queries

    def test_failed_expansion_also_cached(self):
        # A strategy that always errors out — still produces a result;
        # we cache it to avoid re-issuing the same failing LLM call.
        cache = SafeCache(MemoryCache())

        class _AlwaysFail:
            name = "broken"
            def expand(self, query):
                return ExpansionResult(strategy=self.name, error="boom", latency_ms=0.1)

        strat = _AlwaysFail()
        inner = QueryExpansionPipeline(strategies=[strat])
        wrapped = CachingExpansionPipeline(inner, cache, cache_key_model="m", cache_key_params={})

        wrapped.expand("query")
        wrapped.expand("query")
        assert cache.stats.hits == 1
