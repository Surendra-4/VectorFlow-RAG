# tests/test_cache_backends.py

"""
Unit tests for cache backends, codec, keys, factory, and SafeCache.

Redis is tested through ``fakeredis`` so we don't need a real server.
Real-Redis smoke tests are skipped automatically when ``redis-server``
isn't reachable.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from src.cache.codec import PickleCodec
from src.cache.factory import make_cache
from src.cache.keys import CacheKeys
from src.cache.memory import MemoryCache
from src.cache.null import NullCache
from src.cache.safe import CacheStats, SafeCache
from src.config import CacheSettings


# --------------------------------------------------------------------------- #
# PickleCodec
# --------------------------------------------------------------------------- #


class TestPickleCodec:
    def test_round_trip_primitives(self):
        c = PickleCodec()
        for v in ("hello", 42, 3.14, True, None, [1, 2, 3], {"a": 1}):
            assert c.decode(c.encode(v)) == v

    def test_round_trip_numpy(self):
        c = PickleCodec()
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = c.decode(c.encode(arr))
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(arr, out)

    def test_round_trip_nested(self):
        c = PickleCodec()
        v = {"text": "hi", "scores": [0.1, 0.2], "meta": {"page": 3}}
        assert c.decode(c.encode(v)) == v


# --------------------------------------------------------------------------- #
# CacheKeys
# --------------------------------------------------------------------------- #


class TestCacheKeys:
    def test_embedding_deterministic(self):
        a = CacheKeys.embedding("model-x", "hello")
        b = CacheKeys.embedding("model-x", "hello")
        assert a == b

    def test_embedding_distinguishes_text(self):
        a = CacheKeys.embedding("model-x", "hello")
        b = CacheKeys.embedding("model-x", "world")
        assert a != b

    def test_embedding_distinguishes_model(self):
        a = CacheKeys.embedding("model-x", "hello")
        b = CacheKeys.embedding("model-y", "hello")
        assert a != b

    def test_expansion_deterministic(self):
        params = {"n": 3, "temp": 0.7}
        a = CacheKeys.expansion("m", ["multi_query"], params, "q")
        b = CacheKeys.expansion("m", ["multi_query"], params, "q")
        assert a == b

    def test_expansion_strategy_order_invariant(self):
        # Strategies are sorted before hashing so call-site order
        # doesn't fragment the cache.
        a = CacheKeys.expansion("m", ["multi_query", "hyde"], {}, "q")
        b = CacheKeys.expansion("m", ["hyde", "multi_query"], {}, "q")
        assert a == b

    def test_reranker_chunk_order_invariant(self):
        a = CacheKeys.reranker("model", "q", ["doc:0:a", "doc:1:b", "doc:2:c"])
        b = CacheKeys.reranker("model", "q", ["doc:2:c", "doc:0:a", "doc:1:b"])
        assert a == b

    def test_corpus_fingerprint_changes_with_chunks(self):
        a = CacheKeys.corpus_fingerprint(["doc:0:a", "doc:1:b"])
        b = CacheKeys.corpus_fingerprint(["doc:0:a", "doc:1:b", "doc:2:c"])
        assert a != b

    def test_retrieval_key_includes_fingerprint(self):
        a = CacheKeys.retrieval("fp_a", {"k": 5}, "q", 5)
        b = CacheKeys.retrieval("fp_b", {"k": 5}, "q", 5)
        assert a != b

    def test_keys_have_namespace_prefix(self):
        assert CacheKeys.embedding("m", "t").startswith("vfr:emb:v1:")
        assert CacheKeys.expansion("m", [], {}, "q").startswith("vfr:exp:v1:")
        assert CacheKeys.reranker("m", "q", []).startswith("vfr:rrk:v1:")
        assert CacheKeys.retrieval("fp", {}, "q", 1).startswith("vfr:ret:v1:")


# --------------------------------------------------------------------------- #
# NullCache
# --------------------------------------------------------------------------- #


class TestNullCache:
    def test_get_always_none(self):
        c = NullCache()
        c.set("k", "v")
        assert c.get("k") is None

    def test_no_op_methods_dont_raise(self):
        c = NullCache()
        c.delete("k")
        c.clear()
        assert len(c) == 0


# --------------------------------------------------------------------------- #
# MemoryCache
# --------------------------------------------------------------------------- #


class TestMemoryCache:
    def test_basic_set_get(self):
        c = MemoryCache()
        c.set("k", "v")
        assert c.get("k") == "v"

    def test_missing_key_returns_none(self):
        assert MemoryCache().get("absent") is None

    def test_overwrite_updates_value(self):
        c = MemoryCache()
        c.set("k", "first")
        c.set("k", "second")
        assert c.get("k") == "second"

    def test_lru_eviction(self):
        c = MemoryCache(max_entries=3)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.get("a")  # touch a → mru
        c.set("d", 4)  # evicts b (lru)
        assert c.get("a") == 1
        assert c.get("b") is None
        assert c.get("c") == 3
        assert c.get("d") == 4

    def test_ttl_expiry(self):
        c = MemoryCache(default_ttl_s=0)  # 0 TTL = never expires
        c.set("k", "v", ttl=1)
        assert c.get("k") == "v"
        time.sleep(1.1)
        assert c.get("k") is None

    def test_default_ttl_none_never_expires(self):
        c = MemoryCache(default_ttl_s=None)
        c.set("k", "v")
        time.sleep(0.1)
        assert c.get("k") == "v"

    def test_delete(self):
        c = MemoryCache()
        c.set("k", "v")
        c.delete("k")
        assert c.get("k") is None

    def test_clear(self):
        c = MemoryCache()
        c.set("k1", 1)
        c.set("k2", 2)
        c.clear()
        assert len(c) == 0

    def test_max_entries_must_be_positive(self):
        with pytest.raises(ValueError):
            MemoryCache(max_entries=0)

    def test_thread_safety_under_writes(self):
        """Concurrent writers shouldn't corrupt internal state."""
        c = MemoryCache(max_entries=200)

        def worker(prefix: str):
            for i in range(50):
                c.set(f"{prefix}_{i}", i)
                c.get(f"{prefix}_{i}")

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All 8*50 = 400 entries, capped at 200; LRU should keep us at exactly 200.
        assert len(c) == 200

    def test_lazy_purge_of_expired(self):
        c = MemoryCache(max_entries=100)
        c.set("k", "v", ttl=1)
        time.sleep(1.1)
        # Trigger a write to invoke lazy purge.
        c.set("other", "w")
        # k should be gone (purged during set).
        assert c.get("k") is None


# --------------------------------------------------------------------------- #
# SafeCache
# --------------------------------------------------------------------------- #


class _ExplodingBackend:
    backend_name = "exploding"

    def __init__(self):
        self.boom_count = 0

    def get(self, key):
        self.boom_count += 1
        raise RuntimeError("backend down")

    def set(self, key, value, ttl=None):
        raise RuntimeError("backend down")

    def delete(self, key):
        raise RuntimeError("backend down")

    def clear(self):
        raise RuntimeError("backend down")


class TestSafeCache:
    def test_passes_through_to_backend(self):
        backend = MemoryCache()
        c = SafeCache(backend)
        c.set("k", "v")
        assert c.get("k") == "v"

    def test_records_hits_and_misses(self):
        c = SafeCache(MemoryCache())
        c.set("k", "v")
        c.get("k")          # hit
        c.get("missing")    # miss
        c.get("missing2")   # miss
        snap = c.stats.snapshot()
        assert snap["hits"] == 1 and snap["misses"] == 2
        assert snap["hit_ratio"] == pytest.approx(1 / 3, abs=1e-3)

    def test_set_increments_sets_counter(self):
        c = SafeCache(MemoryCache())
        c.set("k", "v")
        c.set("k", "v2")
        assert c.stats.sets == 2

    def test_get_error_counted_and_returns_none(self):
        c = SafeCache(_ExplodingBackend())
        assert c.get("k") is None
        assert c.stats.errors == 1
        assert c.stats.misses == 1  # error treated as miss

    def test_set_error_counted_and_swallowed(self):
        c = SafeCache(_ExplodingBackend())
        c.set("k", "v")
        assert c.stats.errors == 1

    def test_delete_error_counted_and_swallowed(self):
        c = SafeCache(_ExplodingBackend())
        c.delete("k")
        assert c.stats.errors == 1

    def test_underlying_property_returns_inner_backend(self):
        backend = MemoryCache()
        assert SafeCache(backend).underlying is backend

    def test_backend_name_reflects_wrapped(self):
        assert SafeCache(NullCache()).backend_name == "null"
        assert SafeCache(MemoryCache()).backend_name == "memory"

    def test_stats_reset(self):
        c = SafeCache(MemoryCache())
        c.set("k", "v")
        c.get("k")
        c.stats.reset()
        assert c.stats.snapshot()["hits"] == 0


# --------------------------------------------------------------------------- #
# RedisCache via fakeredis
# --------------------------------------------------------------------------- #


class TestRedisCacheWithFakeRedis:
    """Test the RedisCache adapter using fakeredis (in-process Redis sim)."""

    @pytest.fixture
    def redis_cache(self):
        from src.cache.redis_cache import RedisCache
        import fakeredis

        # Patch redis.from_url in the RedisCache module so it returns a fake.
        fake = fakeredis.FakeRedis()
        with patch("redis.from_url", return_value=fake):
            c = RedisCache(url="redis://fake:6379/0")
        yield c
        fake.flushall()

    def test_set_get_round_trip(self, redis_cache):
        redis_cache.set("k", {"a": 1})
        assert redis_cache.get("k") == {"a": 1}

    def test_missing_key_returns_none(self, redis_cache):
        assert redis_cache.get("absent") is None

    def test_numpy_round_trip(self, redis_cache):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        redis_cache.set("emb", arr)
        out = redis_cache.get("emb")
        np.testing.assert_array_equal(out, arr)

    def test_ttl_expiry(self, redis_cache):
        redis_cache.set("k", "v", ttl=1)
        assert redis_cache.get("k") == "v"
        # fakeredis honors TTL via time.sleep
        time.sleep(1.1)
        assert redis_cache.get("k") is None

    def test_delete(self, redis_cache):
        redis_cache.set("k", "v")
        redis_cache.delete("k")
        assert redis_cache.get("k") is None

    def test_clear_only_wipes_vfr_namespace(self, redis_cache):
        # Pretend some non-vfr key exists; clear should not touch it.
        redis_cache.client.set("other:key", b"keep")
        redis_cache.set("vfr:emb:v1:k", "to_wipe")
        redis_cache.clear()
        assert redis_cache.client.get("other:key") == b"keep"
        assert redis_cache.get("vfr:emb:v1:k") is None


class TestRedisCacheConnectionFailure:
    def test_construction_raises_on_unreachable_server(self):
        from src.cache.redis_cache import RedisCache

        with pytest.raises(ConnectionError):
            RedisCache(url="redis://127.0.0.1:65535/0", socket_timeout_s=0.5)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


class TestFactory:
    def test_none_returns_null_safe(self):
        c = make_cache(CacheSettings(backend="none"))
        assert isinstance(c, SafeCache)
        assert c.backend_name == "null"

    def test_memory_returns_memory_safe(self):
        c = make_cache(CacheSettings(backend="memory"))
        assert c.backend_name == "memory"

    def test_redis_falls_back_to_memory_when_unreachable(self):
        c = make_cache(
            CacheSettings(backend="redis", redis_url="redis://127.0.0.1:65535/0")
        )
        # Connection-time failure → SafeCache(MemoryCache).
        assert c.backend_name == "memory"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            make_cache(CacheSettings(backend="lucene"))  # type: ignore[arg-type]

    def test_factory_returns_usable_cache_on_redis_success(self):
        # Patch redis.from_url so factory thinks Redis is up.
        import fakeredis
        from src.cache import redis_cache as rc_mod

        fake = fakeredis.FakeRedis()
        with patch.object(rc_mod, "_redis_mod", None, create=True):
            with patch("redis.from_url", return_value=fake):
                c = make_cache(CacheSettings(backend="redis"))
                assert c.backend_name == "redis"
                c.set("k", "v")
                assert c.get("k") == "v"
