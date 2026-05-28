# src/cache/__init__.py

"""
Cache layer for VectorFlow-RAG.

Three pluggable backends conform to :class:`src.interfaces.CacheProtocol`:

* :class:`NullCache`    — no-op. Used when ``settings.cache.backend == "none"``.
* :class:`MemoryCache`  — thread-safe in-process LRU+TTL. Default when caching is on.
* :class:`RedisCache`   — shared cache across processes. Lazy redis import.

All backends are wrapped by :class:`SafeCache` at the factory boundary so a
cache error never breaks a retrieval — at worst it degrades to a cache miss.

Cache *keys* are built by :mod:`src.cache.keys` to keep namespacing,
schema-versioning, and hashing in one place. Values are encoded by
:mod:`src.cache.codec` (pickle by default for numpy compatibility).

Use :func:`make_cache` to construct the configured backend.
"""

from src.cache.codec import Codec, PickleCodec
from src.cache.factory import make_cache
from src.cache.keys import CacheKeys
from src.cache.memory import MemoryCache
from src.cache.null import NullCache
from src.cache.safe import CacheStats, SafeCache

__all__ = [
    "Codec",
    "PickleCodec",
    "CacheKeys",
    "CacheStats",
    "MemoryCache",
    "NullCache",
    "SafeCache",
    "make_cache",
]
