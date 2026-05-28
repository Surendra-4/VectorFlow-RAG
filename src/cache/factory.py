# src/cache/factory.py

"""
Cache factory — selects a backend from settings and wraps it in :class:`SafeCache`.

Behavior matrix:

================  ==================================================
``backend``       Result
================  ==================================================
``"none"``        ``SafeCache(NullCache())`` — zero overhead
``"memory"``      ``SafeCache(MemoryCache(...))``
``"redis"``       ``SafeCache(RedisCache(...))``; on connection
                   failure logs a warning and falls back to a
                   ``MemoryCache`` so retrieval still works.
================  ==================================================

The factory never raises — callers always get a usable cache object.
"""

from __future__ import annotations

from typing import Optional

from src.cache.memory import MemoryCache
from src.cache.null import NullCache
from src.cache.safe import CacheStats, SafeCache
from src.config import CacheSettings, get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


def make_cache(
    settings: Optional[CacheSettings] = None,
    stats: Optional[CacheStats] = None,
) -> SafeCache:
    cfg = settings if settings is not None else get_settings().cache
    backend_name = cfg.backend.lower()

    if backend_name == "none":
        return SafeCache(NullCache(), stats=stats)

    if backend_name == "memory":
        return SafeCache(
            MemoryCache(default_ttl_s=cfg.ttl_seconds, max_entries=10_000),
            stats=stats,
        )

    if backend_name == "redis":
        # Lazy import + connection-time fallback. If Redis is unreachable
        # we don't crash; we degrade to MemoryCache and log once.
        from src.cache.redis_cache import RedisCache

        try:
            backend = RedisCache(url=cfg.redis_url, default_ttl_s=cfg.ttl_seconds)
            return SafeCache(backend, stats=stats)
        except Exception as exc:
            logger.warning(
                "Redis cache unavailable at %s (%s); falling back to in-memory cache",
                cfg.redis_url, exc,
            )
            return SafeCache(
                MemoryCache(default_ttl_s=cfg.ttl_seconds, max_entries=10_000),
                stats=stats,
            )

    raise ValueError(f"Unknown cache backend: {backend_name!r}")
