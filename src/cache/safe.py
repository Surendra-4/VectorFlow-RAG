# src/cache/safe.py

"""
SafeCache — wraps any cache backend so transient errors degrade to misses.

This is the boundary at which the application meets the cache. The
contract:

* ``get`` returns ``None`` on any underlying error (treats it as a miss).
* ``set`` swallows any error after logging it.
* Both increment per-operation counters in :class:`CacheStats`.

The rest of the codebase relies on this — wrappers (CachingEmbedder,
CachingExpansionPipeline, retrieval cache) never have to defend against
backend failures themselves.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

from src.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Per-process counters; reset by :meth:`reset`. Thread-safe."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0
    deletes: int = 0

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.errors = 0
        self.deletes = 0

    @property
    def total_ops(self) -> int:
        return self.hits + self.misses + self.sets + self.deletes

    @property
    def hit_ratio(self) -> float:
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0

    def snapshot(self) -> dict:
        """Lightweight dict view suitable for RetrievalTrace embedding."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_ratio": round(self.hit_ratio, 4),
        }


class SafeCache:
    """Wraps a backend, never lets cache failures escape."""

    def __init__(self, backend, stats: Optional[CacheStats] = None):
        self._backend = backend
        self.stats = stats or CacheStats()
        self._lock = threading.Lock()
        self.backend_name = getattr(backend, "backend_name", type(backend).__name__)

    # ------------------------------------------------------------------ #
    # CacheProtocol API
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self._backend.get(key)
        except Exception as exc:
            with self._lock:
                self.stats.errors += 1
                self.stats.misses += 1
            logger.warning("SafeCache.get error key=%s: %s", key, exc)
            return None
        with self._lock:
            if value is None:
                self.stats.misses += 1
            else:
                self.stats.hits += 1
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            self._backend.set(key, value, ttl=ttl)
            with self._lock:
                self.stats.sets += 1
        except Exception as exc:
            with self._lock:
                self.stats.errors += 1
            logger.warning("SafeCache.set error key=%s: %s", key, exc)

    def delete(self, key: str) -> None:
        try:
            self._backend.delete(key)
            with self._lock:
                self.stats.deletes += 1
        except Exception as exc:
            with self._lock:
                self.stats.errors += 1
            logger.warning("SafeCache.delete error key=%s: %s", key, exc)

    def clear(self) -> None:
        try:
            self._backend.clear()
        except Exception as exc:
            with self._lock:
                self.stats.errors += 1
            logger.warning("SafeCache.clear error: %s", exc)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def underlying(self):
        """Direct access to the wrapped backend (for tests / debugging)."""
        return self._backend

    def __len__(self) -> int:
        try:
            return len(self._backend)
        except Exception:
            return 0
