# src/cache/memory.py

"""
In-process thread-safe cache with LRU eviction and per-entry TTL.

Implementation notes:

* ``OrderedDict`` provides O(1) LRU semantics via ``move_to_end``.
* A single ``RLock`` guards all mutations — sufficient for our access
  pattern (predominantly reads with bursty writes during ingestion /
  retrieval). For higher concurrency, a sharded version is trivial.
* TTL is checked lazily on read; an entry past its TTL is removed and
  treated as a miss. We also purge a small batch of expired entries
  on each write to bound memory growth between LRU evictions.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple


class MemoryCache:
    """Thread-safe LRU cache with optional per-entry TTL."""

    backend_name = "memory"

    def __init__(self, max_entries: int = 10_000, default_ttl_s: Optional[int] = None):
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")
        self._max_entries = max_entries
        self._default_ttl_s = default_ttl_s
        self._store: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_expired(self, expires_at: float, now: Optional[float] = None) -> bool:
        if expires_at <= 0:  # 0 = never expires
            return False
        return (now if now is not None else time.monotonic()) > expires_at

    def _purge_some_expired(self, budget: int = 16) -> None:
        """Best-effort expired-entry sweep, bounded so writes stay fast."""
        if self._default_ttl_s is None and not self._store:
            return
        now = time.monotonic()
        examined = 0
        for key in list(self._store.keys()):
            if examined >= budget:
                break
            expires_at, _ = self._store[key]
            if self._is_expired(expires_at, now):
                self._store.pop(key, None)
            examined += 1

    # ------------------------------------------------------------------ #
    # CacheProtocol API
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._is_expired(expires_at):
                self._store.pop(key, None)
                return None
            # LRU bookkeeping — move to end so this is the most-recently-used.
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl_resolved = ttl if ttl is not None else self._default_ttl_s
        expires_at = (time.monotonic() + ttl_resolved) if ttl_resolved else 0.0
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (expires_at, value)
            self._purge_some_expired()
            # LRU eviction.
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
