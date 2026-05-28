# src/cache/redis_cache.py

"""
Redis-backed cache with lazy connection + graceful degradation.

Construction is fail-fast: ``RedisCache(url)`` pings the server, and
raises ``ConnectionError`` if it can't reach it. The factory (see
:mod:`src.cache.factory`) catches that error and falls back to an
in-memory cache, so the application never crashes on a missing Redis.

Once connected, per-call errors (network blip, transient timeout) are
contained by the :class:`SafeCache` wrapper.

Cache values are bytes (pickled by the codec layer). TTL is set
per-key in seconds.
"""

from __future__ import annotations

from typing import Any, Optional

from src.cache.codec import Codec, PickleCodec
from src.logging_setup import get_logger

logger = get_logger(__name__)


class RedisCache:
    """Adapter conforming to :class:`src.interfaces.CacheProtocol`."""

    backend_name = "redis"

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        default_ttl_s: Optional[int] = None,
        codec: Optional[Codec] = None,
        socket_timeout_s: float = 2.0,
    ):
        # Lazy import — keep ``redis`` an optional dependency unless this
        # backend is actually chosen.
        try:
            import redis as _redis
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ImportError("redis-py is required for RedisCache") from exc

        self._redis_mod = _redis
        self.url = url
        self.codec = codec or PickleCodec()
        self.default_ttl_s = default_ttl_s

        self.client = _redis.from_url(
            url,
            socket_timeout=socket_timeout_s,
            socket_connect_timeout=socket_timeout_s,
        )
        # Eager ping so we fail loudly at construction rather than on
        # the first cache call. The factory catches this and falls back.
        try:
            self.client.ping()
        except Exception as exc:
            raise ConnectionError(f"Could not reach Redis at {url}: {exc}") from exc

        logger.info("RedisCache connected url=%s default_ttl=%s", url, default_ttl_s)

    # ------------------------------------------------------------------ #
    # CacheProtocol API
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[Any]:
        raw = self.client.get(key)
        if raw is None:
            return None
        return self.codec.decode(raw)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        encoded = self.codec.encode(value)
        ttl_resolved = ttl if ttl is not None else self.default_ttl_s
        if ttl_resolved:
            self.client.setex(key, int(ttl_resolved), encoded)
        else:
            self.client.set(key, encoded)

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def clear(self) -> None:
        # Pattern-scoped wipe — only our namespace, never the whole DB.
        for k in self.client.scan_iter(match="vfr:*"):
            self.client.delete(k)

    def __len__(self) -> int:
        return sum(1 for _ in self.client.scan_iter(match="vfr:*"))
