# src/cache/null.py

"""No-op cache. Useful as the default when caching is disabled."""

from __future__ import annotations

from typing import Any, Optional


class NullCache:
    """All operations succeed silently; ``get`` always returns ``None``."""

    backend_name = "null"

    def get(self, key: str) -> Optional[Any]:
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        return None

    def delete(self, key: str) -> None:
        return None

    def clear(self) -> None:
        return None

    def __len__(self) -> int:
        return 0
