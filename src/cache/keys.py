# src/cache/keys.py

"""
Cache key builders.

All keys are namespaced ``vfr:<namespace>:v<schema>:...`` so:

* future schema-format changes can bump ``v`` and orphan stale entries
* operators can wipe a single namespace with ``KEYS vfr:emb:*`` etc.
* cross-namespace key collisions are structurally impossible
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

SCHEMA_VERSION = 1

# Standard slice widths — 8 hex chars (32 bits) for short
# discriminators (model names, config), 32 hex chars (128 bits) for
# content that needs collision-free identity.
_SHORT = 8
_LONG = 32


def _sha(data: str, length: int) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:length]


def _hash_obj(obj: object, length: int) -> str:
    """Hash a JSON-serializable object stably (sorted keys)."""
    serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return _sha(serialized, length)


class CacheKeys:
    """Static builders for every cache namespace in the system."""

    @staticmethod
    def embedding(model_name: str, text: str, input_type: str = "") -> str:
        # input_type distinguishes asymmetric-model embeddings (E5 query vs
        # passage produce different vectors for the same text). Empty for
        # symmetric models — keeps keys stable across the English path.
        it = input_type or ""
        return (
            f"vfr:emb:v{SCHEMA_VERSION}:{_sha(model_name, _SHORT)}:"
            f"{_sha(it, _SHORT)}:{_sha(text, _LONG)}"
        )

    @staticmethod
    def expansion(
        expansion_model: str,
        strategies: Iterable[str],
        params: dict,
        query: str,
    ) -> str:
        strat_part = _hash_obj(sorted(strategies), _SHORT)
        params_part = _hash_obj(params, _SHORT)
        return (
            f"vfr:exp:v{SCHEMA_VERSION}:"
            f"{_sha(expansion_model, _SHORT)}:{strat_part}:{params_part}:"
            f"{_sha(query, _LONG)}"
        )

    @staticmethod
    def reranker(model_name: str, query: str, chunk_ids: Iterable[str]) -> str:
        # Sorted chunk IDs so different orderings of the same pool
        # produce the same key (rerank is order-invariant on input).
        ids_part = _hash_obj(sorted(chunk_ids), _LONG)
        return (
            f"vfr:rrk:v{SCHEMA_VERSION}:"
            f"{_sha(model_name, _SHORT)}:{_sha(query, _LONG)}:{ids_part}"
        )

    @staticmethod
    def retrieval(
        corpus_fingerprint: str,
        rag_config: dict,
        query: str,
        k: int,
    ) -> str:
        config_part = _hash_obj(rag_config, _SHORT)
        return (
            f"vfr:ret:v{SCHEMA_VERSION}:"
            f"{corpus_fingerprint}:{config_part}:{_sha(query, _LONG)}:k{k}"
        )

    @staticmethod
    def corpus_fingerprint(chunk_ids: Iterable[str]) -> str:
        """Stable hash of the current corpus state — drives retrieval invalidation."""
        joined = "|".join(sorted(chunk_ids))
        return _sha(joined, 16)
