# src/cache/caching_embedder.py

"""
Cache wrapper around an :class:`src.embedder.Embedder`.

Strategy: cache each *individual text*'s embedding by ``(model_name,
sha256(text))``. Batch ``encode`` calls hit the cache per-text, then
issue a single sub-batch embedding for the misses, then stitch the
results back in original order. This keeps the cache hit rate high
even when callers always pass batches (e.g. ingestion).

The wrapper conforms to the same shape as ``Embedder`` for the
methods used by the rest of the codebase (``encode``, ``model_name``,
``dimension``), so it slots in transparently.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Union

import numpy as np

from src.cache.keys import CacheKeys
from src.cache.safe import SafeCache
from src.logging_setup import get_logger

logger = get_logger(__name__)


class CachingEmbedder:
    """Wraps an :class:`Embedder` with a per-text embedding cache."""

    def __init__(self, inner, cache: SafeCache):
        self._inner = inner
        self._cache = cache
        # Mirror the inner embedder's surface so HybridRetriever can
        # use this in place of the real embedder without changes.
        self.model_name = inner.model_name
        self.dimension = inner.dimension
        # Some callers (eg. CachingEmbedder of a CachingEmbedder) want
        # the underlying device.
        self.device = getattr(inner, "device", "cpu")

    # ------------------------------------------------------------------ #
    # Embedder-compatible API
    # ------------------------------------------------------------------ #

    def encode(
        self,
        texts: Union[str, Iterable[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        input_type: Optional[str] = None,
    ) -> np.ndarray:
        """
        Embed ``texts`` using the cache where possible.

        ``input_type`` ("query"/"passage"/None) is threaded to the inner
        embedder AND folded into the cache key — asymmetric models produce
        different vectors per input type for the same text, so they must
        cache separately.

        Returns a 2D float array of shape ``(n_texts, dim)`` in the same
        order as the input — even when some embeddings were cached and
        others computed.
        """
        if isinstance(texts, str):
            texts_list: List[str] = [texts]
        else:
            texts_list = list(texts)

        if not texts_list:
            return np.zeros((0, self.dimension), dtype=np.float32)

        it = input_type or ""

        results: List[np.ndarray] = [None] * len(texts_list)  # type: ignore[assignment]
        miss_indices: List[int] = []
        miss_texts: List[str] = []
        seen_misses: dict[str, int] = {}

        for i, text in enumerate(texts_list):
            key = CacheKeys.embedding(self.model_name, text, it)
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
                continue
            if text in seen_misses:
                miss_indices.append(i)
                miss_texts.append(text)
                continue
            seen_misses[text] = len(miss_texts)
            miss_indices.append(i)
            miss_texts.append(text)

        if miss_texts:
            unique_texts = list(dict.fromkeys(miss_texts))
            new_embs = self._inner.encode(
                unique_texts,
                batch_size=batch_size,
                show_progress=show_progress,
                input_type=input_type,
            )
            embedded_by_text = dict(zip(unique_texts, new_embs))

            for text, emb in embedded_by_text.items():
                self._cache.set(
                    CacheKeys.embedding(self.model_name, text, it),
                    np.asarray(emb, dtype=np.float32),
                )

            for idx, text in zip(miss_indices, miss_texts):
                results[idx] = embedded_by_text[text]

        return np.vstack([np.asarray(r, dtype=np.float32) for r in results])
