# src/reranker.py

"""
Cross-encoder reranker for second-stage retrieval refinement.

A cross-encoder jointly encodes ``(query, candidate)`` pairs and emits a
relevance score per pair. This is significantly more accurate than the
bi-encoder cosine of the dense retriever because the model attends across
both inputs — at the cost of being O(N) per query, with N being the
candidate pool size.

Usage:

    from src.reranker import CrossEncoderReranker
    rr = CrossEncoderReranker()  # config-driven defaults
    top = rr.rerank(query, candidates, top_n=3)

Conforms to :class:`src.interfaces.RerankerProtocol`.

Design notes:

* **Lazy model load** — the model is downloaded/loaded on first call to
  :meth:`rerank`, so import-time cost stays zero.
* **Auto device** — picks CUDA → MPS → CPU unless overridden.
* **Batched** — uses ``CrossEncoder.predict``'s built-in batching.
* **Pure annotation** — never mutates input; produces shallow-copied
  result dicts with an added ``rerank_score`` key.
* **Future-proof** — ``batch_size`` and ``device`` are kwargs so caller
  code can later opt into GPU acceleration without API change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.config import get_settings
from src.embedder import _resolve_device
from src.logging_setup import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        top_n: Optional[int] = None,
        batch_size: int = 32,
    ):
        cfg = get_settings().reranker
        self.model_name = model_name or cfg.model_name
        # ``device=None`` (sentinel) → auto. Explicit string → respected.
        self.device = _resolve_device(device if device is not None else cfg.device)
        self.top_n = top_n if top_n is not None else cfg.top_n
        self.batch_size = batch_size
        self._model = None  # lazy

    @property
    def model(self):
        """Lazy-load the cross-encoder model on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker model=%s device=%s", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: Optional[int] = None,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Score each candidate against the query and return the top-N.

        Args:
            query: search query
            candidates: list of candidate dicts; each must carry ``text_key``
            top_n: number of results to return (defaults to ``self.top_n``)
            text_key: dict key whose value is the candidate text

        Returns:
            New list of dicts (shallow copies of inputs) with an added
            ``rerank_score`` field, sorted by ``rerank_score`` descending and
            truncated to ``top_n``. Stable on ties (Python's sort is stable).
        """
        n = top_n if top_n is not None else self.top_n
        if not candidates:
            return []

        pairs = [(query, c[text_key]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        annotated: List[Dict[str, Any]] = []
        for c, s in zip(candidates, scores):
            new = dict(c)
            new["rerank_score"] = float(s)
            annotated.append(new)

        # Stable sort preserves relative order on score ties.
        annotated.sort(key=lambda x: x["rerank_score"], reverse=True)
        return annotated[:n]


if __name__ == "__main__":
    rr = CrossEncoderReranker()
    candidates = [
        {"text": "The Eiffel Tower is in Paris."},
        {"text": "Python is a programming language."},
        {"text": "Paris is the capital of France."},
    ]
    print(rr.rerank("Where is the Eiffel Tower?", candidates, top_n=2))
