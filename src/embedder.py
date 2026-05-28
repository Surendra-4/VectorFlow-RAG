# src/embedder.py

"""
Sentence-Transformers embedding wrapper with auto device detection and
encapsulated instruction-prefix handling.

Some multilingual models (notably the ``multilingual-e5`` family) are
*asymmetric*: queries must be prefixed ``"query: "`` and passages
``"passage: "``. That quirk is fully encapsulated here — callers express
*intent* via ``input_type="query"`` / ``"passage"`` and never learn which
model is active. For symmetric models (e.g. MiniLM, BGE-M3) the prefixes
resolve to empty strings, so the English path is byte-identical.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

InputType = Optional[str]  # "query" | "passage" | None


def _resolve_device(preferred: Optional[str]) -> str:
    """
    Pick a device for inference.

    If ``preferred`` is set, return it as-is (caller's choice). Otherwise,
    auto-detect: CUDA → MPS (Apple Silicon) → CPU. Falls back to CPU if
    ``torch`` is unavailable for any reason.
    """
    if preferred:
        return preferred
    try:
        import torch  # local import keeps cold-start cheap when device is given

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("torch device probe failed, falling back to cpu: %s", exc)
    return "cpu"


def _auto_prefixes(model_name: str) -> tuple[str, str]:
    """
    Infer (query_prefix, passage_prefix) from the model name.

    Only the E5 family is asymmetric among our supported models. Everything
    else is symmetric → empty prefixes. This keeps model-specific knowledge
    inside the embedder adapter, exactly one place.
    """
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    return "", ""


class Embedder:
    """Embed texts into normalized dense vectors using sentence-transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None,
        query_prefix: Optional[str] = None,
        passage_prefix: Optional[str] = None,
    ):
        cfg = get_settings().embedder
        self.model_name = model_name or cfg.model_name
        self.device = _resolve_device(device if device is not None else cfg.device)
        self.normalize = cfg.normalize if normalize is None else normalize

        # Prefix resolution: explicit arg > config > auto-detect.
        auto_q, auto_p = _auto_prefixes(self.model_name)
        cfg_q = cfg.query_prefix
        cfg_p = cfg.passage_prefix
        self.query_prefix = (
            query_prefix if query_prefix is not None
            else (cfg_q if cfg_q is not None else auto_q)
        )
        self.passage_prefix = (
            passage_prefix if passage_prefix is not None
            else (cfg_p if cfg_p is not None else auto_p)
        )

        logger.info(
            "Loading embedder model=%s device=%s query_prefix=%r passage_prefix=%r",
            self.model_name, self.device, self.query_prefix, self.passage_prefix,
        )
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.debug("Embedder ready: dim=%d", self.dimension)

    def _prefix_for(self, input_type: InputType) -> str:
        if input_type == "query":
            return self.query_prefix
        if input_type == "passage":
            return self.passage_prefix
        return ""  # unspecified → no prefix (safe; preserves English exactly)

    def encode(
        self,
        texts: Union[str, Iterable[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        input_type: InputType = None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)

        prefix = self._prefix_for(input_type)
        if prefix:
            texts = [prefix + t for t in texts]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )


if __name__ == "__main__":
    e = Embedder()
    emb = e.encode(["Hello", "World"], show_progress=False)
    print(emb.shape)
