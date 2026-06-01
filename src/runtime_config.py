# src/runtime_config.py

"""
Runtime configuration store (Phase 12c).

``Settings`` (env-driven, lru_cache singleton) is the **boot baseline**. This
module adds a separate, *mutable* layer that callers can update at runtime
without restarting the worker.

Two strict groups are kept distinct — silently mixing them is a Phase 12
non-goal:

* :class:`LiveQuerySettings` — settings safely applied at the next request
  (active chat provider/model, reranker on/off + model, query-expansion
  on/off + strategies, retrieval k / RRF / candidates).
* :class:`IndexConstructionSettings` — settings that require a *rebuild*
  (embedding model, chunk size/overlap, vector backend, FAISS topology,
  BM25 stemmer policy). These are **staged** here; a rebuild only happens
  when the IndexManager (Phase 12e) is explicitly invoked.

The store persists to ``<project_root>/var/runtime_config.json`` so the
operator's choices survive process restarts. The file is created lazily on
first write — fresh installs reproduce the env baseline byte-for-byte.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.logging_setup import get_logger
from src.providers import (
    ChatModelConfig,
    EmbeddingModelConfig,
    RerankerModelConfig,
    is_registered,
)

if TYPE_CHECKING:
    from src.config import Settings

logger = get_logger(__name__)

RUNTIME_CONFIG_VERSION = 1
_VALID_EXPANSION_STRATEGIES = {"multi_query", "hyde"}
_VALID_VECTOR_BACKENDS = {"chromadb", "faiss"}
# Phase 12c knows the simple FAISS types; Phase 12f extends this set.
_VALID_FAISS_TYPES = {"hnsw", "flat", "ivf"}


# --------------------------------------------------------------------------- #
# Live (per-request) settings
# --------------------------------------------------------------------------- #


@dataclass
class LiveQuerySettings:
    """Knobs that can be flipped between two consecutive requests."""

    chat: ChatModelConfig = field(default_factory=ChatModelConfig)
    reranker_enabled: bool = False
    reranker: RerankerModelConfig = field(default_factory=RerankerModelConfig)
    expansion_enabled: bool = False
    expansion_strategies: List[str] = field(default_factory=lambda: ["multi_query"])
    expansion_multi_query_count: int = 3
    expansion_hyde_count: int = 1
    retrieval_k_default: int = 5
    retrieval_candidates_per_modality: int = 10
    retrieval_rrf_k: int = 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat": self.chat.to_dict(),
            "reranker_enabled": self.reranker_enabled,
            "reranker": self.reranker.to_dict(),
            "expansion_enabled": self.expansion_enabled,
            "expansion_strategies": list(self.expansion_strategies),
            "expansion_multi_query_count": self.expansion_multi_query_count,
            "expansion_hyde_count": self.expansion_hyde_count,
            "retrieval_k_default": self.retrieval_k_default,
            "retrieval_candidates_per_modality": self.retrieval_candidates_per_modality,
            "retrieval_rrf_k": self.retrieval_rrf_k,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiveQuerySettings":
        out = cls()
        if "chat" in data and isinstance(data["chat"], dict):
            out.chat = ChatModelConfig.from_dict(data["chat"])
        if "reranker" in data and isinstance(data["reranker"], dict):
            out.reranker = RerankerModelConfig.from_dict(data["reranker"])
        for attr in (
            "reranker_enabled", "expansion_enabled", "expansion_strategies",
            "expansion_multi_query_count", "expansion_hyde_count",
            "retrieval_k_default", "retrieval_candidates_per_modality",
            "retrieval_rrf_k",
        ):
            if attr in data:
                setattr(out, attr, data[attr])
        return out

    @classmethod
    def from_settings(cls, settings: "Settings") -> "LiveQuerySettings":
        return cls(
            chat=ChatModelConfig(
                provider="ollama",
                model=settings.llm.model,
                base_url=settings.llm.base_url,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
                request_timeout_s=settings.llm.request_timeout_s,
            ),
            reranker_enabled=settings.reranker.enabled,
            reranker=RerankerModelConfig(
                provider="cross_encoder",
                model=settings.reranker.model_name,
                enabled=settings.reranker.enabled,
                top_n=settings.reranker.top_n,
                device=settings.reranker.device,
            ),
            expansion_enabled=settings.query_expansion.enabled,
            expansion_strategies=list(settings.query_expansion.strategies),
            expansion_multi_query_count=settings.query_expansion.multi_query_count,
            expansion_hyde_count=settings.query_expansion.hyde_count,
            retrieval_k_default=settings.retrieval.k_default,
            retrieval_candidates_per_modality=settings.retrieval.candidates_per_modality,
            retrieval_rrf_k=settings.retrieval.rrf_k,
        )


# --------------------------------------------------------------------------- #
# Index-construction (staged, rebuild-required) settings
# --------------------------------------------------------------------------- #


@dataclass
class IndexConstructionSettings:
    """Knobs that change the *shape* of the index — never silently applied."""

    embedding: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    chunk_size: int = 500
    chunk_overlap: int = 50
    vector_backend: str = "chromadb"
    faiss_index_type: str = "hnsw"
    faiss_hnsw_m: int = 32
    faiss_hnsw_ef_construction: int = 200
    faiss_hnsw_ef_search: int = 64
    bm25_use_stemmer: bool = True
    bm25_language: str = "english"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding": self.embedding.to_dict(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_backend": self.vector_backend,
            "faiss_index_type": self.faiss_index_type,
            "faiss_hnsw_m": self.faiss_hnsw_m,
            "faiss_hnsw_ef_construction": self.faiss_hnsw_ef_construction,
            "faiss_hnsw_ef_search": self.faiss_hnsw_ef_search,
            "bm25_use_stemmer": self.bm25_use_stemmer,
            "bm25_language": self.bm25_language,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexConstructionSettings":
        out = cls()
        if "embedding" in data and isinstance(data["embedding"], dict):
            out.embedding = EmbeddingModelConfig.from_dict(data["embedding"])
        for attr in (
            "chunk_size", "chunk_overlap", "vector_backend",
            "faiss_index_type", "faiss_hnsw_m", "faiss_hnsw_ef_construction",
            "faiss_hnsw_ef_search", "bm25_use_stemmer", "bm25_language",
        ):
            if attr in data:
                setattr(out, attr, data[attr])
        return out

    @classmethod
    def from_settings(cls, settings: "Settings") -> "IndexConstructionSettings":
        return cls(
            embedding=EmbeddingModelConfig(
                provider="sentence_transformers",
                model=settings.embedder.model_name,
                device=settings.embedder.device,
                normalize=settings.embedder.normalize,
            ),
            chunk_size=settings.chunker.chunk_size,
            chunk_overlap=settings.chunker.overlap,
            vector_backend=settings.vector_store.backend,
            faiss_index_type=settings.vector_store.faiss_index_type,
            faiss_hnsw_m=settings.vector_store.faiss_hnsw_m,
            faiss_hnsw_ef_construction=settings.vector_store.faiss_hnsw_ef_construction,
            faiss_hnsw_ef_search=settings.vector_store.faiss_hnsw_ef_search,
            bm25_use_stemmer=settings.bm25.use_stemmer,
            bm25_language=settings.bm25.language,
        )


# --------------------------------------------------------------------------- #
# Composite + store
# --------------------------------------------------------------------------- #


@dataclass
class RuntimeConfig:
    """Snapshot of all runtime-mutable state."""

    live: LiveQuerySettings
    staged_index: IndexConstructionSettings
    # What the *currently loaded* index was actually built with. Used by the
    # compatibility validator (12g) to decide whether a config change requires
    # a rebuild before it takes effect.
    active_index: IndexConstructionSettings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": RUNTIME_CONFIG_VERSION,
            "live": self.live.to_dict(),
            "staged_index": self.staged_index.to_dict(),
            "active_index": self.active_index.to_dict(),
        }


class RuntimeConfigError(ValueError):
    """Raised when an update fails validation. Carries the offending field."""

    def __init__(self, message: str, *, field: Optional[str] = None):
        super().__init__(message)
        self.field = field


class RuntimeConfigStore:
    """Thread-safe, file-backed mutable layer on top of ``Settings``.

    Validation is centralised in :meth:`_validate_live` / :meth:`_validate_index`
    so the API and tests share the same rules.
    """

    def __init__(self, settings: "Settings", path: Optional[Path] = None):
        self.path = Path(path) if path else Path(settings.app.project_root) / "var" / "runtime_config.json"
        self._lock = threading.RLock()
        # Hydrate from settings first, then overlay anything persisted on disk.
        self._config = RuntimeConfig(
            live=LiveQuerySettings.from_settings(settings),
            staged_index=IndexConstructionSettings.from_settings(settings),
            active_index=IndexConstructionSettings.from_settings(settings),
        )
        self._load()

    # ------------------------------------------------------------------ #
    # Read-only views (return fresh copies so callers can't mutate state)
    # ------------------------------------------------------------------ #

    @property
    def live(self) -> LiveQuerySettings:
        with self._lock:
            return LiveQuerySettings.from_dict(self._config.live.to_dict())

    @property
    def staged_index(self) -> IndexConstructionSettings:
        with self._lock:
            return IndexConstructionSettings.from_dict(self._config.staged_index.to_dict())

    @property
    def active_index(self) -> IndexConstructionSettings:
        with self._lock:
            return IndexConstructionSettings.from_dict(self._config.active_index.to_dict())

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._config.to_dict()

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #

    def update_live(self, patch: Dict[str, Any]) -> LiveQuerySettings:
        """Apply a partial patch to live settings. Returns the new snapshot."""
        with self._lock:
            current = self._config.live.to_dict()
            merged = _deep_merge(current, patch)
            new_live = LiveQuerySettings.from_dict(merged)
            self._validate_live(new_live)
            self._config.live = new_live
            self._persist()
            return self.live

    def stage_index(self, patch: Dict[str, Any]) -> IndexConstructionSettings:
        """Apply a partial patch to staged index settings. NEVER rebuilds."""
        with self._lock:
            current = self._config.staged_index.to_dict()
            merged = _deep_merge(current, patch)
            new_staged = IndexConstructionSettings.from_dict(merged)
            self._validate_index(new_staged)
            self._config.staged_index = new_staged
            self._persist()
            return self.staged_index

    def mark_index_active(self, index_settings: IndexConstructionSettings) -> None:
        """Called by IndexManager after a successful build/load — the active
        index now reflects exactly these settings."""
        with self._lock:
            self._validate_index(index_settings)
            self._config.active_index = IndexConstructionSettings.from_dict(
                index_settings.to_dict()
            )
            self._persist()

    def reset_staged_to_active(self) -> IndexConstructionSettings:
        """Discard staged changes — go back to whatever the active index uses."""
        with self._lock:
            self._config.staged_index = IndexConstructionSettings.from_dict(
                self._config.active_index.to_dict()
            )
            self._persist()
            return self.staged_index

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_live(live: LiveQuerySettings) -> None:
        if not is_registered(live.chat.provider):
            raise RuntimeConfigError(
                f"Unknown chat provider: {live.chat.provider!r}", field="chat.provider"
            )
        if not live.chat.model:
            raise RuntimeConfigError("chat.model is required", field="chat.model")
        if live.chat.max_tokens < 1:
            raise RuntimeConfigError("chat.max_tokens must be >= 1", field="chat.max_tokens")
        if not 0.0 <= live.chat.temperature <= 2.0:
            raise RuntimeConfigError(
                "chat.temperature must be between 0.0 and 2.0", field="chat.temperature"
            )
        bad = [s for s in live.expansion_strategies if s not in _VALID_EXPANSION_STRATEGIES]
        if bad:
            raise RuntimeConfigError(
                f"Unknown expansion strategies: {bad}", field="expansion_strategies"
            )
        if live.retrieval_k_default < 1:
            raise RuntimeConfigError("retrieval_k_default must be >= 1",
                                     field="retrieval_k_default")
        if live.retrieval_candidates_per_modality < 1:
            raise RuntimeConfigError(
                "retrieval_candidates_per_modality must be >= 1",
                field="retrieval_candidates_per_modality",
            )
        if live.retrieval_rrf_k < 1:
            raise RuntimeConfigError("retrieval_rrf_k must be >= 1",
                                     field="retrieval_rrf_k")
        if live.reranker.top_n < 1:
            raise RuntimeConfigError("reranker.top_n must be >= 1",
                                     field="reranker.top_n")

    @staticmethod
    def _validate_index(idx: IndexConstructionSettings) -> None:
        if idx.vector_backend not in _VALID_VECTOR_BACKENDS:
            raise RuntimeConfigError(
                f"vector_backend must be one of {sorted(_VALID_VECTOR_BACKENDS)}",
                field="vector_backend",
            )
        if idx.faiss_index_type not in _VALID_FAISS_TYPES:
            raise RuntimeConfigError(
                f"faiss_index_type must be one of {sorted(_VALID_FAISS_TYPES)}",
                field="faiss_index_type",
            )
        if idx.chunk_size <= 0:
            raise RuntimeConfigError("chunk_size must be > 0", field="chunk_size")
        if idx.chunk_overlap < 0 or idx.chunk_overlap >= idx.chunk_size:
            raise RuntimeConfigError(
                "chunk_overlap must satisfy 0 <= overlap < chunk_size",
                field="chunk_overlap",
            )
        if idx.faiss_hnsw_m < 4 or idx.faiss_hnsw_m > 128:
            raise RuntimeConfigError(
                "faiss_hnsw_m must be in [4, 128]", field="faiss_hnsw_m"
            )
        if idx.faiss_hnsw_ef_construction < 4:
            raise RuntimeConfigError(
                "faiss_hnsw_ef_construction must be >= 4",
                field="faiss_hnsw_ef_construction",
            )
        if idx.faiss_hnsw_ef_search < 1:
            raise RuntimeConfigError(
                "faiss_hnsw_ef_search must be >= 1", field="faiss_hnsw_ef_search"
            )
        if not idx.embedding.model:
            raise RuntimeConfigError("embedding.model is required",
                                     field="embedding.model")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._config.to_dict(), indent=2), encoding="utf-8")
        import os
        os.replace(tmp, self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read runtime_config (using boot defaults): %s", exc)
            return
        if data.get("version") != RUNTIME_CONFIG_VERSION:
            logger.warning(
                "Runtime config version mismatch (got %s, expected %s); ignoring file.",
                data.get("version"), RUNTIME_CONFIG_VERSION,
            )
            return
        try:
            if "live" in data:
                live = LiveQuerySettings.from_dict(data["live"])
                self._validate_live(live)
                self._config.live = live
            if "staged_index" in data:
                staged = IndexConstructionSettings.from_dict(data["staged_index"])
                self._validate_index(staged)
                self._config.staged_index = staged
            if "active_index" in data:
                active = IndexConstructionSettings.from_dict(data["active_index"])
                self._validate_index(active)
                self._config.active_index = active
        except RuntimeConfigError as exc:
            # Invalid persisted file → boot defaults win. Log it.
            logger.warning("runtime_config validation failed (%s); keeping defaults.", exc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict where ``patch`` is recursively merged into ``base``."""
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
