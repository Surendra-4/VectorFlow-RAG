# src/indexing/profile.py

"""
IndexProfile (Phase 12e) — the named, persisted identity of a vector index.

An index stops being an implicit side effect of ingestion and becomes a
first-class runtime entity you can name, list, switch between, benchmark,
export and import. The profile captures everything needed to:

* **reconstruct** the index (backend, recipe/index_type, build/search params),
* **validate compatibility** with a configuration change (embedding model +
  dimension + chunking — see Phase 12g),
* **invalidate** it when the corpus changes (corpus_fingerprint),
* **display** it (created_at / last_used / num_vectors / metrics).

The profile is a plain dataclass so it serializes to/from the registry JSON
with no framework coupling. ``build_params`` / ``search_params`` are free-form
dicts so Phase 12f can add advanced FAISS recipe parameters without a schema
change here.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CompatibilitySignature:
    """The subset of an index's identity that must match a configuration for
    the index to be *reusable* without a rebuild.

    Two indexes with the same signature are interchangeable for retrieval;
    any difference here means a rebuild is required (Phase 12g turns a
    mismatch into the "create a new index?" UX).
    """

    embedding_model: str
    vector_dimension: int
    chunk_size: int
    chunk_overlap: int
    backend: str
    normalize: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompatibilitySignature":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def matches(self, other: "CompatibilitySignature") -> bool:
        return self.to_dict() == other.to_dict()


@dataclass
class IndexProfile:
    """A named vector index and all of its reconstruction + display metadata."""

    name: str
    backend: str                      # "chromadb" | "faiss"
    index_type: str                   # "default" (chroma) | recipe id (faiss)
    embedding_model: str
    embedding_provider: str = "sentence_transformers"
    vector_dimension: int = 0
    build_params: Dict[str, Any] = field(default_factory=dict)
    search_params: Dict[str, Any] = field(default_factory=dict)
    corpus_fingerprint: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50
    normalize: bool = True
    num_vectors: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    # Free-form notes / description shown in the UI.
    description: str = ""

    # ------------------------------------------------------------------ #

    def compatibility_signature(self) -> CompatibilitySignature:
        return CompatibilitySignature(
            embedding_model=self.embedding_model,
            vector_dimension=self.vector_dimension,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            backend=self.backend,
            normalize=self.normalize,
        )

    def touch(self) -> None:
        """Mark this index as just used (updates ``last_used``)."""
        self.last_used = time.time()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["compatibility_info"] = self.compatibility_signature().to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexProfile":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        # compatibility_info is derived on serialize; ignore it on load.
        return cls(**{k: v for k, v in data.items() if k in allowed})
