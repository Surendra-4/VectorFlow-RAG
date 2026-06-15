# src/interfaces.py

"""
Structural interfaces (PEP 544 ``Protocol``) for swappable backends.

These contracts are:

- **Runtime-checkable** — ``isinstance(obj, Protocol)`` works
- **Inheritance-free** — any duck-typed class satisfies them
- **Forward-looking** — they describe both today's implementations
  (ChromaDB, Ollama) and tomorrow's (FAISS HNSW, Redis cache,
  cross-encoder rerankers, remote LLM endpoints)

Purpose: define a single point where future swap-in components must conform,
so feature work can land independently without scattered shape changes.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np

# Type aliases used across the project. Kept loose by design — concrete
# implementations may use numpy, lists, or torch tensors.
Embedding = Union[np.ndarray, Sequence[float]]
Embeddings = Union[np.ndarray, Sequence[Sequence[float]]]


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Embedding model contract."""

    model_name: str
    dimension: int

    def encode(
        self,
        texts: Union[str, Iterable[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        input_type: Optional[str] = None,
    ) -> np.ndarray:
        """``input_type`` is "query"/"passage"/None — advisory for asymmetric
        models (e.g. e5). Symmetric models ignore it."""
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """
    Vector-store contract — implemented today by ChromaDB; FAISS HNSW
    backend will conform to the same shape.
    """

    persist_directory: str
    collection_name: str

    def add_documents(
        self,
        texts: Sequence[str],
        embeddings: Embeddings,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> None: ...

    def search(
        self,
        query_embedding: Embedding,
        n_results: int = 5,
    ) -> Dict[str, List[Any]]:
        """
        Returns a dict with at minimum these keys:

        * ``documents``: list of chunk texts
        * ``distances``: list of floats (monotonic in similarity; smaller = closer)
        * ``metadatas``: list of per-chunk metadata dicts
        * ``ids``: list of stable chunk identifiers (Phase 3.5+).
          For BC, callers should treat missing/empty ``ids`` as a signal
          to fall back to text-based join.

        All four lists are the same length and aligned positionally.
        """
        ...

    def get_embeddings(self, ids: Sequence[str]) -> Optional[np.ndarray]:
        """Return stored embeddings for ``ids`` as a float32 ``(n, d)`` array,
        aligned positionally to ``ids``.

        Returns ``None`` when the exact stored vectors can't be reproduced —
        any id is missing, or the backend can't reconstruct losslessly. Callers
        treat ``None`` as "recompute the embeddings instead", so this must never
        return approximate or mis-ordered vectors.
        """
        ...

    def delete_collection(self, name: Optional[str] = None) -> None: ...

    def get_stats(self) -> Dict[str, Any]: ...


@runtime_checkable
class BM25RetrieverProtocol(Protocol):
    """Sparse keyword retriever contract."""

    corpus: Sequence[str]

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]: ...


@runtime_checkable
class CacheProtocol(Protocol):
    """
    Cache backend contract — implemented later by an in-memory cache and
    a Redis-backed cache.
    """

    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def clear(self) -> None: ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """
    Reranker contract — applied to candidate documents after initial
    retrieval. Implemented later by a cross-encoder reranker.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]: ...


@runtime_checkable
class LLMClientProtocol(Protocol):
    """LLM generation contract — implemented today by ``OllamaClient``."""

    model: str

    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str: ...
