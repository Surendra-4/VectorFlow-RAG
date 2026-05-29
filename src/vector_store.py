# src/vector_store.py

"""
Vector Store module using ChromaDB for efficient vector storage and retrieval.

Conforms to :class:`src.interfaces.VectorStoreProtocol`. The FAISS HNSW
backend (``src.faiss_store.FAISSVectorStore``) plugs into the same protocol.

Use :func:`make_vector_store` to construct a backend-specific store from
the active configuration. Direct ``VectorStore(...)`` calls always return
the ChromaDB-backed implementation for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.interfaces import VectorStoreProtocol
from src.logging_setup import get_logger

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB-backed vector store."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        cfg = get_settings().vector_store

        # Resolve persistence path. Accept ``None``, str, or Path; normalize
        # legacy Windows-style separators so paths created on Windows still
        # work when the project is checked out on macOS/Linux.
        raw = persist_directory if persist_directory is not None else cfg.persist_directory
        self.persist_directory = str(Path(str(raw).replace("\\", "/")))

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        try:
            # anonymized_telemetry=False silences ChromaDB's posthog telemetry,
            # which both leaks usage data and (on some posthog/chromadb version
            # combos) spams "capture() takes 1 positional argument" ERROR logs.
            self.client = PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as exc:
            raise RuntimeError(f"Could not initialize chromadb client: {exc}") from exc

        self.collection_name = collection_name or cfg.collection_name
        self.collection = self._get_or_create_collection(self.collection_name)
        self.documents: list[str] = []
        self.embeddings: list[list[float]] = []
        self.embedding_dim = 0

    def _get_or_create_collection(self, name: str):
        try:
            return self.client.get_or_create_collection(name=name, metadata={"desc": "docs"})
        except Exception:
            try:
                return self.client.create_collection(name=name, metadata={"desc": "docs"})
            except Exception as exc:
                raise RuntimeError(f"Unable to create or retrieve collection '{name}': {exc}") from exc

    def create_collection(self, reset: bool = False):
        if reset:
            self.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection(self.collection_name)
        return self.collection

    def delete_collection(self, name: Optional[str] = None):
        name = name or self.collection_name
        try:
            if hasattr(self.client, "delete_collection"):
                self.client.delete_collection(name)
                logger.info("Collection '%s' deleted", name)
            elif hasattr(self.client, "get_collection"):
                col = self.client.get_collection(name)
                if col and hasattr(col, "delete"):
                    col.delete()
                    logger.info("Collection '%s' deleted", name)
        except Exception as exc:
            logger.warning("Could not delete collection '%s': %s", name, exc)

    def add_documents(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ):
        try:
            embeddings_list = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
        except Exception:
            embeddings_list = list(embeddings)

        ids = list(ids) if ids else [f"id_{i}" for i in range(len(texts))]
        raw_metadatas = list(metadatas) if metadatas is not None else [{"source": "manual"} for _ in texts]

        # ChromaDB's metadata schema only accepts str/int/float/bool values —
        # ``None`` (e.g. placeholder ``page_number``) is rejected. Strip None
        # keys here at the backend boundary so callers can use ``None`` freely
        # as a provenance placeholder.
        sanitized: List[Dict[str, Any]] = []
        for meta in raw_metadatas:
            if not meta:
                sanitized.append({"source": "manual"})
                continue
            cleaned = {k: v for k, v in meta.items() if v is not None}
            # ChromaDB also rejects empty metadata dicts in some versions —
            # ensure at least one key is present.
            if not cleaned:
                cleaned = {"source": "manual"}
            sanitized.append(cleaned)

        self.collection.add(
            documents=list(texts),
            embeddings=embeddings_list,
            metadatas=sanitized,
            ids=ids,
        )

        self.documents.extend(texts)
        self.embeddings.extend(embeddings_list)
        self.embedding_dim = len(embeddings_list[0]) if embeddings_list else 0

    def search(self, query_embedding, n_results=5):
        # Empty-store guard checks the COLLECTION, not the in-memory shadow.
        # A freshly reloaded VectorStore has self.documents=[] even when the
        # persisted collection holds data; the old shadow check would
        # short-circuit and silently break persistence.
        try:
            if self.collection.count() == 0:
                return {"documents": [], "distances": [], "metadatas": [], "ids": []}
        except Exception:
            pass

        q_emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        n_results = max(1, int(n_results))

        res = self.collection.query(query_embeddings=[q_emb], n_results=n_results)
        documents = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]

        return {
            "documents": documents,
            "distances": distances,
            "metadatas": metadatas,
            "ids": ids,
        }

    def get_stats(self):
        return {
            "collection_name": self.collection_name,
            "num_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "total_documents": len(self.documents),
            "backend": "chromadb",
        }


# --------------------------------------------------------------------------- #
# Backend factory
# --------------------------------------------------------------------------- #


def make_vector_store(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    backend: Optional[str] = None,
    **kwargs: Any,
) -> VectorStoreProtocol:
    """
    Construct a vector store using the configured backend.

    Args:
        persist_directory: Where to persist the index. Backend-specific defaults
            apply when omitted (ChromaDB → ``indices/chroma_db``; FAISS →
            ``indices/faiss``).
        collection_name: Logical collection name (matches ``settings.vector_store.collection_name``).
        backend: ``"chromadb"`` or ``"faiss"``. Defaults to ``settings.vector_store.backend``.
        **kwargs: Backend-specific options forwarded to the implementation
            (e.g. ``index_type`` for FAISS).

    Returns:
        An object implementing :class:`src.interfaces.VectorStoreProtocol`.
    """
    cfg = get_settings().vector_store
    chosen = (backend or cfg.backend).lower()

    if chosen == "chromadb":
        return VectorStore(persist_directory=persist_directory, collection_name=collection_name)
    if chosen == "faiss":
        # Local import — FAISS only loaded when actually selected.
        from src.faiss_store import FAISSVectorStore

        return FAISSVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            **kwargs,
        )
    raise ValueError(f"Unknown vector store backend: {chosen!r}")
