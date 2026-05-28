# src/faiss_store.py

"""
FAISS-backed vector store conforming to :class:`src.interfaces.VectorStoreProtocol`.

Default index type is HNSW (graph-based ANN) with cosine similarity via
inner product on normalized embeddings. Flat (exact) and IVF indexes are
also available via the ``index_type`` argument or ``VFR_VECTOR_STORE__FAISS_INDEX_TYPE``.

Persistence layout::

    <persist_directory>/
      <collection>.faiss   -- binary FAISS index (faiss.write_index format)
      <collection>.json    -- sidecar: documents, metadatas, ids, dim, index_type

The sidecar is the source of truth for documents/metadata/ids; the FAISS
binary holds only embeddings + graph. On reload, both files must be
present and consistent. Inconsistent state → fresh empty index.

Internal state is **ID-first**: a string ID maps to a FAISS internal
position (0-indexed). This positions us for the planned chunk-id /
document-id migration without restructuring later.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

SIDECAR_VERSION = 1


class FAISSVectorStore:
    """FAISS-backed vector store. Drop-in replacement for ``VectorStore``."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        index_type: Optional[Literal["hnsw", "flat", "ivf"]] = None,
        hnsw_m: Optional[int] = None,
        hnsw_ef_construction: Optional[int] = None,
        hnsw_ef_search: Optional[int] = None,
    ):
        # Lazy import — keeps `import src.*` cheap when FAISS isn't used.
        import faiss

        self._faiss = faiss

        cfg = get_settings().vector_store

        # Resolve persistence path. Tolerate Windows-style separators.
        if persist_directory is None:
            raw = cfg.persist_directory.parent / "faiss"
        else:
            raw = persist_directory
        self.persist_directory: str = str(Path(str(raw).replace("\\", "/")))
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.collection_name: str = collection_name or cfg.collection_name
        self.index_type: str = index_type or cfg.faiss_index_type

        self.hnsw_m: int = hnsw_m if hnsw_m is not None else cfg.faiss_hnsw_m
        self.hnsw_ef_construction: int = (
            hnsw_ef_construction if hnsw_ef_construction is not None else cfg.faiss_hnsw_ef_construction
        )
        self.hnsw_ef_search: int = (
            hnsw_ef_search if hnsw_ef_search is not None else cfg.faiss_hnsw_ef_search
        )

        # Sidecar tables — ID-first.
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self._id_to_pos: Dict[str, int] = {}
        self.embedding_dim: int = 0
        self.index = None  # type: ignore[assignment]  # built on first add or load

        self._index_path = Path(self.persist_directory) / f"{self.collection_name}.faiss"
        self._sidecar_path = Path(self.persist_directory) / f"{self.collection_name}.json"

        if self._index_path.exists() and self._sidecar_path.exists():
            try:
                self._load()
            except Exception as exc:
                logger.warning("Failed to load FAISS index, starting fresh: %s", exc)
                self._reset_in_memory()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _reset_in_memory(self) -> None:
        self.documents = []
        self.metadatas = []
        self.ids = []
        self._id_to_pos = {}
        self.embedding_dim = 0
        self.index = None

    def _build_index(self, dim: int):
        """Construct a fresh FAISS index of the configured type."""
        faiss = self._faiss
        if self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = self.hnsw_ef_construction
            idx.hnsw.efSearch = self.hnsw_ef_search
        elif self.index_type == "flat":
            idx = faiss.IndexFlatIP(dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            # nlist heuristic: ~sqrt(n_expected). Safe default with a small floor.
            nlist = 16
            idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unsupported FAISS index_type: {self.index_type!r}")
        return idx

    def _to_float32(self, embeddings: Sequence[Sequence[float]]) -> np.ndarray:
        """Coerce embeddings to a contiguous (n, d) float32 array."""
        arr = np.asarray(
            [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings],
            dtype=np.float32,
        )
        if arr.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {arr.shape}")
        return np.ascontiguousarray(arr)

    def _persist(self) -> None:
        """Atomic-ish persist: write to .tmp then rename."""
        if self.index is None:
            return

        tmp_index = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        tmp_sidecar = self._sidecar_path.with_suffix(self._sidecar_path.suffix + ".tmp")

        self._faiss.write_index(self.index, str(tmp_index))

        sidecar = {
            "version": SIDECAR_VERSION,
            "collection_name": self.collection_name,
            "dimension": self.embedding_dim,
            "index_type": self.index_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        with open(tmp_sidecar, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False)

        os.replace(tmp_index, self._index_path)
        os.replace(tmp_sidecar, self._sidecar_path)

    def _load(self) -> None:
        """Restore index + sidecar from disk."""
        with open(self._sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)

        if sidecar.get("version") != SIDECAR_VERSION:
            raise RuntimeError(f"Unsupported sidecar version: {sidecar.get('version')}")

        self.documents = sidecar["documents"]
        self.metadatas = sidecar["metadatas"]
        self.ids = sidecar["ids"]
        self._id_to_pos = {iid: i for i, iid in enumerate(self.ids)}
        self.embedding_dim = sidecar["dimension"]
        self.index_type = sidecar["index_type"]
        self.hnsw_m = sidecar.get("hnsw_m", self.hnsw_m)
        self.hnsw_ef_construction = sidecar.get("hnsw_ef_construction", self.hnsw_ef_construction)

        self.index = self._faiss.read_index(str(self._index_path))
        # Apply current efSearch (search-time tunable, not persisted as authoritative).
        if self.index_type == "hnsw" and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.hnsw_ef_search

        logger.info(
            "Loaded FAISS index '%s' n=%d dim=%d type=%s",
            self.collection_name, len(self.documents), self.embedding_dim, self.index_type,
        )

    # ------------------------------------------------------------------ #
    # Public API (matches VectorStoreProtocol)
    # ------------------------------------------------------------------ #

    def add_documents(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        if len(texts) == 0:
            return

        emb = self._to_float32(embeddings)
        n, dim = emb.shape

        if self.index is None:
            self.embedding_dim = dim
            self.index = self._build_index(dim)
        elif self.embedding_dim != dim:
            raise ValueError(
                f"Embedding dimension mismatch: existing={self.embedding_dim}, new={dim}"
            )

        # IVF requires training before adds (Flat / HNSW don't).
        if self.index_type == "ivf" and not self.index.is_trained:
            # Train on what we have; small corpora may have fewer vectors than nlist.
            n_train = max(emb.shape[0], 1)
            self.index.train(emb[:n_train])

        # Generate IDs / metadatas if absent.
        if ids is None:
            base = len(self.documents)
            ids_list = [f"id_{base + i}" for i in range(n)]
        else:
            ids_list = list(ids)
        if metadatas is None:
            meta_list = [{"source": "manual"} for _ in range(n)]
        else:
            meta_list = list(metadatas)

        if len(ids_list) != n or len(meta_list) != n or len(texts) != n:
            raise ValueError(
                f"Length mismatch: texts={len(texts)} ids={len(ids_list)} "
                f"metadatas={len(meta_list)} embeddings={n}"
            )

        self.index.add(emb)

        for text, meta, iid in zip(texts, meta_list, ids_list):
            self._id_to_pos[iid] = len(self.documents)
            self.documents.append(text)
            self.metadatas.append(meta)
            self.ids.append(iid)

        self._persist()
        logger.debug(
            "FAISS add: n=%d total=%d collection=%s", n, len(self.documents), self.collection_name
        )

    def search(self, query_embedding, n_results: int = 5) -> Dict[str, List[Any]]:
        if self.index is None or len(self.documents) == 0:
            return {"documents": [], "distances": [], "metadatas": [], "ids": []}

        q = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        q_arr = np.ascontiguousarray(np.asarray([q], dtype=np.float32))

        n_results = max(1, min(int(n_results), len(self.documents)))
        sims, indices = self.index.search(q_arr, n_results)

        documents: List[str] = []
        distances: List[float] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        for pos, sim in zip(indices[0], sims[0]):
            if pos < 0 or pos >= len(self.documents):
                # FAISS returns -1 for empty slots when fewer than n_results match.
                continue
            documents.append(self.documents[pos])
            # Convert IP similarity → cosine distance for monotonicity parity
            # with downstream code that expects "smaller = closer".
            distances.append(float(1.0 - sim))
            metadatas.append(self.metadatas[pos])
            ids.append(self.ids[pos])

        return {"documents": documents, "distances": distances, "metadatas": metadatas, "ids": ids}

    def delete_collection(self, name: Optional[str] = None) -> None:
        target_name = name or self.collection_name
        self._reset_in_memory()
        # Drop disk artifacts for the target collection.
        index_path = Path(self.persist_directory) / f"{target_name}.faiss"
        sidecar_path = Path(self.persist_directory) / f"{target_name}.json"
        try:
            if index_path.exists():
                index_path.unlink()
            if sidecar_path.exists():
                sidecar_path.unlink()
            logger.info("FAISS collection '%s' deleted", target_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not delete FAISS collection '%s': %s", target_name, exc)

    def create_collection(self, reset: bool = False):
        """ChromaDB-style helper. Returns ``self`` for parity with ``VectorStore``."""
        if reset:
            self.delete_collection()
        return self

    def get_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "num_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "total_documents": len(self.documents),
            "backend": "faiss",
            "index_type": self.index_type,
        }

    # ------------------------------------------------------------------ #
    # FAISS-specific extras (not in the protocol; useful for future work)
    # ------------------------------------------------------------------ #

    def set_ef_search(self, ef_search: int) -> None:
        """Adjust HNSW search depth at query time. No effect for non-HNSW indexes."""
        self.hnsw_ef_search = ef_search
        if self.index is not None and self.index_type == "hnsw" and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search

    def __len__(self) -> int:
        return len(self.documents)
