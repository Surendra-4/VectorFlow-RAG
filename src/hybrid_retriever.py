# src/hybrid_retriever.py

"""
Hybrid retrieval combining BM25 and dense vector search via RRF.

Phase 3.5: RRF joins on stable ``chunk_id`` when both modalities expose
IDs; otherwise falls back to text-based join (legacy behavior). This
fixes two classes of bug at the join boundary:

* Identical text in different source documents being collapsed into a
  single retrieval entry (source attribution lost).
* Re-ingestion producing duplicate vectors that wouldn't dedupe under
  text equality.

Each returned result carries:

* ``text``         — raw chunk text (display fidelity)
* ``chunk_id``     — stable hash; ``None`` only when both modalities
                     returned no IDs (pure-string corpora)
* ``doc_id``       — parent document id (from chunk metadata; ``None`` if absent)
* ``hybrid_score`` — RRF score (legacy field name, kept for BC)
* ``rrf_score``    — alias of ``hybrid_score``
* ``vector_rank``  — rank in dense list (``None`` if not retrieved by it)
* ``bm25_rank``    — rank in BM25 list (``None`` if not retrieved by it)
* ``metadata``     — full metadata dict from the corpus entry
* plus surfaced provenance keys at the top level:
    ``document_name``, ``source_path``, ``page_number``, ``chunk_index``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.bm25_retriever import BM25Retriever
from src.config import get_settings
from src.embedder import Embedder
from src.logging_setup import get_logger
from src.rrf import rrf_with_ranks
from src.vector_store import VectorStore

logger = get_logger(__name__)

# Provenance keys we lift to the top level of each result dict for
# easy consumption by downstream callers (citations, UI, LLM prompts).
_PROVENANCE_KEYS_TO_LIFT = ("document_name", "source_path", "page_number", "chunk_index")


class HybridRetriever:
    """Hybrid retriever using Reciprocal Rank Fusion over stable chunk IDs."""

    def __init__(
        self,
        embedder,
        vector_store,
        bm25_retriever,
        alpha: Optional[float] = None,
        candidates_per_modality: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ):
        cfg = get_settings().retrieval
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25_retriever

        # Stored as metadata only (alpha-fusion replaced by RRF in Phase 1).
        self.alpha = cfg.alpha if alpha is None else alpha

        self.candidates_per_modality = (
            cfg.candidates_per_modality if candidates_per_modality is None else candidates_per_modality
        )
        self.rrf_k = cfg.rrf_k if rrf_k is None else rrf_k

    def search(self, query: str, k: Optional[int] = None, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-``k`` results via RRF over BM25 + dense modalities.

        Identity strategy:

        * If both modalities expose chunk IDs, RRF joins on IDs.
        * Otherwise it joins on text. The fallback maintains BC for
          string corpora and tests that don't set up IDs.
        """
        k_final = n_results or k or self.candidates_per_modality
        pool_size = max(self.candidates_per_modality, k_final)

        # ---- Dense retrieval --------------------------------------------- #
        # input_type="query" lets asymmetric embedders (e.g. e5) apply the
        # query prefix; symmetric models ignore it (English path unchanged).
        query_emb = self.embedder.encode(query, show_progress=False, input_type="query")[0]
        dense = self.vector_store.search(query_emb, n_results=pool_size)
        dense_docs: List[str] = list(dense.get("documents", []))
        dense_ids: List[Any] = list(dense.get("ids", []) or [])
        dense_metas: List[Dict[str, Any]] = list(dense.get("metadatas", []) or [])

        # ---- Sparse retrieval -------------------------------------------- #
        bm25_pool = min(pool_size, len(self.bm25.corpus))
        bm25_results = self.bm25.search(query, k=bm25_pool) if bm25_pool > 0 else []
        bm25_docs: List[str] = [r["text"] for r in bm25_results]
        bm25_ids: List[Any] = [r.get("chunk_id") for r in bm25_results]
        bm25_metas: List[Optional[Dict[str, Any]]] = [r.get("metadata") for r in bm25_results]

        # ---- Pick a join key strategy ------------------------------------ #
        # Use IDs only when both modalities provided complete ID lists.
        dense_has_ids = bool(dense_ids) and all(i is not None for i in dense_ids)
        bm25_has_ids = bool(bm25_ids) and all(i is not None for i in bm25_ids)
        use_ids = dense_has_ids and bm25_has_ids

        # Build a lookup from join-key → (text, metadata) so we can re-hydrate
        # full result objects after RRF returns just keys + scores.
        lookup: Dict[Any, Dict[str, Any]] = {}
        if use_ids:
            for iid, doc, meta in zip(dense_ids, dense_docs, dense_metas):
                lookup.setdefault(iid, {"text": doc, "metadata": meta or {}})
            for iid, doc, meta in zip(bm25_ids, bm25_docs, bm25_metas):
                lookup.setdefault(iid, {"text": doc, "metadata": meta or {}})
            vector_ranked: List[Any] = dense_ids
            bm25_ranked: List[Any] = bm25_ids
        else:
            for doc, meta in zip(dense_docs, dense_metas):
                lookup.setdefault(doc, {"text": doc, "metadata": meta or {}})
            for doc, meta in zip(bm25_docs, bm25_metas):
                lookup.setdefault(doc, {"text": doc, "metadata": meta or {}})
            vector_ranked = dense_docs
            bm25_ranked = bm25_docs

        # ---- Reciprocal Rank Fusion -------------------------------------- #
        fused = rrf_with_ranks(
            {"vector": vector_ranked, "bm25": bm25_ranked},
            k=self.rrf_k,
        )

        results: List[Dict[str, Any]] = []
        for entry in fused[:k_final]:
            join_key = entry["item"]
            ranks = entry["ranks"]
            hydrated = lookup.get(join_key, {"text": "", "metadata": {}})
            meta = hydrated["metadata"] or {}

            result: Dict[str, Any] = {
                "text": hydrated["text"],
                "chunk_id": join_key if use_ids else meta.get("chunk_id"),
                "doc_id": meta.get("document_id"),
                "hybrid_score": float(entry["score"]),  # legacy alias
                "rrf_score": float(entry["score"]),
                "vector_rank": ranks.get("vector"),
                "bm25_rank": ranks.get("bm25"),
                "metadata": meta,
            }
            for key in _PROVENANCE_KEYS_TO_LIFT:
                result[key] = meta.get(key)
            results.append(result)

        logger.debug(
            "Hybrid RRF: query=%r pool=%d vec=%d bm25=%d returned=%d join=%s",
            query, pool_size, len(vector_ranked), len(bm25_ranked),
            len(results), "ids" if use_ids else "text",
        )
        return results


if __name__ == "__main__":
    docs = [
        "Apple is a fruit",
        "Banana is yellow",
        "Cherry is red",
        "Dates are sweet",
        "Elderberry is healthy",
        "Fig is delicious",
    ]

    embedder = Embedder()
    vector_store = VectorStore(str(Path("indices") / "hybrid_test"))
    vector_store.create_collection(reset=True)
    vector_store.add_documents(docs, embedder.encode(docs).tolist())

    bm25 = BM25Retriever(docs)
    hybrid = HybridRetriever(embedder, vector_store, bm25)

    print(hybrid.search("Apple", k=3))
