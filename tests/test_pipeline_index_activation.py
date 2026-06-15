# tests/test_pipeline_index_activation.py

"""
Phase 13a — pipeline-level named-index activation.

Verifies that activating a named index swaps only the vector half of hybrid
retrieval (BM25 + chunk-id join preserved), that provenance still resolves,
that reverting to default works, that re-ingestion resets activation, and that
the retrieval-cache key differs by active index.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.indexing import IndexManager, IndexProfile, IndexRegistry
from src.rag_pipeline import RAGPipeline


DOCS = [
    "Photosynthesis converts light energy into chemical energy in chloroplasts.",
    "Mitochondria generate ATP via oxidative phosphorylation in eukaryotic cells.",
    "Reciprocal rank fusion combines ranked lists from multiple retrievers.",
    "BM25 is a sparse keyword retrieval algorithm based on TF-IDF.",
    "Vector embeddings map text to a continuous high-dimensional space.",
]
META = [{"document_name": f"doc{i}.txt", "source_path": f"/tmp/doc{i}.txt"} for i in range(len(DOCS))]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    persist = tmp_path_factory.mktemp("p13_pipeline")
    rag = RAGPipeline(index_dir=str(persist), enable_cache=False)
    rag.ingest_documents(DOCS, metadatas=META)
    return rag


@pytest.fixture
def manager(tmp_path):
    return IndexManager(
        registry=IndexRegistry(path=tmp_path / "reg.json"),
        indices_root=tmp_path / "named",
    )


def _build_named_from_pipeline(pipeline, manager, name, index_type="flat"):
    """Build a named index from the pipeline's *actual* chunk records."""
    texts, ids, metas = pipeline.iter_chunk_records()
    emb = pipeline.embedder.encode(texts, show_progress=False, input_type="passage")
    profile = IndexProfile(
        name=name, backend="faiss", index_type=index_type,
        embedding_model=pipeline.embedder.model_name,
        vector_dimension=pipeline.embedder.dimension,
        corpus_fingerprint=pipeline.corpus_fingerprint,
    )
    manager.create_index(profile, texts, np.asarray(emb, dtype=np.float32), metas, ids)
    return profile


# --------------------------------------------------------------------------- #
# Identity retention
# --------------------------------------------------------------------------- #


def test_pipeline_retains_chunk_records(pipeline):
    texts, ids, metas = pipeline.iter_chunk_records()
    assert len(texts) == len(ids) == len(metas)
    assert all(cid for cid in ids)  # real chunk_ids, not empty
    assert pipeline.active_index_name is None  # default is live after ingest
    assert pipeline._default_vector_store is not None


# --------------------------------------------------------------------------- #
# Ingest-time embedding reuse (index build optimization)
# --------------------------------------------------------------------------- #


def test_get_corpus_embeddings_matches_reembedding(pipeline):
    """The reused vectors must equal what re-embedding the chunks produces, so
    a named index built from them is identical to one built by re-embedding."""
    texts, ids, _ = pipeline.iter_chunk_records()
    reused = pipeline.get_corpus_embeddings(ids)
    assert reused is not None
    assert reused.shape == (len(texts), pipeline.embedder.dimension)

    recomputed = np.asarray(
        pipeline.embedder.encode(texts, show_progress=False, input_type="passage"),
        dtype=np.float32,
    )
    assert np.allclose(reused, recomputed, atol=1e-6)


def test_get_corpus_embeddings_aligns_to_requested_id_order(pipeline):
    texts, ids, _ = pipeline.iter_chunk_records()
    full = pipeline.get_corpus_embeddings(ids)
    reversed_vecs = pipeline.get_corpus_embeddings(list(reversed(ids)))
    assert reversed_vecs is not None
    # Row i of the reversed request must equal row -(i+1) of the in-order one.
    assert np.allclose(reversed_vecs, full[::-1], atol=1e-6)


def test_get_corpus_embeddings_missing_id_returns_none(pipeline):
    assert pipeline.get_corpus_embeddings(["no-such-chunk-id"]) is None


# --------------------------------------------------------------------------- #
# Activation swaps vector half, keeps BM25 + provenance
# --------------------------------------------------------------------------- #


def test_activate_named_index_preserves_results_and_provenance(pipeline, manager):
    profile = _build_named_from_pipeline(pipeline, manager, "flat_named")
    store = manager.load_index("flat_named")

    query = "How do retrievers combine ranked lists?"
    before = pipeline.search(query, k=3)
    assert pipeline.active_index_name is None

    pipeline.activate_named_index(profile.name, store)
    assert pipeline.active_index_name == "flat_named"

    after = pipeline.search(query, k=3)
    # Same chunk_id space (BM25 + named vector store share ids) → top hit holds.
    assert after[0]["chunk_id"] == before[0]["chunk_id"]
    # Provenance still resolves from metadata carried into the named index.
    assert after[0].get("document_name", "").startswith("doc")

    # Revert.
    pipeline.activate_default_index()
    assert pipeline.active_index_name is None
    reverted = pipeline.search(query, k=3)
    assert reverted[0]["chunk_id"] == before[0]["chunk_id"]


def test_hybrid_join_still_works_after_switch(pipeline, manager):
    _build_named_from_pipeline(pipeline, manager, "join_named", index_type="flat")
    store = manager.load_index("join_named")
    pipeline.activate_named_index("join_named", store)
    try:
        results = pipeline.search("BM25 keyword retrieval TF-IDF", k=3)
        # A keyword-heavy query should still surface the BM25 doc with a
        # vector_rank or bm25_rank set (i.e. fusion ran, not vector-only).
        assert any(r.get("bm25_rank") is not None for r in results)
    finally:
        pipeline.activate_default_index()


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_activate_before_ingest_raises(tmp_path):
    rag = RAGPipeline(index_dir=str(tmp_path / "empty"), enable_cache=False)
    with pytest.raises(ValueError):
        rag.activate_named_index("x", object())
    with pytest.raises(ValueError):
        rag.activate_default_index()


def test_reingestion_resets_activation(tmp_path, manager):
    rag = RAGPipeline(index_dir=str(tmp_path / "reset"), enable_cache=False)
    rag.ingest_documents(DOCS, metadatas=META)
    texts, ids, metas = rag.iter_chunk_records()
    emb = rag.embedder.encode(texts, show_progress=False, input_type="passage")
    profile = IndexProfile(
        name="tmp_named", backend="faiss", index_type="flat",
        embedding_model=rag.embedder.model_name, vector_dimension=rag.embedder.dimension,
    )
    manager.create_index(profile, texts, np.asarray(emb, dtype=np.float32), metas, ids)
    rag.activate_named_index("tmp_named", manager.load_index("tmp_named"))
    assert rag.active_index_name == "tmp_named"

    # Re-ingest → back to default.
    rag.ingest_documents(DOCS, metadatas=META)
    assert rag.active_index_name is None


# --------------------------------------------------------------------------- #
# Cache-key isolation
# --------------------------------------------------------------------------- #


def test_retrieval_cache_key_differs_by_active_index(tmp_path):
    from src.cache.factory import make_cache
    from src.config import CacheSettings

    cache = make_cache(CacheSettings(backend="memory"))
    rag = RAGPipeline(index_dir=str(tmp_path / "cachekey"), cache=cache, enable_cache=True)
    rag.ingest_documents(DOCS, metadatas=META)
    assert rag.cache.backend_name != "null"
    key_default = rag._retrieval_cache_key("q", 3)
    rag.active_index_name = "some_named"
    key_named = rag._retrieval_cache_key("q", 3)
    assert key_default and key_named and key_default != key_named
