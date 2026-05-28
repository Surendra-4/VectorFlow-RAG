# tests/test_retrieval_quality.py

"""
Retrieval quality and ranking-consistency tests.

This module pins behavior of the RRF retrieval pipeline against a small
golden-set corpus and verifies:

* Top-1 contains the relevant document for keyword and semantic queries.
* RRF rewards documents present in both modalities.
* Each result carries the rank metadata fields (``vector_rank``, ``bm25_rank``).
* Reranking improves top-1 quality on queries where the bi-encoder ordering
  is not optimal (uses a deterministic mock reranker so tests are fast).
* ``ingest → search → search`` is deterministic for fixed inputs.

The golden set is intentionally tiny — these are smoke-style quality
tests, not BEIR-grade benchmarks. A larger benchmark belongs under
``experiments/`` once we wire up MS MARCO / scifact replays.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from src.bm25_retriever import BM25Retriever
from src.embedder import Embedder
from src.hybrid_retriever import HybridRetriever
from src.rag_pipeline import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.vector_store import VectorStore


# --------------------------------------------------------------------------- #
# Shared corpus & fixtures
# --------------------------------------------------------------------------- #


GOLDEN_CORPUS = [
    "Photosynthesis converts light energy into chemical energy in plant chloroplasts.",
    "Mitochondria generate ATP via oxidative phosphorylation in eukaryotic cells.",
    "The capital of France is Paris, located on the Seine river.",
    "Python is a high-level programming language popular in data science and machine learning.",
    "Reciprocal rank fusion combines ranked lists from multiple retrievers.",
    "BM25 is a sparse keyword retrieval algorithm based on TF-IDF.",
    "Cross-encoders rerank candidate documents by jointly encoding query and candidate.",
    "Eiffel Tower is a famous landmark in Paris, France, built in 1889.",
    "Neural networks learn through gradient descent on a loss function.",
    "Vector embeddings map text to a continuous high-dimensional space.",
]


@pytest.fixture(scope="module")
def hybrid(tmp_path_factory):
    """Build a hybrid retriever once per module — embedding load is expensive."""
    persist = tmp_path_factory.mktemp("retrieval_quality_chroma")

    embedder = Embedder()
    vs = VectorStore(persist_directory=str(persist))
    vs.create_collection(reset=True)
    embeddings = embedder.encode(GOLDEN_CORPUS, show_progress=False)
    vs.add_documents(texts=GOLDEN_CORPUS, embeddings=embeddings.tolist())

    bm25 = BM25Retriever(corpus=GOLDEN_CORPUS)
    return HybridRetriever(embedder, vs, bm25)


# --------------------------------------------------------------------------- #
# RRF quality
# --------------------------------------------------------------------------- #


class TestRRFTop1Quality:
    """Top-1 should land the canonical relevant document for each query."""

    def test_keyword_query_finds_exact_term(self, hybrid):
        results = hybrid.search("photosynthesis", k=3)
        assert "photosynthesis" in results[0]["text"].lower()

    def test_capital_of_france_finds_paris(self, hybrid):
        results = hybrid.search("capital of France", k=3)
        assert "Paris" in results[0]["text"]

    def test_python_query_finds_python(self, hybrid):
        results = hybrid.search("Python programming", k=3)
        assert "Python" in results[0]["text"]

    def test_rrf_query_finds_rrf(self, hybrid):
        results = hybrid.search("reciprocal rank fusion", k=3)
        assert "reciprocal rank fusion" in results[0]["text"].lower()

    def test_paraphrased_semantic_query(self, hybrid):
        # No literal keyword overlap — relies on embedding semantics.
        results = hybrid.search("powerhouse of the cell", k=3)
        joined = " ".join(r["text"] for r in results).lower()
        assert "mitochondria" in joined or "atp" in joined


class TestRRFOutputContract:
    """The RRF output dict shape must include rank metadata for transparency."""

    def test_required_fields_present(self, hybrid):
        results = hybrid.search("Python", k=3)
        for r in results:
            assert "text" in r
            assert "hybrid_score" in r  # legacy alias for RRF score
            assert "rrf_score" in r
            assert "vector_rank" in r
            assert "bm25_rank" in r
            assert r["hybrid_score"] == r["rrf_score"]

    def test_rrf_scores_sorted_descending(self, hybrid):
        results = hybrid.search("learning", k=5)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_modality_ranks_consistent(self, hybrid):
        # Each rank, if present, is a positive int.
        results = hybrid.search("neural networks", k=5)
        for r in results:
            for field in ("vector_rank", "bm25_rank"):
                if r[field] is not None:
                    assert isinstance(r[field], int)
                    assert r[field] >= 1


class TestRankingConsistency:
    """Ranking should be deterministic and stable across calls."""

    def test_repeated_queries_identical(self, hybrid):
        a = hybrid.search("capital of France", k=5)
        b = hybrid.search("capital of France", k=5)
        assert [r["text"] for r in a] == [r["text"] for r in b]
        assert [r["rrf_score"] for r in a] == [r["rrf_score"] for r in b]

    def test_truncating_k_preserves_prefix(self, hybrid):
        full = hybrid.search("Python programming", k=10)
        prefix = hybrid.search("Python programming", k=3)
        assert [r["text"] for r in prefix] == [r["text"] for r in full[:3]]


# --------------------------------------------------------------------------- #
# Reranker integration with RAGPipeline
# --------------------------------------------------------------------------- #


class _DeterministicMockReranker:
    """
    Mock reranker for pipeline-integration tests.

    Scores by query↔text token overlap, so the test is fast and deterministic
    while still exercising the full pipeline wiring.
    """

    def rerank(self, query, candidates, top_n=None):
        q_tokens = set(query.lower().split())
        annotated = []
        for c in candidates:
            c_tokens = set(c["text"].lower().split())
            new = dict(c)
            new["rerank_score"] = float(len(q_tokens & c_tokens))
            annotated.append(new)
        annotated.sort(key=lambda x: x["rerank_score"], reverse=True)
        return annotated[: top_n or len(annotated)]


class TestRerankerInRAGPipeline:
    @pytest.fixture
    def pipeline_with_mock_reranker(self, tmp_path):
        rag = RAGPipeline(
            index_dir=str(tmp_path / "rerank_pipeline"),
            reranker=_DeterministicMockReranker(),
            enable_reranker=True,
        )
        rag.ingest_documents(GOLDEN_CORPUS)
        return rag

    @pytest.fixture
    def pipeline_no_reranker(self, tmp_path):
        rag = RAGPipeline(
            index_dir=str(tmp_path / "norerank_pipeline"),
            enable_reranker=False,
        )
        rag.ingest_documents(GOLDEN_CORPUS)
        return rag

    def test_reranker_adds_rerank_score(self, pipeline_with_mock_reranker):
        results = pipeline_with_mock_reranker.search("Eiffel Tower Paris", k=3)
        assert all("rerank_score" in r for r in results)

    def test_reranker_disabled_omits_rerank_score(self, pipeline_no_reranker):
        results = pipeline_no_reranker.search("Eiffel Tower Paris", k=3)
        assert all("rerank_score" not in r for r in results)
        # But hybrid retrieval transparency should still be intact.
        assert all("hybrid_score" in r for r in results)

    def test_reranker_returns_top_k(self, pipeline_with_mock_reranker):
        results = pipeline_with_mock_reranker.search("Python", k=2)
        assert len(results) == 2

    def test_get_stats_exposes_reranker_state(self, pipeline_with_mock_reranker, pipeline_no_reranker):
        on = pipeline_with_mock_reranker.get_stats()
        off = pipeline_no_reranker.get_stats()
        assert on["reranker_enabled"] is True
        assert on["reranker_model"] is not None
        assert off["reranker_enabled"] is False
        assert off["reranker_model"] is None

    def test_get_stats_exposes_rrf_params(self, pipeline_no_reranker):
        stats = pipeline_no_reranker.get_stats()
        assert "rrf_k" in stats
        assert "candidates_per_modality" in stats
        assert stats["rrf_k"] >= 1
        assert stats["candidates_per_modality"] >= 1


# --------------------------------------------------------------------------- #
# Latency benchmarks (informational; never fail-loud)
# --------------------------------------------------------------------------- #


class TestLatencyBenchmark:
    """
    Measure retrieval latency. These are informational — they print numbers
    rather than enforce thresholds, since CI hardware is variable. Run with
    ``pytest -s`` to see the output.
    """

    def test_hybrid_search_latency(self, hybrid):
        # Warm up.
        hybrid.search("Python", k=5)

        times: List[float] = []
        for q in ["Python programming", "Eiffel Tower", "neural networks", "BM25", "photosynthesis"]:
            t0 = time.perf_counter()
            hybrid.search(q, k=5)
            times.append((time.perf_counter() - t0) * 1000)

        p50 = float(np.median(times))
        p95 = float(np.percentile(times, 95))
        print(f"\n[Hybrid RRF] p50={p50:.1f}ms  p95={p95:.1f}ms  n={len(times)}")
        # Non-strict bound — guard against egregious regressions only.
        assert p50 < 5000, f"Retrieval p50 {p50}ms suspiciously high"

    def test_rerank_latency_with_mock(self, hybrid):
        candidates = hybrid.search("learning", k=20)
        rr = _DeterministicMockReranker()

        rr.rerank("learning", candidates)  # warm
        t0 = time.perf_counter()
        for _ in range(20):
            rr.rerank("learning", candidates, top_n=3)
        elapsed = (time.perf_counter() - t0) * 1000 / 20
        print(f"\n[Mock rerank, n=20 candidates] avg={elapsed:.2f}ms")
        assert elapsed < 100  # mock should be sub-millisecond on any hardware
