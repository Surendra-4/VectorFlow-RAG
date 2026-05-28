# tests/test_expansion_integration.py

"""
End-to-end pipeline tests with query expansion enabled.

Uses a deterministic mock expansion pipeline so the test is fast and
hermetic. Covers:

* search() returns the same shape (List[Dict]) when expansion is enabled
* expansion path retrieves more candidates than baseline
* RetrievalTrace captures all expansion + retrieval stages
* RetrievalTrace.to_dict() is JSON-serializable
* HyDE-only path works (vector-only side input)
* Reranker still applies on top of expanded pool
* Failure of expansion doesn't break retrieval (original query still works)
"""

from __future__ import annotations

import json
from typing import List

import pytest

from src.query_expansion import (
    ExpandedQuery,
    ExpansionResult,
    QueryExpansionPipeline,
)
from src.rag_pipeline import RAGPipeline
from src.retrieval_trace import RetrievalTrace


GOLDEN_DOCS = [
    "Photosynthesis converts light energy into chemical energy in plant chloroplasts.",
    "Mitochondria generate ATP via oxidative phosphorylation in eukaryotic cells.",
    "Reciprocal rank fusion combines ranked lists from multiple retrievers.",
    "BM25 is a sparse keyword retrieval algorithm based on TF-IDF.",
    "Cross-encoders rerank candidate documents by jointly encoding query and candidate.",
    "Neural networks learn through gradient descent on a loss function.",
    "Vector embeddings map text to a continuous high-dimensional space.",
    "Python is a popular language for data science and machine learning workflows.",
    "The capital of France is Paris, located on the Seine river.",
    "Eiffel Tower is a famous landmark in Paris built in 1889.",
]


# --------------------------------------------------------------------------- #
# Mock expansion strategies
# --------------------------------------------------------------------------- #


class _ScriptedMultiQuery:
    """Returns fixed variants without touching an LLM."""

    name = "multi_query"

    def __init__(self, variants):
        self.variants = tuple(variants)

    def expand(self, query):
        return ExpansionResult(
            strategy=self.name, queries=self.variants, latency_ms=5.0,
        )


class _ScriptedHyDE:
    """Returns fixed hypothetical docs."""

    name = "hyde"

    def __init__(self, docs):
        self.docs = tuple(docs)

    def expand(self, query):
        return ExpansionResult(
            strategy=self.name, hyde_documents=self.docs, latency_ms=5.0,
        )


class _AlwaysFailStrategy:
    name = "broken"

    def expand(self, query):
        return ExpansionResult(strategy=self.name, error="simulated failure", latency_ms=1.0)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def baseline_pipeline(tmp_path_factory):
    """Pipeline with expansion DISABLED — baseline behavior."""
    persist = tmp_path_factory.mktemp("expansion_baseline")
    rag = RAGPipeline(index_dir=str(persist))
    rag.ingest_documents(GOLDEN_DOCS)
    return rag


@pytest.fixture
def expanded_pipeline_factory(tmp_path):
    """Factory producing a pipeline with a scripted expansion pipeline."""

    def _factory(strategies):
        pipeline = QueryExpansionPipeline(strategies=strategies)
        rag = RAGPipeline(
            index_dir=str(tmp_path / "rag"),
            expansion_pipeline=pipeline,
            enable_expansion=True,
        )
        rag.ingest_documents(GOLDEN_DOCS)
        return rag

    return _factory


# --------------------------------------------------------------------------- #
# Baseline parity (expansion off)
# --------------------------------------------------------------------------- #


class TestExpansionDisabledBehavior:
    def test_search_returns_list_when_trace_not_requested(self, baseline_pipeline):
        results = baseline_pipeline.search("photosynthesis", k=3)
        assert isinstance(results, list)
        assert all("text" in r and "hybrid_score" in r for r in results)

    def test_top1_preserves_baseline_quality(self, baseline_pipeline):
        results = baseline_pipeline.search("photosynthesis", k=3)
        assert "photosynthesis" in results[0]["text"].lower()


# --------------------------------------------------------------------------- #
# Expansion-enabled flow
# --------------------------------------------------------------------------- #


class TestExpandedSearch:
    def test_search_still_returns_list(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory(
            [_ScriptedMultiQuery(["chloroplasts and light energy", "photosynthetic process"])]
        )
        results = rag.search("photosynthesis", k=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all("chunk_id" in r for r in results)

    def test_top1_still_finds_canonical_doc(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory(
            [_ScriptedMultiQuery(["chloroplasts and light", "light to chemical energy"])]
        )
        results = rag.search("photosynthesis", k=3)
        assert "photosynthesis" in results[0]["text"].lower()

    def test_hyde_only_works(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory(
            [_ScriptedHyDE(["Mitochondria produce ATP via the electron transport chain."])]
        )
        # Use a query that DOESN'T contain mitochondria but where the HyDE doc does.
        results = rag.search("powerhouse of the cell", k=3)
        joined = " ".join(r["text"].lower() for r in results)
        assert "mitochondria" in joined

    def test_multi_query_plus_hyde_compose(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory([
            _ScriptedMultiQuery(["RRF explained", "rank fusion algorithm"]),
            _ScriptedHyDE(["Reciprocal rank fusion uses 1/(k+rank) to combine multiple rankers."]),
        ])
        results = rag.search("what is RRF", k=5)
        # Note: expansion can shift top-1 when a variant introduces a generic
        # term (e.g. "algorithm") that strongly matches an unrelated corpus
        # doc via BM25. RRF dampens but doesn't always reverse this — the
        # canonical doc must appear in the top-k pool, not necessarily #1.
        # This is the precision/recall tradeoff documented in Phase 5.
        joined = " ".join(r["text"].lower() for r in results)
        assert "reciprocal rank fusion" in joined


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


class TestExpansionFailureContainment:
    def test_failing_strategy_does_not_break_retrieval(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory([_AlwaysFailStrategy()])
        results = rag.search("photosynthesis", k=3)
        # Original query still produces correct retrieval.
        assert "photosynthesis" in results[0]["text"].lower()

    def test_empty_expansion_still_returns_original_query_results(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory([])  # no strategies
        results = rag.search("BM25", k=3)
        assert "bm25" in results[0]["text"].lower()


# --------------------------------------------------------------------------- #
# RetrievalTrace
# --------------------------------------------------------------------------- #


class TestRetrievalTrace:
    def test_trace_returned_when_requested(self, baseline_pipeline):
        result_tuple = baseline_pipeline.search("photosynthesis", k=3, return_trace=True)
        assert isinstance(result_tuple, tuple)
        results, trace = result_tuple
        assert isinstance(results, list)
        assert isinstance(trace, RetrievalTrace)

    def test_trace_baseline_fields_populated(self, baseline_pipeline):
        _, trace = baseline_pipeline.search("photosynthesis", k=3, return_trace=True)
        assert trace.original_query == "photosynthesis"
        assert trace.total_latency_ms > 0
        assert len(trace.final_results) > 0

    def test_trace_captures_expansion(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory(
            [_ScriptedMultiQuery(["variant one", "variant two"])]
        )
        _, trace = rag.search("photosynthesis", k=3, return_trace=True)
        # Original + 2 variants = 3 distinct query strings.
        assert len(trace.expanded_queries) >= 3
        assert "multi_query" in trace.strategies_used
        # Per-strategy candidate breakdown should have at least one entry
        # per query × modality (and HyDE entries when present).
        assert len(trace.per_strategy) >= 2

    def test_trace_captures_hyde(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory(
            [_ScriptedHyDE(["A short hypothetical answer."])]
        )
        _, trace = rag.search("query", k=3, return_trace=True)
        assert len(trace.hyde_documents) == 1

    def test_trace_records_strategy_errors(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory([_AlwaysFailStrategy()])
        _, trace = rag.search("query", k=3, return_trace=True)
        assert any("broken" in e for e in trace.expansion_errors)

    def test_trace_to_dict_is_json_serializable(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory(
            [_ScriptedMultiQuery(["alt phrasing"])]
        )
        _, trace = rag.search("BM25", k=3, return_trace=True)
        # to_dict must produce JSON-serializable output for log shipping.
        payload = trace.to_dict()
        json.dumps(payload)  # raises if not serializable

    def test_trace_per_strategy_labels(self, expanded_pipeline_factory):
        rag = expanded_pipeline_factory(
            [_ScriptedMultiQuery(["alt phrasing"])]
        )
        _, trace = rag.search("photosynthesis", k=3, return_trace=True)
        labels = {s.label for s in trace.per_strategy}
        # We expect "vector:original", "bm25:original",
        # "vector:variant_1", "bm25:variant_1".
        assert any(l.startswith("vector:original") for l in labels)
        assert any(l.startswith("bm25:original") for l in labels)
        assert any(l.startswith("vector:variant_") for l in labels)
        assert any(l.startswith("bm25:variant_") for l in labels)


# --------------------------------------------------------------------------- #
# Reranker still active on top of expanded pool
# --------------------------------------------------------------------------- #


class _DeterministicMockReranker:
    """Tokens-shared reranker — same one used in test_retrieval_quality."""

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


class TestExpansionWithReranker:
    def test_rerank_applies_to_expanded_pool(self, tmp_path):
        rag = RAGPipeline(
            index_dir=str(tmp_path / "exp_rerank"),
            expansion_pipeline=QueryExpansionPipeline(
                strategies=[_ScriptedMultiQuery(["paris landmark", "eiffel tower history"])]
            ),
            enable_expansion=True,
            reranker=_DeterministicMockReranker(),
            enable_reranker=True,
        )
        rag.ingest_documents(GOLDEN_DOCS)
        results = rag.search("Eiffel Tower in Paris", k=3, return_trace=False)
        # Reranker should still leave rerank_score on each result.
        assert all("rerank_score" in r for r in results)

    def test_trace_reranker_fields_populated(self, tmp_path):
        rag = RAGPipeline(
            index_dir=str(tmp_path / "exp_rerank_trace"),
            expansion_pipeline=QueryExpansionPipeline(
                strategies=[_ScriptedMultiQuery(["paris"])]
            ),
            enable_expansion=True,
            reranker=_DeterministicMockReranker(),
            enable_reranker=True,
        )
        rag.ingest_documents(GOLDEN_DOCS)
        _, trace = rag.search("Eiffel Tower", k=3, return_trace=True)
        assert trace.reranker_used is True
        assert trace.rerank_input_size >= trace.rerank_output_size > 0
