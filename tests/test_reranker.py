# tests/test_reranker.py

"""
Tests for the cross-encoder reranker.

Two layers of coverage:

* Fast tests that mock the underlying model (``self._model``) — these run
  in CI without downloading the cross-encoder weights.
* Slow tests that load the real default model — marked ``slow`` and gated
  on a real model load (~80MB download once).
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pytest

from src.config import RerankerSettings, Settings, reset_settings_cache
from src.interfaces import RerankerProtocol
from src.reranker import CrossEncoderReranker


# --------------------------------------------------------------------------- #
# Mock model for fast tests
# --------------------------------------------------------------------------- #


class _MockCrossEncoder:
    """
    Stand-in that replaces the real ``CrossEncoder`` model.

    Returns a deterministic score that prefers candidates whose text shares
    more words with the query. Sufficient for verifying the rerank ordering
    logic without loading any neural net.
    """

    def __init__(self):
        self.calls = 0

    @staticmethod
    def _score_pair(q: str, c: str) -> float:
        q_tokens = set(q.lower().split())
        c_tokens = set(c.lower().split())
        return float(len(q_tokens & c_tokens))

    def predict(self, pairs, batch_size: int = 32, show_progress_bar: bool = False):
        self.calls += 1
        return np.array([self._score_pair(q, c) for q, c in pairs], dtype=np.float32)


@pytest.fixture
def mock_reranker(monkeypatch):
    """Return a CrossEncoderReranker with the model attribute pre-injected."""
    rr = CrossEncoderReranker()
    rr._model = _MockCrossEncoder()
    return rr


# --------------------------------------------------------------------------- #
# Fast tests (no real model)
# --------------------------------------------------------------------------- #


class TestRerankerInitialization:
    def test_defaults_pulled_from_settings(self):
        rr = CrossEncoderReranker()
        assert rr.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert rr.top_n == 3
        # Device resolves to a concrete string (cpu/cuda/mps), never None.
        assert rr.device in ("cpu", "cuda", "mps")

    def test_explicit_args_override_settings(self):
        rr = CrossEncoderReranker(model_name="foo/bar", device="cpu", top_n=7, batch_size=8)
        assert rr.model_name == "foo/bar"
        assert rr.device == "cpu"
        assert rr.top_n == 7
        assert rr.batch_size == 8

    def test_model_lazy_loaded(self):
        rr = CrossEncoderReranker()
        assert rr._model is None  # no load on construction

    def test_satisfies_reranker_protocol(self, mock_reranker):
        assert isinstance(mock_reranker, RerankerProtocol)


class TestRerankBehavior:
    def test_empty_candidates_returns_empty(self, mock_reranker):
        assert mock_reranker.rerank("query", []) == []

    def test_returns_top_n(self, mock_reranker):
        candidates = [{"text": "the quick brown fox"} for _ in range(10)]
        out = mock_reranker.rerank("the quick", candidates, top_n=4)
        assert len(out) == 4

    def test_rerank_score_added(self, mock_reranker):
        out = mock_reranker.rerank("hello world", [{"text": "hello there"}, {"text": "goodbye"}])
        assert all("rerank_score" in r for r in out)
        assert all(isinstance(r["rerank_score"], float) for r in out)

    def test_results_sorted_by_rerank_score_desc(self, mock_reranker):
        candidates = [
            {"text": "no overlap whatsoever"},
            {"text": "machine learning algorithms"},
            {"text": "deep learning systems"},
        ]
        out = mock_reranker.rerank("learning", candidates, top_n=3)
        scores = [r["rerank_score"] for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_input_candidates_not_mutated(self, mock_reranker):
        original = [{"text": "alpha"}, {"text": "beta"}]
        snapshot = [dict(c) for c in original]
        _ = mock_reranker.rerank("alpha", original, top_n=2)
        assert original == snapshot
        assert all("rerank_score" not in c for c in original)

    def test_stable_ordering_on_ties(self, mock_reranker):
        # Two candidates with same overlap score against query.
        candidates = [
            {"text": "alpha gamma", "id": 1},
            {"text": "alpha delta", "id": 2},
        ]
        out = mock_reranker.rerank("alpha", candidates, top_n=2)
        # Both have score 1; Python's stable sort preserves input order.
        assert out[0]["id"] == 1 and out[1]["id"] == 2

    def test_default_top_n_used_when_not_passed(self, mock_reranker):
        mock_reranker.top_n = 2
        candidates = [{"text": f"doc {i}"} for i in range(5)]
        out = mock_reranker.rerank("doc", candidates)
        assert len(out) == 2

    def test_extra_fields_preserved(self, mock_reranker):
        candidates = [{"text": "alpha", "hybrid_score": 0.5, "vector_rank": 1}]
        out = mock_reranker.rerank("alpha", candidates, top_n=1)
        assert out[0]["hybrid_score"] == 0.5
        assert out[0]["vector_rank"] == 1
        assert out[0]["rerank_score"] is not None

    def test_text_key_configurable(self, mock_reranker):
        candidates = [{"chunk": "alpha foo"}, {"chunk": "no match"}]
        out = mock_reranker.rerank("alpha", candidates, top_n=2, text_key="chunk")
        assert out[0]["chunk"] == "alpha foo"

    def test_batch_size_passed_through(self, mock_reranker):
        rr = CrossEncoderReranker(batch_size=16)
        rr._model = _MockCrossEncoder()
        rr.rerank("q", [{"text": "a"}, {"text": "b"}])
        # Mock records being called once with the batched predict call.
        assert rr._model.calls == 1


class TestRerankerConfigOverrides:
    def test_env_overrides_picked_up(self, monkeypatch):
        monkeypatch.setenv("VFR_RERANKER__MODEL_NAME", "BAAI/bge-reranker-base")
        monkeypatch.setenv("VFR_RERANKER__TOP_N", "5")
        reset_settings_cache()
        try:
            rr = CrossEncoderReranker()
            assert rr.model_name == "BAAI/bge-reranker-base"
            assert rr.top_n == 5
        finally:
            reset_settings_cache()

    def test_top_n_kwarg_at_call_overrides_init(self, mock_reranker):
        mock_reranker.top_n = 10
        out = mock_reranker.rerank("q", [{"text": f"d{i}"} for i in range(5)], top_n=2)
        assert len(out) == 2


# --------------------------------------------------------------------------- #
# Slow tests — load the real default cross-encoder model.
# Skipped automatically in CI; runnable locally with `pytest -m slow`.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
class TestRerankerWithRealModel:
    def test_real_model_returns_reasonable_ranking(self):
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.skip("Skipping real-model load in CI")

        rr = CrossEncoderReranker()
        candidates = [
            {"text": "Python is a popular programming language used widely in data science."},
            {"text": "The Eiffel Tower is a landmark in Paris, France."},
            {"text": "Machine learning models can be trained on labeled data."},
        ]
        out = rr.rerank("What is the capital of France?", candidates, top_n=3)
        # The Paris-related candidate should rank first.
        assert "Paris" in out[0]["text"] or "Eiffel" in out[0]["text"]
        assert all("rerank_score" in r for r in out)
