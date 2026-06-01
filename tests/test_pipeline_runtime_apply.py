# tests/test_pipeline_runtime_apply.py

"""Tests for RAGPipeline.set_chat_provider / apply_live_settings (Phase 12c).

These exercise the methods directly on a stub pipeline (the only attributes
they touch are listed below) so we don't pay the cost of loading the embedder.
The parity guarantee — default behavior unchanged — is asserted explicitly.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.providers import ChatModelConfig, ProviderError
from src.providers.ollama import OllamaProvider
from src.rag_pipeline import RAGPipeline
from src.runtime_config import LiveQuerySettings


def _stub_pipeline(settings):
    """Build a RAGPipeline shell with only the attributes the runtime-mutation
    methods read or write. Skips the real ``__init__`` (and its model loads)."""
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.settings = settings
    pipe._owns_settings = False
    pipe._reranker = None
    pipe._expansion_pipeline = None
    pipe.enable_reranker = settings.reranker.enabled
    pipe.enable_expansion = settings.query_expansion.enabled
    # Sentinel objects we can identity-compare against.
    pipe.llm = object()
    return pipe


# --------------------------------------------------------------------------- #
# Parity: default-derived live settings produce a no-op apply
# --------------------------------------------------------------------------- #


def test_apply_default_live_is_a_noop_for_observable_state():
    s = Settings()
    pipe = _stub_pipeline(s)
    live = LiveQuerySettings.from_settings(s)

    pipe.apply_live_settings(live)

    # All observable state matches the boot baseline.
    assert pipe.enable_reranker is False
    assert pipe.enable_expansion is False
    assert pipe.settings.retrieval.rrf_k == s.retrieval.rrf_k
    assert pipe.settings.retrieval.candidates_per_modality == s.retrieval.candidates_per_modality
    assert pipe.settings.reranker.model_name == s.reranker.model_name


# --------------------------------------------------------------------------- #
# Settings ownership: mutation must not leak to the shared singleton
# --------------------------------------------------------------------------- #


def test_apply_live_settings_clones_settings_on_first_mutation():
    s = Settings()
    pipe = _stub_pipeline(s)
    assert pipe.settings is s  # before any mutation, we share the instance

    live = LiveQuerySettings.from_settings(s)
    live.retrieval_rrf_k = 99
    pipe.apply_live_settings(live)

    assert pipe.settings is not s        # detached
    assert pipe.settings.retrieval.rrf_k == 99
    assert s.retrieval.rrf_k == 60       # singleton untouched


# --------------------------------------------------------------------------- #
# Live mutations
# --------------------------------------------------------------------------- #


def test_apply_live_toggles_reranker_and_rebuilds_on_model_change():
    s = Settings()
    pipe = _stub_pipeline(s)
    pipe._reranker = object()  # pretend we already lazy-built one

    live = LiveQuerySettings.from_settings(s)
    live.reranker_enabled = True
    live.reranker.enabled = True
    live.reranker.model = "BAAI/bge-reranker-v2-m3"  # changed → invalidate
    live.reranker.top_n = 7

    pipe.apply_live_settings(live)
    assert pipe.enable_reranker is True
    assert pipe.settings.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert pipe.settings.reranker.top_n == 7
    assert pipe._reranker is None  # invalidated for lazy rebuild


def test_apply_live_keeps_reranker_when_model_unchanged():
    s = Settings()
    pipe = _stub_pipeline(s)
    existing = object()
    pipe._reranker = existing

    live = LiveQuerySettings.from_settings(s)
    live.reranker_enabled = True
    # Note: model NOT changed.
    pipe.apply_live_settings(live)
    assert pipe._reranker is existing  # not invalidated — model didn't change


def test_apply_live_expansion_strategy_change_invalidates_pipeline():
    s = Settings()
    pipe = _stub_pipeline(s)
    pipe._expansion_pipeline = object()

    live = LiveQuerySettings.from_settings(s)
    live.expansion_enabled = True
    live.expansion_strategies = ["multi_query", "hyde"]

    pipe.apply_live_settings(live)
    assert pipe.enable_expansion is True
    assert pipe.settings.query_expansion.strategies == ["multi_query", "hyde"]
    assert pipe._expansion_pipeline is None


def test_apply_live_retrieval_knobs():
    s = Settings()
    pipe = _stub_pipeline(s)
    live = LiveQuerySettings.from_settings(s)
    live.retrieval_k_default = 8
    live.retrieval_candidates_per_modality = 25
    live.retrieval_rrf_k = 90

    pipe.apply_live_settings(live)
    assert pipe.settings.retrieval.k_default == 8
    assert pipe.settings.retrieval.candidates_per_modality == 25
    assert pipe.settings.retrieval.rrf_k == 90


# --------------------------------------------------------------------------- #
# Chat provider swap
# --------------------------------------------------------------------------- #


def test_set_chat_provider_swaps_llm_and_updates_settings():
    s = Settings()
    pipe = _stub_pipeline(s)

    cfg = ChatModelConfig(
        provider="ollama", model="llama3.2:1b",
        base_url="http://localhost:11434", request_timeout_s=60,
        max_tokens=256, temperature=0.2,
    )
    pipe.set_chat_provider(cfg)

    assert isinstance(pipe.llm, OllamaProvider)
    assert pipe.llm.model == "llama3.2:1b"
    assert pipe.settings.llm.model == "llama3.2:1b"
    assert pipe.settings.llm.max_tokens == 256
    assert pipe.settings.llm.temperature == 0.2
    # Singleton not touched.
    assert s.llm.model != "llama3.2:1b"


def test_set_chat_provider_invalidates_expansion_pipeline():
    s = Settings()
    pipe = _stub_pipeline(s)
    pipe._expansion_pipeline = object()

    pipe.set_chat_provider(ChatModelConfig(provider="ollama", model="tinyllama"))
    assert pipe._expansion_pipeline is None


def test_set_chat_provider_unknown_provider_raises():
    s = Settings()
    pipe = _stub_pipeline(s)
    with pytest.raises(ProviderError):
        pipe.set_chat_provider(ChatModelConfig(provider="no-such", model="x"))
