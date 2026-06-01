# tests/test_runtime_config.py

"""Tests for the RuntimeConfigStore (Phase 12c)."""

from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.runtime_config import (
    IndexConstructionSettings,
    LiveQuerySettings,
    RuntimeConfig,
    RuntimeConfigError,
    RuntimeConfigStore,
    _deep_merge,
)


@pytest.fixture
def base_settings():
    return Settings()


@pytest.fixture
def store(base_settings, temp_dir):
    return RuntimeConfigStore(settings=base_settings, path=temp_dir / "runtime_config.json")


# --------------------------------------------------------------------------- #
# Hydration from Settings
# --------------------------------------------------------------------------- #


def test_live_settings_hydrate_from_settings(base_settings):
    live = LiveQuerySettings.from_settings(base_settings)
    assert live.chat.provider == "ollama"
    assert live.chat.model == base_settings.llm.model
    assert live.reranker_enabled is False  # English baseline
    assert live.expansion_enabled is False
    assert live.retrieval_rrf_k == base_settings.retrieval.rrf_k


def test_index_settings_hydrate_from_settings(base_settings):
    idx = IndexConstructionSettings.from_settings(base_settings)
    assert idx.embedding.model == base_settings.embedder.model_name
    assert idx.chunk_size == base_settings.chunker.chunk_size
    assert idx.vector_backend == base_settings.vector_store.backend
    assert idx.faiss_index_type == base_settings.vector_store.faiss_index_type


def test_initial_store_state_matches_settings(store, base_settings):
    assert store.live.chat.model == base_settings.llm.model
    assert store.staged_index.embedding.model == base_settings.embedder.model_name
    # active mirrors staged at boot.
    assert store.active_index.embedding.model == base_settings.embedder.model_name


# --------------------------------------------------------------------------- #
# Roundtrips
# --------------------------------------------------------------------------- #


def test_live_roundtrip(base_settings):
    live = LiveQuerySettings.from_settings(base_settings)
    live.reranker_enabled = True
    live.expansion_strategies = ["multi_query", "hyde"]
    d = live.to_dict()
    rebuilt = LiveQuerySettings.from_dict(d)
    assert rebuilt == live


def test_index_roundtrip(base_settings):
    idx = IndexConstructionSettings.from_settings(base_settings)
    idx.faiss_hnsw_m = 64
    rebuilt = IndexConstructionSettings.from_dict(idx.to_dict())
    assert rebuilt == idx


# --------------------------------------------------------------------------- #
# Snapshot views are immutable from the outside
# --------------------------------------------------------------------------- #


def test_live_view_is_a_copy(store):
    live = store.live
    live.reranker_enabled = True  # mutate the copy
    assert store.live.reranker_enabled is False  # store is unaffected


# --------------------------------------------------------------------------- #
# update_live
# --------------------------------------------------------------------------- #


def test_update_live_partial_patch(store):
    new = store.update_live({"reranker_enabled": True, "retrieval_rrf_k": 90})
    assert new.reranker_enabled is True
    assert new.retrieval_rrf_k == 90
    # Untouched fields unchanged.
    assert new.chat.provider == "ollama"
    # Persisted.
    assert json.loads(store.path.read_text())["live"]["retrieval_rrf_k"] == 90


def test_update_live_chat_provider_validation(store):
    with pytest.raises(RuntimeConfigError) as ei:
        store.update_live({"chat": {"provider": "no-such-provider"}})
    assert ei.value.field == "chat.provider"


def test_update_live_temperature_bounds(store):
    with pytest.raises(RuntimeConfigError) as ei:
        store.update_live({"chat": {"temperature": 99.0}})
    assert ei.value.field == "chat.temperature"


def test_update_live_bad_expansion_strategy(store):
    with pytest.raises(RuntimeConfigError) as ei:
        store.update_live({"expansion_strategies": ["multi_query", "telepathy"]})
    assert ei.value.field == "expansion_strategies"


def test_update_live_rejects_zero_k(store):
    with pytest.raises(RuntimeConfigError):
        store.update_live({"retrieval_k_default": 0})


def test_update_live_switches_to_online_provider(store):
    new = store.update_live({"chat": {"provider": "openai", "model": "gpt-4o-mini"}})
    assert new.chat.provider == "openai"
    assert new.chat.model == "gpt-4o-mini"


# --------------------------------------------------------------------------- #
# stage_index
# --------------------------------------------------------------------------- #


def test_stage_index_does_not_touch_active(store):
    store.stage_index({"chunk_size": 1000, "chunk_overlap": 100})
    assert store.staged_index.chunk_size == 1000
    assert store.active_index.chunk_size == 500  # active untouched — no silent rebuild


def test_stage_index_overlap_must_be_less_than_size(store):
    with pytest.raises(RuntimeConfigError) as ei:
        store.stage_index({"chunk_size": 200, "chunk_overlap": 200})
    assert ei.value.field == "chunk_overlap"


def test_stage_index_rejects_unknown_backend(store):
    with pytest.raises(RuntimeConfigError) as ei:
        store.stage_index({"vector_backend": "annoy"})
    assert ei.value.field == "vector_backend"


def test_stage_index_rejects_unknown_faiss_type(store):
    with pytest.raises(RuntimeConfigError) as ei:
        store.stage_index({"faiss_index_type": "magic"})
    assert ei.value.field == "faiss_index_type"


def test_stage_index_validates_hnsw_m_bounds(store):
    with pytest.raises(RuntimeConfigError):
        store.stage_index({"faiss_hnsw_m": 1000})
    with pytest.raises(RuntimeConfigError):
        store.stage_index({"faiss_hnsw_m": 2})


def test_mark_index_active_updates_active(store):
    staged = store.stage_index({"chunk_size": 800})
    store.mark_index_active(staged)
    assert store.active_index.chunk_size == 800


def test_reset_staged_to_active(store):
    store.stage_index({"chunk_size": 800})
    store.reset_staged_to_active()
    assert store.staged_index.chunk_size == 500


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_persistence_across_instances(base_settings, temp_dir):
    path = temp_dir / "rc.json"
    s1 = RuntimeConfigStore(base_settings, path=path)
    s1.update_live({"chat": {"provider": "anthropic", "model": "claude-3-5-haiku-latest"}})
    s1.stage_index({"chunk_size": 1200, "chunk_overlap": 100})

    s2 = RuntimeConfigStore(base_settings, path=path)
    assert s2.live.chat.provider == "anthropic"
    assert s2.live.chat.model == "claude-3-5-haiku-latest"
    assert s2.staged_index.chunk_size == 1200


def test_corrupt_file_falls_back_to_defaults(base_settings, temp_dir):
    path = temp_dir / "rc.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    s = RuntimeConfigStore(base_settings, path=path)
    assert s.live.chat.provider == "ollama"  # boot default


def test_version_mismatch_uses_defaults(base_settings, temp_dir):
    path = temp_dir / "rc.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 9999, "live": {}, "staged_index": {}, "active_index": {},
    }), encoding="utf-8")
    s = RuntimeConfigStore(base_settings, path=path)
    assert s.live.chat.provider == "ollama"


# --------------------------------------------------------------------------- #
# Snapshot + helper
# --------------------------------------------------------------------------- #


def test_snapshot_serializes_full_state(store):
    snap = store.snapshot()
    assert snap["version"] == 1
    assert "live" in snap and "staged_index" in snap and "active_index" in snap


def test_deep_merge():
    assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 9, "z": 3}}) == {
        "a": {"x": 1, "y": 9, "z": 3},
    }
    # Non-dict value at depth replaces, never merges into a dict shape.
    assert _deep_merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}
