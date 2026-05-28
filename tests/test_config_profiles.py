# tests/test_config_profiles.py

"""
Tests for the multilingual profile system (Phase 11).

The hard invariant: the ``english`` profile (default) leaves every model
choice at the validated baseline. Profiles only override fields still at
their English default; explicit overrides always win.
"""

from __future__ import annotations

import pytest

from src.config import (
    ENGLISH_EMBEDDER_MODEL,
    ENGLISH_RERANKER_MODEL,
    BM25Settings,
    EmbedderSettings,
    RerankerSettings,
    Settings,
    reset_settings_cache,
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    import os

    for key in list(os.environ.keys()):
        if key.startswith("VFR_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


class TestEnglishProfileBaseline:
    def test_default_profile_is_english(self):
        assert Settings().profile == "english"

    def test_english_profile_keeps_baseline_models(self):
        s = Settings()
        assert s.embedder.model_name == ENGLISH_EMBEDDER_MODEL
        assert s.reranker.model_name == ENGLISH_RERANKER_MODEL
        assert s.bm25.use_stemmer is True
        assert s.bm25.language == "english"

    def test_english_embedder_prefixes_default_none(self):
        # None → auto-detect in the adapter (which yields no prefix for MiniLM).
        s = Settings()
        assert s.embedder.query_prefix is None
        assert s.embedder.passage_prefix is None


class TestMultilingualProfile:
    def test_swaps_embedder_and_reranker(self):
        s = Settings(profile="multilingual")
        assert s.embedder.model_name == "intfloat/multilingual-e5-small"
        assert s.reranker.model_name == "jinaai/jina-reranker-v2-base-multilingual"

    def test_disables_stemmer(self):
        s = Settings(profile="multilingual")
        assert s.bm25.use_stemmer is False

    def test_quality_profile_uses_bge(self):
        s = Settings(profile="multilingual_quality")
        assert s.embedder.model_name == "BAAI/bge-m3"
        assert s.reranker.model_name == "BAAI/bge-reranker-v2-m3"
        assert s.bm25.use_stemmer is False


class TestExplicitOverridesWin:
    def test_explicit_embedder_model_survives_profile(self):
        s = Settings(
            profile="multilingual",
            embedder=EmbedderSettings(model_name="my/custom-embedder"),
        )
        assert s.embedder.model_name == "my/custom-embedder"

    def test_explicit_reranker_model_survives_profile(self):
        s = Settings(
            profile="multilingual",
            reranker=RerankerSettings(model_name="my/custom-reranker"),
        )
        assert s.reranker.model_name == "my/custom-reranker"

    def test_explicit_language_keeps_stemmer(self):
        # A user who set a specific BM25 language wants stemming for a known
        # monolingual corpus — the profile must not flip it off.
        s = Settings(
            profile="multilingual",
            bm25=BM25Settings(language="german"),
        )
        assert s.bm25.use_stemmer is True
        assert s.bm25.language == "german"


class TestProfileViaEnv:
    def test_profile_from_env(self, monkeypatch):
        monkeypatch.setenv("VFR_PROFILE", "multilingual")
        reset_settings_cache()
        s = Settings()
        assert s.profile == "multilingual"
        assert s.embedder.model_name == "intfloat/multilingual-e5-small"

    def test_env_embedder_override_beats_profile(self, monkeypatch):
        monkeypatch.setenv("VFR_PROFILE", "multilingual")
        monkeypatch.setenv("VFR_EMBEDDER__MODEL_NAME", "intfloat/multilingual-e5-base")
        reset_settings_cache()
        s = Settings()
        assert s.embedder.model_name == "intfloat/multilingual-e5-base"

    def test_invalid_profile_rejected(self, monkeypatch):
        monkeypatch.setenv("VFR_PROFILE", "klingon")
        reset_settings_cache()
        with pytest.raises(Exception):
            Settings()


class TestNewFieldDefaults:
    def test_ocr_lang_default(self):
        assert Settings().ingestion.ocr_lang == "eng"

    def test_detect_language_off_by_default(self):
        assert Settings().ingestion.detect_language is False

    def test_bm25_use_stemmer_default_true(self):
        assert Settings().bm25.use_stemmer is True
