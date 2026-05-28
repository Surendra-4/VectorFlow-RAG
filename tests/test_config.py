# tests/test_config.py

"""Tests for the centralized Settings system."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    AppSettings,
    BM25Settings,
    CacheSettings,
    ChunkerSettings,
    EmbedderSettings,
    LLMSettings,
    LoggingSettings,
    RerankerSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
    get_settings,
    reset_settings_cache,
)


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    """Wipe VFR_* env vars and the cached singleton before each test."""
    for key in list(os_environ_keys()):
        if key.startswith("VFR_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


def os_environ_keys():
    import os

    return list(os.environ.keys())


class TestDefaults:
    def test_all_sections_present(self):
        s = Settings()
        assert isinstance(s.app, AppSettings)
        assert isinstance(s.embedder, EmbedderSettings)
        assert isinstance(s.chunker, ChunkerSettings)
        assert isinstance(s.vector_store, VectorStoreSettings)
        assert isinstance(s.bm25, BM25Settings)
        assert isinstance(s.llm, LLMSettings)
        assert isinstance(s.retrieval, RetrievalSettings)
        assert isinstance(s.cache, CacheSettings)
        assert isinstance(s.reranker, RerankerSettings)
        assert isinstance(s.logging, LoggingSettings)

    def test_embedder_defaults(self):
        e = Settings().embedder
        assert e.model_name == "all-MiniLM-L6-v2"
        assert e.device is None  # auto-detect sentinel
        assert e.batch_size == 32
        assert e.normalize is True

    def test_chunker_defaults(self):
        c = Settings().chunker
        assert c.chunk_size == 500
        assert c.overlap == 50

    def test_vector_store_defaults(self):
        vs = Settings().vector_store
        assert vs.backend == "chromadb"
        assert vs.collection_name == "vectorflow_docs"
        assert isinstance(vs.persist_directory, Path)
        # Path resolves to project_root/indices/chroma_db
        assert vs.persist_directory.name == "chroma_db"
        assert vs.persist_directory.parent.name == "indices"

    def test_llm_defaults(self):
        llm = Settings().llm
        assert llm.model == "tinyllama"
        assert llm.base_url == "http://localhost:11434"
        assert llm.max_tokens == 512

    def test_retrieval_defaults(self):
        r = Settings().retrieval
        assert r.alpha == 0.5
        assert r.k_default == 5
        assert r.rrf_k == 60

    def test_cache_disabled_by_default(self):
        assert Settings().cache.backend == "none"

    def test_reranker_disabled_by_default(self):
        assert Settings().reranker.enabled is False

    def test_logging_defaults(self):
        log = Settings().logging
        assert log.level == "INFO"
        assert log.format == "text"
        assert log.file is None


class TestEnvironmentOverrides:
    def test_simple_override(self, monkeypatch):
        monkeypatch.setenv("VFR_EMBEDDER__MODEL_NAME", "BAAI/bge-large-en")
        s = Settings()
        assert s.embedder.model_name == "BAAI/bge-large-en"

    def test_nested_override_for_path(self, monkeypatch, tmp_path):
        target = tmp_path / "custom_chroma"
        monkeypatch.setenv("VFR_VECTOR_STORE__PERSIST_DIRECTORY", str(target))
        s = Settings()
        assert s.vector_store.persist_directory == target

    def test_logging_level_uppercased(self, monkeypatch):
        monkeypatch.setenv("VFR_LOGGING__LEVEL", "debug")
        s = Settings()
        assert s.logging.level == "DEBUG"

    def test_bool_override(self, monkeypatch):
        monkeypatch.setenv("VFR_RERANKER__ENABLED", "true")
        s = Settings()
        assert s.reranker.enabled is True

    def test_int_override(self, monkeypatch):
        monkeypatch.setenv("VFR_RETRIEVAL__RRF_K", "120")
        s = Settings()
        assert s.retrieval.rrf_k == 120


class TestPathNormalization:
    def test_windows_separator_in_persist_directory(self):
        s = Settings(vector_store=VectorStoreSettings(persist_directory="indices\\foo\\bar"))
        # Backslashes flattened; Path is portable
        assert s.vector_store.persist_directory == Path("indices/foo/bar")

    def test_windows_separator_in_log_file(self):
        s = Settings(logging=LoggingSettings(file="logs\\app.log"))
        assert s.logging.file == Path("logs/app.log")

    def test_empty_log_file_becomes_none(self):
        s = Settings(logging=LoggingSettings(file=""))
        assert s.logging.file is None


class TestProgrammaticOverride:
    def test_explicit_kwargs_win(self):
        s = Settings(
            embedder=EmbedderSettings(model_name="mock-model", device="cpu"),
            retrieval=RetrievalSettings(alpha=0.7),
        )
        assert s.embedder.model_name == "mock-model"
        assert s.embedder.device == "cpu"
        assert s.retrieval.alpha == 0.7


class TestCachedSingleton:
    def test_get_settings_returns_same_instance(self):
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_reset_clears_cache(self, monkeypatch):
        first = get_settings()
        monkeypatch.setenv("VFR_LLM__MODEL", "llama3.2:1b")
        # Without resetting, the cached instance still wins
        assert get_settings() is first
        reset_settings_cache()
        second = get_settings()
        assert second is not first
        assert second.llm.model == "llama3.2:1b"
