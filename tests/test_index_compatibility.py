# tests/test_index_compatibility.py

"""Tests for the safe index compatibility system (Phase 12g)."""

from __future__ import annotations

from src.indexing import (
    Action,
    IndexProfile,
    IndexTargetConfig,
    Severity,
    check_compatibility,
    target_from_index_settings,
)


def _profile(**over):
    base = dict(
        name="idx",
        backend="faiss",
        index_type="hnsw",
        embedding_model="all-MiniLM-L6-v2",
        vector_dimension=384,
        chunk_size=500,
        chunk_overlap=50,
        corpus_fingerprint="fp-abc",
    )
    base.update(over)
    return IndexProfile(**base)


def _target(**over):
    base = dict(
        embedding_model="all-MiniLM-L6-v2",
        backend="faiss",
        index_type="hnsw",
        chunk_size=500,
        chunk_overlap=50,
        vector_dimension=384,
        corpus_fingerprint="fp-abc",
    )
    base.update(over)
    return IndexTargetConfig(**base)


# --------------------------------------------------------------------------- #
# Compatible
# --------------------------------------------------------------------------- #


def test_identical_config_is_reusable():
    rep = check_compatibility(_profile(), _target())
    assert rep.compatible is True
    assert rep.action == Action.REUSE
    assert rep.issues == []
    assert "compatible" in rep.message.lower()


# --------------------------------------------------------------------------- #
# BLOCKING → create_new
# --------------------------------------------------------------------------- #


def test_embedding_model_change_is_blocking():
    rep = check_compatibility(_profile(), _target(embedding_model="BAAI/bge-m3"))
    assert rep.compatible is False
    assert rep.action == Action.CREATE_NEW
    assert any(i.field == "embedding_model" and i.severity == Severity.BLOCKING
               for i in rep.issues)
    assert "Create a new index?" in rep.message


def test_dimension_change_is_blocking():
    rep = check_compatibility(_profile(), _target(embedding_model="other", vector_dimension=768))
    assert rep.action == Action.CREATE_NEW
    assert any(i.field == "vector_dimension" for i in rep.issues)


def test_dimension_unknown_skips_dim_check():
    # target dim 0 → rely on model-name check only; same model → reusable
    rep = check_compatibility(_profile(), _target(vector_dimension=0))
    assert rep.compatible is True


def test_backend_change_is_blocking():
    rep = check_compatibility(_profile(), _target(backend="chromadb"))
    assert rep.action == Action.CREATE_NEW
    assert any(i.field == "backend" and i.severity == Severity.BLOCKING for i in rep.issues)


# --------------------------------------------------------------------------- #
# REBUILD → rebuild
# --------------------------------------------------------------------------- #


def test_chunking_change_requires_rebuild():
    rep = check_compatibility(_profile(), _target(chunk_size=1000))
    assert rep.compatible is False
    assert rep.action == Action.REBUILD
    assert any(i.field == "chunking" and i.severity == Severity.REBUILD for i in rep.issues)


def test_topology_change_requires_rebuild():
    rep = check_compatibility(_profile(), _target(index_type="ivf_pq"))
    assert rep.action == Action.REBUILD
    assert any(i.field == "index_type" for i in rep.issues)


def test_stale_corpus_requires_rebuild():
    rep = check_compatibility(_profile(), _target(corpus_fingerprint="fp-NEW"))
    assert rep.action == Action.REBUILD
    assert any(i.field == "corpus_fingerprint" for i in rep.issues)


def test_corpus_check_skipped_when_target_fp_none():
    rep = check_compatibility(_profile(), _target(corpus_fingerprint=None))
    assert rep.compatible is True


# --------------------------------------------------------------------------- #
# Severity precedence: blocking beats rebuild
# --------------------------------------------------------------------------- #


def test_blocking_takes_precedence_over_rebuild():
    rep = check_compatibility(
        _profile(),
        _target(embedding_model="other", chunk_size=999, index_type="flat"),
    )
    # Both blocking and rebuild issues present, but action is create_new.
    assert rep.action == Action.CREATE_NEW
    assert len(rep.blocking_issues) >= 1
    assert len(rep.rebuild_issues) >= 1


def test_report_serializes():
    rep = check_compatibility(_profile(), _target(embedding_model="other"))
    d = rep.to_dict()
    assert d["compatible"] is False
    assert d["action"] == "create_new"
    assert isinstance(d["issues"], list)
    assert d["issues"][0]["severity"] == "blocking"


# --------------------------------------------------------------------------- #
# Bridge from runtime IndexConstructionSettings
# --------------------------------------------------------------------------- #


def test_target_from_index_settings():
    from src.providers import EmbeddingModelConfig
    from src.runtime_config import IndexConstructionSettings

    settings = IndexConstructionSettings(
        embedding=EmbeddingModelConfig(model="all-MiniLM-L6-v2"),
        chunk_size=500, chunk_overlap=50,
        vector_backend="faiss", faiss_index_type="hnsw",
    )
    target = target_from_index_settings(settings, vector_dimension=384,
                                        corpus_fingerprint="fp-abc")
    assert target.embedding_model == "all-MiniLM-L6-v2"
    assert target.backend == "faiss"
    assert target.index_type == "hnsw"
    assert target.vector_dimension == 384

    # And it round-trips into a reuse verdict against a matching profile.
    rep = check_compatibility(_profile(), target)
    assert rep.compatible is True
