# tests/test_indexing.py

"""
Tests for the named-index subsystem (Phase 12e): IndexProfile, IndexRegistry,
IndexManager. Uses FAISS with synthetic embeddings — no model load, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.indexing import (
    IndexManager,
    IndexProfile,
    IndexRegistry,
    IndexRegistryError,
    validate_index_name,
)
from src.indexing.profile import CompatibilitySignature


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _vectors(n=12, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    # normalize for inner-product/cosine parity
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def _faiss_profile(name="idx1", dim=8, index_type="flat"):
    return IndexProfile(
        name=name,
        backend="faiss",
        index_type=index_type,
        embedding_model="all-MiniLM-L6-v2",
        vector_dimension=dim,
        build_params={"hnsw_m": 16, "hnsw_ef_construction": 100},
        search_params={"hnsw_ef_search": 32},
        chunk_size=500,
        chunk_overlap=50,
    )


@pytest.fixture
def registry(temp_dir):
    return IndexRegistry(path=temp_dir / "index_registry.json")


@pytest.fixture
def manager(temp_dir, registry):
    return IndexManager(registry=registry, indices_root=temp_dir / "named")


# --------------------------------------------------------------------------- #
# IndexProfile
# --------------------------------------------------------------------------- #


def test_profile_roundtrip_and_compat_info():
    p = _faiss_profile()
    d = p.to_dict()
    assert d["compatibility_info"]["embedding_model"] == "all-MiniLM-L6-v2"
    assert d["compatibility_info"]["vector_dimension"] == 8
    # round-trips (compatibility_info is derived, ignored on load)
    p2 = IndexProfile.from_dict(d)
    assert p2.name == p.name
    assert p2.build_params == p.build_params


def test_compatibility_signature_matches():
    a = CompatibilitySignature("m", 384, 500, 50, "faiss")
    b = CompatibilitySignature("m", 384, 500, 50, "faiss")
    c = CompatibilitySignature("m", 768, 500, 50, "faiss")
    assert a.matches(b)
    assert not a.matches(c)


def test_profile_touch_updates_last_used():
    p = _faiss_profile()
    before = p.last_used
    import time
    time.sleep(0.01)
    p.touch()
    assert p.last_used > before


# --------------------------------------------------------------------------- #
# IndexRegistry
# --------------------------------------------------------------------------- #


def test_name_validation():
    assert validate_index_name("my-index_1.v2") == "my-index_1.v2"
    for bad in ["", "../escape", "a/b", ".", "..", "x" * 65, "-leading"]:
        with pytest.raises(IndexRegistryError):
            validate_index_name(bad)


def test_register_get_list_active(registry):
    p = _faiss_profile("first")
    registry.register(p)
    # First registered index becomes active automatically.
    assert registry.active_name == "first"
    assert registry.exists("first")
    assert registry.get("first").name == "first"
    assert [pr.name for pr in registry.list_profiles()] == ["first"]


def test_register_duplicate_rejected(registry):
    registry.register(_faiss_profile("dup"))
    with pytest.raises(IndexRegistryError):
        registry.register(_faiss_profile("dup"))
    # overwrite=True allowed
    registry.register(_faiss_profile("dup"), overwrite=True)


def test_set_active_and_remove(registry):
    registry.register(_faiss_profile("a"))
    registry.register(_faiss_profile("b"))
    assert registry.active_name == "a"  # first wins
    registry.set_active("b")
    assert registry.active_name == "b"
    # removing the active one re-points to a remaining index
    registry.remove("b")
    assert registry.active_name == "a"
    registry.remove("a")
    assert registry.active_name is None


def test_registry_persists_across_instances(temp_dir):
    path = temp_dir / "reg.json"
    r1 = IndexRegistry(path=path)
    r1.register(_faiss_profile("persisted"))
    r2 = IndexRegistry(path=path)
    assert r2.exists("persisted")
    assert r2.active_name == "persisted"


def test_registry_get_missing_raises(registry):
    with pytest.raises(IndexRegistryError):
        registry.get("nope")


# --------------------------------------------------------------------------- #
# IndexManager — create / load / switch / delete
# --------------------------------------------------------------------------- #


def test_create_index_builds_and_registers(manager):
    v = _vectors(10, 8)
    texts = [f"doc {i}" for i in range(10)]
    ids = [f"c{i}" for i in range(10)]
    metas = [{"document_name": "d"} for _ in range(10)]

    profile = manager.create_index(_faiss_profile("built"), texts, v, metas, ids,
                                   make_active=True)
    assert profile.num_vectors == 10
    assert profile.vector_dimension == 8
    assert manager.registry.active_name == "built"
    assert manager.index_dir("built").exists()


def test_create_then_load_and_search(manager):
    v = _vectors(10, 8, seed=1)
    texts = [f"doc {i}" for i in range(10)]
    ids = [f"c{i}" for i in range(10)]
    manager.create_index(_faiss_profile("searchable"), texts, v, None, ids)

    store = manager.load_index("searchable")
    res = store.search(v[0], n_results=3)
    assert len(res["documents"]) == 3
    # the nearest neighbor to v[0] is itself
    assert res["ids"][0] == "c0"


def test_create_duplicate_rejected(manager):
    v = _vectors(5, 8)
    texts = [f"d{i}" for i in range(5)]
    manager.create_index(_faiss_profile("dup"), texts, v)
    with pytest.raises(IndexRegistryError):
        manager.create_index(_faiss_profile("dup"), texts, v)


def test_switch_index(manager):
    v = _vectors(6, 8)
    manager.create_index(_faiss_profile("one"), [f"a{i}" for i in range(6)], v)
    manager.create_index(_faiss_profile("two"), [f"b{i}" for i in range(6)], v)
    profile, store = manager.switch_index("two")
    assert manager.registry.active_name == "two"
    assert profile.name == "two"
    assert len(store) == 6


def test_delete_index_removes_dir(manager):
    v = _vectors(5, 8)
    manager.create_index(_faiss_profile("gone"), [f"d{i}" for i in range(5)], v)
    d = manager.index_dir("gone")
    assert d.exists()
    manager.delete_index("gone")
    assert not d.exists()
    assert not manager.registry.exists("gone")


# --------------------------------------------------------------------------- #
# IndexManager — export / import
# --------------------------------------------------------------------------- #


def test_export_then_import_roundtrip(manager, temp_dir):
    v = _vectors(8, 8, seed=2)
    texts = [f"doc {i}" for i in range(8)]
    ids = [f"c{i}" for i in range(8)]
    manager.create_index(_faiss_profile("exp"), texts, v, None, ids)

    archive = manager.export_index("exp", temp_dir / "out")
    assert archive.exists() and archive.suffix == ".zip"

    # import under a new name
    imported = manager.import_index(archive, name="exp_copy")
    assert imported.name == "exp_copy"
    assert manager.registry.exists("exp_copy")

    # data survived: nearest neighbor still resolves
    store = manager.load_index("exp_copy")
    res = store.search(v[0], n_results=1)
    assert res["ids"][0] == "c0"


def test_import_duplicate_rejected(manager, temp_dir):
    v = _vectors(5, 8)
    manager.create_index(_faiss_profile("orig"), [f"d{i}" for i in range(5)], v)
    archive = manager.export_index("orig", temp_dir / "out")
    with pytest.raises(IndexRegistryError):
        manager.import_index(archive)  # same name "orig" already exists
    # overwrite allowed
    manager.import_index(archive, overwrite=True)


def test_import_rejects_archive_without_profile(manager, temp_dir):
    import zipfile

    bad = temp_dir / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("data/whatever.txt", "x")
    with pytest.raises(IndexRegistryError):
        manager.import_index(bad, name="bad")


def test_import_rejects_zip_slip(manager, temp_dir):
    import zipfile
    import json

    erk = temp_dir / "evil.zip"
    with zipfile.ZipFile(erk, "w") as zf:
        zf.writestr("_index_profile.json", json.dumps(_faiss_profile("evil").to_dict()))
        zf.writestr("../escape.txt", "pwned")
    with pytest.raises(IndexRegistryError):
        manager.import_index(erk, name="evil")


def test_train_index_noop_for_flat(manager):
    v = _vectors(5, 8)
    manager.create_index(_faiss_profile("flatidx", index_type="flat"),
                         [f"d{i}" for i in range(5)], v)
    # No exception; returns the profile unchanged for non-trainable types.
    p = manager.train_index("flatidx")
    assert p.name == "flatidx"


def test_benchmark_not_implemented_yet(manager):
    v = _vectors(5, 8)
    manager.create_index(_faiss_profile("b"), [f"d{i}" for i in range(5)], v)
    with pytest.raises(NotImplementedError):
        manager.benchmark_index("b")
