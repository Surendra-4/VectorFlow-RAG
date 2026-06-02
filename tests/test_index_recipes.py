# tests/test_index_recipes.py

"""
Tests for FAISS recipe validation/estimation (Phase 12f) and recipe-driven
index construction through IndexManager. Synthetic vectors only — no models.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.indexing import (
    IndexManager,
    IndexProfile,
    IndexRegistry,
    RECIPES,
    build_factory_string,
    list_recipes,
    resolve_params,
    validate_recipe,
)
from src.indexing.recipes import RecipeError, estimate


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def test_catalog_has_required_recipes():
    expected = {
        "flat", "hnsw", "ivf", "pq",                       # basic
        "ivf_pq", "ivf_hnsw", "hnsw_pq", "opq_ivf_pq",     # advanced
        "imi", "index_refine_flat", "multi_d_adc",
    }
    assert expected <= set(RECIPES.keys())


def test_basic_advanced_split():
    basic = {r.id for r in list_recipes("basic")}
    advanced = {r.id for r in list_recipes("advanced")}
    assert basic == {"flat", "hnsw", "ivf", "pq"}
    assert "opq_ivf_pq" in advanced
    assert basic.isdisjoint(advanced)


def test_unknown_recipe_raises():
    with pytest.raises(RecipeError):
        resolve_params("nope", {})


# --------------------------------------------------------------------------- #
# Every recipe constructs in real FAISS at a realistic dim
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("recipe_id", list(RECIPES.keys()))
def test_every_recipe_constructs(recipe_id):
    v = validate_recipe(recipe_id, None, dim=384, n_vectors=2000)
    assert v.ok, f"{recipe_id} failed: {v.errors}"
    assert v.factory_string
    assert v.estimate is not None
    assert v.estimate.memory_bytes > 0


# --------------------------------------------------------------------------- #
# Validation: illegal combinations + dimension divisibility + training reqs
# --------------------------------------------------------------------------- #


def test_pq_m_must_divide_dimension():
    # 100 is not divisible by 8.
    v = validate_recipe("pq", {"pq_m": 8}, dim=100)
    assert not v.ok
    assert any(e["field"] == "pq_m" for e in v.errors)


def test_pq_m_divides_ok():
    v = validate_recipe("pq", {"pq_m": 8}, dim=128)
    assert v.ok, v.errors


def test_opq_m_must_divide_dimension():
    v = validate_recipe("opq_ivf_pq", {"opq_m": 7}, dim=384)
    assert not v.ok
    assert any(e["field"] == "opq_m" for e in v.errors)


def test_nprobe_cannot_exceed_nlist():
    v = validate_recipe("ivf", {"nlist": 10, "nprobe": 50}, dim=384, construct=False)
    assert not v.ok
    assert any(e["field"] == "nprobe" for e in v.errors)


def test_param_range_enforced():
    v = validate_recipe("hnsw", {"M": 9999}, dim=384, construct=False)
    assert not v.ok
    assert any(e["field"] == "M" for e in v.errors)


def test_zero_dim_rejected():
    v = validate_recipe("flat", None, dim=0)
    assert not v.ok
    assert any(e["field"] == "dim" for e in v.errors)


def test_low_training_points_warns_not_errors():
    v = validate_recipe("ivf", {"nlist": 100}, dim=384, n_vectors=50)
    assert v.ok  # still valid…
    assert any("39" in w or "training" in w.lower() for w in v.warnings)  # …but warned


def test_min_training_points_reported():
    v = validate_recipe("ivf_pq", {"nlist": 64, "pq_nbits": 8}, dim=384, n_vectors=5000)
    assert v.estimate.min_training_points == 256  # 2^8 dominates nlist=64


# --------------------------------------------------------------------------- #
# Factory string shapes
# --------------------------------------------------------------------------- #


def test_factory_strings():
    d = 384
    assert build_factory_string("flat", resolve_params("flat", {}), d) == "Flat"
    assert build_factory_string("hnsw", resolve_params("hnsw", {"M": 32}), d) == "HNSW32"
    assert build_factory_string("ivf_pq", resolve_params("ivf_pq", {"nlist": 256, "pq_m": 8}), d) \
        == "IVF256,PQ8x8"
    assert build_factory_string("opq_ivf_pq",
                                resolve_params("opq_ivf_pq", {"opq_m": 16, "nlist": 256, "pq_m": 8}),
                                d) == "OPQ16,IVF256,PQ8x8"


# --------------------------------------------------------------------------- #
# Estimation monotonicity (sanity)
# --------------------------------------------------------------------------- #


def test_pq_uses_less_memory_than_flat():
    flat = estimate("flat", resolve_params("flat", {}), dim=384, n_vectors=100_000)
    pq = estimate("ivf_pq", resolve_params("ivf_pq", {"pq_m": 8}), dim=384, n_vectors=100_000)
    assert pq.memory_bytes < flat.memory_bytes
    assert pq.bytes_per_vector < flat.bytes_per_vector


# --------------------------------------------------------------------------- #
# End-to-end: build + train + search through the manager for several recipes
# --------------------------------------------------------------------------- #


def _data(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    texts = [f"doc {i}" for i in range(n)]
    ids = [f"c{i}" for i in range(n)]
    return texts, v, ids


@pytest.fixture
def manager(temp_dir):
    reg = IndexRegistry(path=temp_dir / "reg.json")
    return IndexManager(registry=reg, indices_root=temp_dir / "named")


@pytest.mark.parametrize("recipe_id,params", [
    ("flat", {}),
    ("hnsw", {"M": 16}),
    ("ivf", {"nlist": 8, "nprobe": 4}),
    ("ivf_pq", {"nlist": 8, "nprobe": 4, "pq_m": 8, "pq_nbits": 4}),
    ("hnsw_pq", {"M": 16, "pq_m": 8, "pq_nbits": 4}),
])
def test_recipe_build_train_search_roundtrip(manager, recipe_id, params):
    dim = 32  # divisible by pq_m=8
    n = 600   # enough to train PQ (2^4=16 centroids) and IVF
    texts, v, ids = _data(n, dim, seed=7)

    profile = IndexProfile(
        name=f"r_{recipe_id}",
        backend="faiss",
        index_type=recipe_id,
        embedding_model="synthetic",
        vector_dimension=dim,
        build_params=params,
        search_params=params,
    )
    manager.create_index(profile, texts, v, None, ids, make_active=True)

    store = manager.load_index(profile.name)
    res = store.search(v[0], n_results=5)
    assert len(res["documents"]) == 5
    # self should be among the top results for any reasonable recipe
    assert "c0" in res["ids"]


def test_recipe_index_persists_and_reloads(manager):
    dim = 32
    texts, v, ids = _data(400, dim, seed=3)
    profile = IndexProfile(
        name="persist_ivfpq", backend="faiss", index_type="ivf_pq",
        embedding_model="synthetic", vector_dimension=dim,
        build_params={"nlist": 8, "pq_m": 8, "pq_nbits": 4},
        search_params={"nprobe": 4},
    )
    manager.create_index(profile, texts, v, None, ids)

    # Fresh manager instance → must reload the factory index from disk.
    reg2 = IndexRegistry(path=manager.registry.path)
    mgr2 = IndexManager(registry=reg2, indices_root=manager.indices_root)
    store = mgr2.load_index("persist_ivfpq")
    assert len(store) == 400
    res = store.search(v[0], n_results=3)
    assert len(res["ids"]) == 3
