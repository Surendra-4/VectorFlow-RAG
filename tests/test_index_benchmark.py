# tests/test_index_benchmark.py

"""Tests for index benchmarking (Phase 12i). Synthetic vectors; no models."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.indexing import (
    IndexManager,
    IndexProfile,
    IndexRegistry,
    benchmark_recipes,
    exact_neighbors,
)
from src.indexing.benchmark import evaluate_store, persist_benchmark


def _corpus(n=200, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def test_exact_neighbors_self_is_top1():
    v = _corpus(50, 8)
    nn = exact_neighbors(v, v, k=3)
    # The nearest neighbor of each vector is itself.
    assert list(nn[:, 0]) == list(range(50))


def test_exact_neighbors_shapes():
    v = _corpus(30, 8)
    q = v[:5]
    nn = exact_neighbors(v, q, k=4)
    assert nn.shape == (5, 4)


# --------------------------------------------------------------------------- #
# evaluate_store
# --------------------------------------------------------------------------- #


def test_flat_index_has_perfect_recall(temp_dir):
    from src.faiss_store import FAISSVectorStore

    v = _corpus(100, 16)
    ids = [f"c{i}" for i in range(100)]
    store = FAISSVectorStore(persist_directory=str(temp_dir / "flat"),
                             collection_name="b", index_type="flat",
                             factory_string="Flat")
    store.add_documents(texts=[f"d{i}" for i in range(100)],
                        embeddings=v.tolist(), ids=ids)

    q = v[:20]
    gt = exact_neighbors(v, q, k=5)
    truth = [[ids[j] for j in row] for row in gt]
    scores = evaluate_store(store, q, truth, k=5)
    assert scores["recall_at_k"] == pytest.approx(1.0)
    assert scores["mrr"] == pytest.approx(1.0)
    # Flat reproduces the exact ranking, so NDCG@k is perfect.
    assert scores["ndcg_at_k"] == pytest.approx(1.0)
    assert scores["latency_ms_mean"] >= 0.0
    assert scores["queries_per_sec"] > 0.0


# --------------------------------------------------------------------------- #
# benchmark_recipes
# --------------------------------------------------------------------------- #


def test_benchmark_recipes_compares_multiple(temp_dir):
    v = _corpus(300, 16, seed=4)
    results = benchmark_recipes(
        v, ["flat", "hnsw", "ivf"], workdir=temp_dir / "wd",
        ids=[f"c{i}" for i in range(300)], k=10,
        params={"ivf": {"nlist": 8, "nprobe": 8}, "hnsw": {"M": 16}},
    )
    by = {r.recipe: r for r in results}
    assert set(by) == {"flat", "hnsw", "ivf"}
    # Exact Flat must achieve perfect recall + NDCG against itself.
    assert by["flat"].recall_at_k == pytest.approx(1.0)
    assert by["flat"].ndcg_at_k == pytest.approx(1.0)
    # All report positive build + memory figures, and NDCG in [0, 1].
    for r in results:
        assert r.num_vectors == 300
        assert r.dimension == 16
        assert r.estimated_memory_bytes > 0
        assert r.ingest_vectors_per_sec > 0
        assert 0.0 <= r.ndcg_at_k <= 1.0


def test_benchmark_skips_unbuildable_recipe(temp_dir):
    # A tiny corpus can't train IVF with nlist=100 (FAISS needs n >= nlist).
    # That recipe must be skipped — not fatal — so the buildable recipes (flat,
    # hnsw: no training) still get scored. Reproduces the "Run benchmark does
    # nothing" bug where one bad recipe sank the whole sweep.
    v = _corpus(20, 16, seed=7)
    skipped: list = []
    results = benchmark_recipes(
        v, ["flat", "hnsw", "ivf"], workdir=temp_dir / "wd_skip",
        ids=[f"c{i}" for i in range(20)], k=5,
        params={"ivf": {"nlist": 100}},
        skipped=skipped,
    )
    got = {r.recipe for r in results}
    assert "flat" in got and "hnsw" in got      # buildable recipes survived
    assert "ivf" not in got                       # untrainable recipe dropped
    assert [s["recipe"] for s in skipped] == ["ivf"]
    assert skipped[0]["reason"]                    # carries a human-readable reason


def test_benchmark_persists_artifact(temp_dir):
    v = _corpus(120, 16)
    out = temp_dir / "artifacts" / "bench.json"
    benchmark_recipes(v, ["flat", "hnsw"], workdir=temp_dir / "wd2",
                      k=5, persist_path=out)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 2
    assert payload["n_vectors"] == 120
    assert len(payload["results"]) == 2
    assert all("ndcg_at_k" in r for r in payload["results"])


def test_benchmark_progress_callback(temp_dir):
    v = _corpus(80, 8)
    seen = []
    benchmark_recipes(v, ["flat", "hnsw"], workdir=temp_dir / "wd3", k=5,
                      progress=lambda pct, msg: seen.append((pct, msg)))
    assert seen
    assert seen[-1][0] == 100.0


# --------------------------------------------------------------------------- #
# Manager.benchmark_index
# --------------------------------------------------------------------------- #


def test_manager_benchmark_index_stores_metrics(temp_dir):
    reg = IndexRegistry(path=temp_dir / "reg.json")
    mgr = IndexManager(registry=reg, indices_root=temp_dir / "named")
    v = _corpus(150, 16, seed=2)
    ids = [f"c{i}" for i in range(150)]
    profile = IndexProfile(name="bench_me", backend="faiss", index_type="flat",
                           embedding_model="synthetic", vector_dimension=16)
    mgr.create_index(profile, [f"d{i}" for i in range(150)], v, None, ids)

    q = v[:15]
    gt = exact_neighbors(v, q, k=5)
    truth = [[ids[j] for j in row] for row in gt]
    scores = mgr.benchmark_index("bench_me", q, truth, k=5)
    assert scores["recall_at_k"] == pytest.approx(1.0)

    # Metrics persisted onto the profile.
    stored = mgr.registry.get("bench_me")
    assert stored.metrics["recall_at_k"] == pytest.approx(1.0)
    assert stored.metrics["k"] == 5


# --------------------------------------------------------------------------- #
# Benchmark job
# --------------------------------------------------------------------------- #


def test_benchmark_job_via_registry(temp_dir):
    import time

    from src.jobs import JobRegistry, benchmark_recipes_job

    reg = JobRegistry(max_workers=2)
    try:
        v = _corpus(150, 16, seed=9)
        job = reg.submit(
            "index_benchmark", benchmark_recipes_job,
            texts=[f"d{i}" for i in range(150)], recipe_ids=["flat", "hnsw"],
            workdir=temp_dir / "wd", embeddings=v, k=5,
        )
        deadline = time.time() + 10
        while time.time() < deadline and not job.status.is_terminal:
            time.sleep(0.02)
        assert job.status.value == "succeeded"
        assert len(job.result["results"]) == 2
        assert job.result["dimension"] == 16
    finally:
        reg.shutdown()
