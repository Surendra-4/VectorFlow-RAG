# tests/test_api_indexes.py

"""
API tests for the index + jobs routes (Phase 12h). Uses a real (small) pipeline
so /indexes create → background build job → /jobs status works end-to-end.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import (
    get_index_manager,
    get_job_registry,
    get_pipeline,
    get_runtime_config,
)
from src.config import Settings
from src.indexing import IndexManager, IndexRegistry
from src.jobs import JobRegistry
from src.rag_pipeline import RAGPipeline
from src.runtime_config import RuntimeConfigStore


GOLDEN_DOCS = [
    "Photosynthesis converts light energy into chemical energy in chloroplasts.",
    "Mitochondria generate ATP via oxidative phosphorylation in eukaryotic cells.",
    "Reciprocal rank fusion combines ranked lists from multiple retrievers.",
    "BM25 is a sparse keyword retrieval algorithm based on TF-IDF.",
    "Vector embeddings map text to a continuous high-dimensional space.",
]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> RAGPipeline:
    persist = tmp_path_factory.mktemp("idx_api_pipeline")
    rag = RAGPipeline(index_dir=str(persist), enable_cache=False)
    rag.ingest_documents(GOLDEN_DOCS)
    return rag


@pytest.fixture
def client(pipeline, tmp_path) -> Iterator[TestClient]:
    manager = IndexManager(
        registry=IndexRegistry(path=tmp_path / "reg.json"),
        indices_root=tmp_path / "named",
    )
    jobs = JobRegistry(max_workers=2)
    runtime = RuntimeConfigStore(Settings(), path=tmp_path / "runtime.json")

    app = create_app(init_pipeline=False)
    lock = threading.Lock()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_index_manager] = lambda: manager
    app.dependency_overrides[get_job_registry] = lambda: jobs
    app.dependency_overrides[get_runtime_config] = lambda: runtime
    with TestClient(app) as c:
        c._manager = manager  # type: ignore[attr-defined]
        c._jobs = jobs        # type: ignore[attr-defined]
        yield c
    app.dependency_overrides.clear()
    jobs.shutdown(wait=False)


def _wait_job(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/jobs/{job_id}")
        if r.json()["job"]["status"] in ("succeeded", "failed", "cancelled"):
            return r.json()["job"]
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #


def test_list_recipes(client):
    r = client.get("/api/v1/indexes/recipes")
    assert r.status_code == 200
    ids = {rec["id"] for rec in r.json()["recipes"]}
    assert {"flat", "hnsw", "ivf_pq", "opq_ivf_pq"} <= ids


def test_list_recipes_basic_mode(client):
    r = client.get("/api/v1/indexes/recipes", params={"mode": "basic"})
    ids = {rec["id"] for rec in r.json()["recipes"]}
    assert ids == {"flat", "hnsw", "ivf", "pq"}


def test_validate_recipe_ok(client):
    r = client.post("/api/v1/indexes/recipes/validate",
                    json={"recipe": "ivf_pq", "params": {"nlist": 64, "pq_m": 8}, "dim": 384,
                          "n_vectors": 5000})
    assert r.status_code == 200
    v = r.json()["validation"]
    assert v["ok"] is True
    assert v["estimate"]["memory_bytes"] > 0


def test_validate_recipe_reports_errors(client):
    r = client.post("/api/v1/indexes/recipes/validate",
                    json={"recipe": "pq", "params": {"pq_m": 7}, "dim": 384})
    v = r.json()["validation"]
    assert v["ok"] is False
    assert any(e["field"] == "pq_m" for e in v["errors"])


# --------------------------------------------------------------------------- #
# Build → job → list/switch/delete
# --------------------------------------------------------------------------- #


def test_create_index_runs_as_job(client):
    r = client.post("/api/v1/indexes", json={
        "name": "flat_idx", "backend": "faiss", "index_type": "flat", "make_active": True,
    })
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    job = _wait_job(client, job_id)
    assert job["status"] == "succeeded"
    assert job["result"]["index_name"] == "flat_idx"
    assert job["result"]["num_vectors"] == len(GOLDEN_DOCS) or job["result"]["num_vectors"] >= 1

    # It now appears in the listing and is active.
    r2 = client.get("/api/v1/indexes")
    names = {i["name"] for i in r2.json()["indexes"]}
    assert "flat_idx" in names
    assert r2.json()["active"] == "flat_idx"


def test_create_index_no_corpus_400(client, tmp_path):
    # A fresh pipeline with nothing ingested.
    empty = RAGPipeline(index_dir=str(tmp_path / "empty"), enable_cache=False)
    client.app.dependency_overrides[get_pipeline] = lambda: empty
    try:
        r = client.post("/api/v1/indexes", json={"name": "x", "index_type": "flat"})
        assert r.status_code == 400
    finally:
        client.app.dependency_overrides[get_pipeline] = lambda: client._manager  # restore-ish


def test_create_index_invalid_recipe_params_400(client):
    r = client.post("/api/v1/indexes", json={
        "name": "bad", "backend": "faiss", "index_type": "pq",
        "build_params": {"pq_m": 7},  # 384 not divisible by 7
    })
    assert r.status_code == 400


def test_switch_and_delete_index(client):
    # Build two indexes.
    for name in ("idx_a", "idx_b"):
        jid = client.post("/api/v1/indexes",
                          json={"name": name, "index_type": "flat"}).json()["job_id"]
        _wait_job(client, jid)

    r = client.post("/api/v1/indexes/idx_b/switch")
    assert r.status_code == 200
    assert r.json()["active"] == "idx_b"

    r = client.request("DELETE", "/api/v1/indexes/idx_a")
    assert r.status_code == 200
    assert r.json()["action"] == "deleted"
    names = {i["name"] for i in client.get("/api/v1/indexes").json()["indexes"]}
    assert "idx_a" not in names


def test_get_unknown_index_404(client):
    assert client.get("/api/v1/indexes/ghost").status_code == 404


# --------------------------------------------------------------------------- #
# Compatibility
# --------------------------------------------------------------------------- #


def test_compatibility_reuse_for_matching_index(client):
    jid = client.post("/api/v1/indexes",
                      json={"name": "compat_idx", "index_type": "hnsw"}).json()["job_id"]
    _wait_job(client, jid)
    # staged index settings default to the same model/backend? default backend
    # is chromadb, so faiss index will report a backend mismatch (rebuild/create).
    r = client.get("/api/v1/indexes/compat_idx/compatibility")
    assert r.status_code == 200
    report = r.json()["report"]
    assert "compatible" in report
    assert "action" in report and "message" in report


# --------------------------------------------------------------------------- #
# Jobs API
# --------------------------------------------------------------------------- #


def test_jobs_list_and_get(client):
    jid = client.post("/api/v1/indexes",
                      json={"name": "for_jobs", "index_type": "flat"}).json()["job_id"]
    _wait_job(client, jid)

    r = client.get("/api/v1/jobs")
    assert r.status_code == 200
    assert any(j["id"] == jid for j in r.json()["jobs"])

    r2 = client.get(f"/api/v1/jobs/{jid}")
    assert r2.status_code == 200
    assert "history" in r2.json()["job"]


def test_job_stream_sse(client):
    jid = client.post("/api/v1/indexes",
                      json={"name": "for_stream", "index_type": "flat"}).json()["job_id"]
    r = client.get(f"/api/v1/jobs/{jid}/stream")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in r.text


def test_cancel_unknown_job_404(client):
    assert client.post("/api/v1/jobs/nope/cancel").status_code == 404


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


def test_benchmark_recipes_runs_as_job(client):
    r = client.post("/api/v1/indexes/benchmark",
                    json={"recipes": ["flat", "hnsw"], "k": 3, "persist": False})
    assert r.status_code == 202
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "succeeded"
    recipes = {res["recipe"] for res in job["result"]["results"]}
    assert recipes == {"flat", "hnsw"}
    # Flat is exact → perfect recall on the self-retrieval benchmark.
    flat = next(res for res in job["result"]["results"] if res["recipe"] == "flat")
    assert flat["recall_at_k"] == 1.0


def test_benchmark_unknown_recipe_400(client):
    r = client.post("/api/v1/indexes/benchmark", json={"recipes": ["flat", "bogus"]})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Phase 13 — live switch into the retrieval path
# --------------------------------------------------------------------------- #


def test_switch_activates_index_in_live_pipeline(client, pipeline):
    client.post("/api/v1/indexes/activate-default")  # order-independent reset
    # Build a named index over the current corpus (carries chunk_ids + meta).
    jid = client.post("/api/v1/indexes",
                      json={"name": "live_flat", "index_type": "flat"}).json()["job_id"]
    job = _wait_job(client, jid)
    assert job["status"] == "succeeded"

    # Baseline search on the default index.
    before = client.post("/api/v1/search", json={"query": "reciprocal rank fusion", "k": 3}).json()
    assert pipeline.active_index_name is None

    # Switch → now serving live retrieval.
    r = client.post("/api/v1/indexes/live_flat/switch")
    assert r.status_code == 200
    assert pipeline.active_index_name == "live_flat"

    # Status reflects the active index.
    assert client.get("/api/v1/status").json()["active_index_name"] == "live_flat"

    # Search still works, same chunk-id space, provenance preserved.
    after = client.post("/api/v1/search", json={"query": "reciprocal rank fusion", "k": 3}).json()
    assert after["results"][0]["chunk_id"] == before["results"][0]["chunk_id"]
    assert after["results"][0]["document_name"] is not None

    # Revert to default.
    rd = client.post("/api/v1/indexes/activate-default")
    assert rd.status_code == 200
    assert pipeline.active_index_name is None
    assert client.get("/api/v1/status").json()["active_index_name"] is None


def test_switch_incompatible_index_returns_409(client, pipeline):
    client.post("/api/v1/indexes/activate-default")  # order-independent reset
    # Build a valid index, then corrupt its profile to a different embedding
    # model so the switch-compatibility check blocks it.
    jid = client.post("/api/v1/indexes",
                      json={"name": "incompat", "index_type": "flat"}).json()["job_id"]
    _wait_job(client, jid)

    prof = client._manager.registry.get("incompat")  # type: ignore[attr-defined]
    prof.embedding_model = "some/other-model"
    client._manager.registry.update(prof)  # type: ignore[attr-defined]

    r = client.post("/api/v1/indexes/incompat/switch")
    assert r.status_code == 409
    body = r.json()
    assert body["code"] == "index_incompatible"
    report = body["details"]["compatibility"]
    assert report["compatible"] is False
    assert report["action"] == "create_new"
    # Pipeline was NOT switched.
    assert pipeline.active_index_name is None


def test_activate_default_idempotent(client):
    r = client.post("/api/v1/indexes/activate-default")
    assert r.status_code == 200
    assert r.json()["action"] == "activated_default"


def test_index_routes_in_openapi(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/indexes" in paths
    assert "/api/v1/indexes/recipes" in paths
    assert "/api/v1/indexes/{name}/compatibility" in paths
    assert "/api/v1/jobs/{job_id}/stream" in paths
