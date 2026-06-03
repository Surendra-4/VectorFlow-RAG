# tests/test_user_stats.py

"""Per-user stats attribution (Phase 14d): authenticated search/ingest bump the
caller's counters; anonymous calls don't; reset zeroes them."""

from __future__ import annotations

import threading
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_ingest_lock, get_pipeline
from src.db.session import reset_engine
from src.rag_pipeline import RAGPipeline


GOLDEN_DOCS = [
    "Reciprocal rank fusion combines ranked lists from multiple retrievers.",
    "BM25 is a sparse keyword retrieval algorithm based on TF-IDF.",
    "Vector embeddings map text to a continuous high-dimensional space.",
]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> RAGPipeline:
    rag = RAGPipeline(index_dir=str(tmp_path_factory.mktemp("stats_pipe")), enable_cache=False)
    rag.ingest_documents(GOLDEN_DOCS)
    return rag


@pytest.fixture
def client(pipeline, tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/stats.db")
    reset_engine()
    app = create_app(init_pipeline=False)
    lock = threading.Lock()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_ingest_lock] = lambda: lock
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    reset_engine()


def _token(client, email="stats@example.com") -> str:
    return client.post(
        "/api/v1/auth/signup", json={"email": email, "password": "password123"}
    ).json()["access_token"]


def _stats(client, token) -> dict:
    return client.get("/api/v1/auth/me/stats", headers={"Authorization": f"Bearer {token}"}).json()["stats"]


def test_authenticated_search_increments_user_stats(client):
    token = _token(client)
    h = {"Authorization": f"Bearer {token}"}
    assert _stats(client, token)["searches"] == 0

    client.post("/api/v1/search", json={"query": "rank fusion", "k": 2}, headers=h)
    client.post("/api/v1/search", json={"query": "bm25", "k": 2}, headers=h)

    s = _stats(client, token)
    assert s["searches"] == 2
    assert s["retrievals"] == 2
    assert s["last_active_at"] is not None


def test_anonymous_search_records_nothing(client):
    token = _token(client, email="anon-watcher@example.com")
    # An anonymous search (no auth header) must not touch any user's stats.
    client.post("/api/v1/search", json={"query": "embeddings", "k": 2})
    assert _stats(client, token)["searches"] == 0


def test_authenticated_ingest_increments(client):
    token = _token(client, email="ingestor@example.com")
    h = {"Authorization": f"Bearer {token}"}
    client.post("/api/v1/ingest/text", json={"documents": ["doc one", "doc two"], "reset": True}, headers=h)
    s = _stats(client, token)
    assert s["documents_ingested"] == 2
    assert s["chunks_ingested"] >= 1


def test_reset_zeroes_user_stats(client):
    token = _token(client, email="resetter@example.com")
    h = {"Authorization": f"Bearer {token}"}
    client.post("/api/v1/search", json={"query": "anything", "k": 2}, headers=h)
    assert _stats(client, token)["searches"] == 1

    rr = client.post("/api/v1/auth/me/stats/reset", headers=h)
    assert rr.status_code == 200
    body = rr.json()["stats"]
    assert body["searches"] == 0
    assert body["reset_at"] is not None
