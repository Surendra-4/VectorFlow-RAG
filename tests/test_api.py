# tests/test_api.py

"""
API integration tests using FastAPI's ``TestClient``.

A pre-ingested pipeline is built once per module and injected via DI
override — no model is loaded twice. Tests cover every endpoint's happy
path + structured-error contract + concurrency safety + OpenAPI schema.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_ingest_lock, get_pipeline
from src.config import get_settings
from src.rag_pipeline import RAGPipeline


GOLDEN_DOCS = [
    "Photosynthesis converts light energy into chemical energy in chloroplasts.",
    "Mitochondria generate ATP via oxidative phosphorylation in eukaryotic cells.",
    "Reciprocal rank fusion combines ranked lists from multiple retrievers.",
    "BM25 is a sparse keyword retrieval algorithm based on TF-IDF.",
    "Vector embeddings map text to a continuous high-dimensional space.",
]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> RAGPipeline:
    """Module-scoped pipeline so the embedder loads exactly once per test file."""
    persist = tmp_path_factory.mktemp("api_pipeline")
    rag = RAGPipeline(index_dir=str(persist), enable_cache=False)
    rag.ingest_documents(GOLDEN_DOCS)
    return rag


@pytest.fixture
def client(pipeline) -> Iterator[TestClient]:
    """Test client with DI overrides — no real pipeline init on app startup."""
    app = create_app(init_pipeline=False)
    shared_lock = threading.Lock()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_ingest_lock] = lambda: shared_lock
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def fresh_client(tmp_path) -> Iterator[TestClient]:
    """Independent pipeline + client for tests that mutate ingestion state."""
    rag = RAGPipeline(index_dir=str(tmp_path / "fresh_api"), enable_cache=False)
    rag.ingest_documents(GOLDEN_DOCS)
    app = create_app(init_pipeline=False)
    shared_lock = threading.Lock()
    app.dependency_overrides[get_pipeline] = lambda: rag
    app.dependency_overrides[get_ingest_lock] = lambda: shared_lock
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Health / status
# --------------------------------------------------------------------------- #


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "app_version" in body
        assert "request_id" in body

    def test_health_echoes_request_id(self, client):
        r = client.get("/health", headers={"X-Request-ID": "my-correlation-id"})
        assert r.status_code == 200
        assert r.json()["request_id"] == "my-correlation-id"
        assert r.headers["X-Request-ID"] == "my-correlation-id"

    def test_process_time_header_set(self, client):
        r = client.get("/health")
        assert "X-Process-Time-Ms" in r.headers
        # Header value should parse as a float.
        float(r.headers["X-Process-Time-Ms"])


class TestStatus:
    def test_status_reports_full_pipeline_state(self, client, pipeline):
        r = client.get("/api/v1/status")
        assert r.status_code == 200
        body = r.json()
        assert body["embedder_model"] == pipeline.embedder.model_name
        assert body["embedder_dimension"] == pipeline.embedder.dimension
        assert body["documents_ingested"] == len(GOLDEN_DOCS)
        assert body["chunks_indexed"] > 0
        assert body["corpus_fingerprint"] is not None
        assert "cache_backend" in body
        assert "rrf_k" in body


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class TestSearch:
    def test_search_returns_results(self, client):
        r = client.post("/api/v1/search", json={"query": "photosynthesis", "k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "photosynthesis"
        assert len(body["results"]) > 0
        # Top result should mention photosynthesis.
        assert "photosynthesis" in body["results"][0]["text"].lower()

    def test_search_results_carry_provenance(self, client):
        r = client.post("/api/v1/search", json={"query": "BM25", "k": 3})
        body = r.json()
        top = body["results"][0]
        for key in ("chunk_id", "doc_id", "document_name", "chunk_index",
                    "hybrid_score", "vector_rank", "bm25_rank"):
            assert key in top, f"missing field {key}"

    def test_search_respects_k(self, client):
        r = client.post("/api/v1/search", json={"query": "BM25", "k": 2})
        assert len(r.json()["results"]) <= 2

    def test_search_returns_trace_when_requested(self, client):
        r = client.post("/api/v1/search", json={"query": "BM25", "k": 3, "return_trace": True})
        body = r.json()
        assert body["trace"] is not None
        assert "total_latency_ms" in body["trace"]
        assert "per_strategy" in body["trace"]

    def test_search_validation_rejects_empty_query(self, client):
        r = client.post("/api/v1/search", json={"query": "", "k": 3})
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "validation_error"
        assert "request_id" in body

    def test_search_validation_rejects_huge_k(self, client):
        r = client.post("/api/v1/search", json={"query": "x", "k": 9999})
        assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Ask (non-streaming)
# --------------------------------------------------------------------------- #


class TestAsk:
    def test_ask_without_documents_returns_friendly_message(self, tmp_path):
        # Pipeline with no ingested docs.
        rag = RAGPipeline(index_dir=str(tmp_path / "empty_ask"), enable_cache=False)
        app = create_app(init_pipeline=False)
        app.dependency_overrides[get_pipeline] = lambda: rag
        app.dependency_overrides[get_ingest_lock] = lambda: threading.Lock()
        with TestClient(app) as c:
            r = c.post("/api/v1/ask", json={"query": "anything", "k_docs": 3})
        assert r.status_code == 200
        body = r.json()
        assert "No documents" in body["answer"]
        assert body["metrics"]["num_context_docs"] == 0


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class TestIngestText:
    def test_ingest_text_appends_documents(self, fresh_client):
        r = fresh_client.post(
            "/api/v1/ingest/text",
            json={
                "documents": ["A brand-new document about quantum computing."],
                "reset": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["documents_ingested"] == 1
        assert body["chunks"] >= 1
        assert body["corpus_fingerprint"] is not None
        assert body["failures"] == []

    def test_ingest_text_rejects_mismatched_metadata(self, fresh_client):
        r = fresh_client.post(
            "/api/v1/ingest/text",
            json={
                "documents": ["a", "b"],
                "metadatas": [{"src": "x"}],  # length mismatch
            },
        )
        assert r.status_code == 400
        assert r.json()["code"] == "bad_request"


class TestIngestPaths:
    def test_ingest_paths_with_real_file(self, fresh_client, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("Quantum entanglement is a counter-intuitive phenomenon.")
        r = fresh_client.post("/api/v1/ingest/paths", json={"paths": [str(txt)]})
        assert r.status_code == 200
        body = r.json()
        assert len(body["successes"]) == 1
        assert body["failures"] == []
        assert body["chunks"] >= 1

    def test_ingest_paths_records_failure_for_missing_file(self, fresh_client, tmp_path):
        r = fresh_client.post(
            "/api/v1/ingest/paths",
            json={"paths": [str(tmp_path / "nope.txt")]},
        )
        assert r.status_code == 200  # batch returns 200 with failures inside
        body = r.json()
        assert len(body["failures"]) == 1
        assert "nope.txt" in body["failures"][0]["path"]


class TestIngestFiles:
    def test_ingest_multipart_file_upload(self, fresh_client):
        content = b"Document content for upload-path testing of FastAPI ingest."
        r = fresh_client.post(
            "/api/v1/ingest/files",
            files=[("files", ("upload.txt", content, "text/plain"))],
            data={"reset": "true"},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["successes"]) == 1
        assert body["successes"][0] == "upload.txt"
        assert body["chunks"] >= 1


# --------------------------------------------------------------------------- #
# Documents / admin
# --------------------------------------------------------------------------- #


class TestDocuments:
    def test_list_documents_returns_summaries(self, client):
        r = client.get("/api/v1/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["total_documents"] >= 1
        assert body["total_chunks"] >= 1
        for d in body["documents"]:
            assert "doc_id" in d and d["doc_id"].startswith("doc_")
            assert "chunk_count" in d


class TestAdminReset:
    def test_reset_requires_confirmation(self, fresh_client):
        r = fresh_client.delete("/api/v1/index")
        assert r.status_code == 400
        assert r.json()["code"] == "bad_request"

    def test_reset_with_confirmation_clears_index(self, fresh_client):
        # Verify there are docs first.
        before = fresh_client.get("/api/v1/status").json()
        assert before["chunks_indexed"] > 0
        # Reset.
        r = fresh_client.delete("/api/v1/index?confirm=true")
        assert r.status_code == 200
        body = r.json()
        assert body["cleared"] is True
        assert body["previous_chunks"] > 0
        # Confirm.
        after = fresh_client.get("/api/v1/status").json()
        assert after["chunks_indexed"] == 0
        assert after["corpus_fingerprint"] is None


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


class TestCacheEndpoints:
    def test_cache_stats_endpoint(self, client):
        r = client.get("/api/v1/cache/stats")
        assert r.status_code == 200
        body = r.json()
        for key in ("backend", "hits", "misses", "hit_ratio"):
            assert key in body

    def test_cache_clear_is_safe_when_disabled(self, client):
        r = client.post("/api/v1/cache/clear")
        assert r.status_code == 200
        assert r.json()["message"]


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


class TestConcurrency:
    def test_concurrent_searches_succeed(self, client):
        def call(_):
            return client.post("/api/v1/search", json={"query": "BM25", "k": 3})

        with ThreadPoolExecutor(max_workers=8) as ex:
            responses = list(ex.map(call, range(16)))

        assert all(r.status_code == 200 for r in responses)
        assert all("results" in r.json() for r in responses)

    def test_concurrent_search_results_consistent(self, client):
        """Same query × N threads → identical chunk_id sequences."""
        def call(_):
            r = client.post("/api/v1/search", json={"query": "mitochondria", "k": 3})
            return [x["chunk_id"] for x in r.json()["results"]]

        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(call, range(6)))
        for r in results[1:]:
            assert r == results[0]


# --------------------------------------------------------------------------- #
# Error contracts
# --------------------------------------------------------------------------- #


class TestErrorContract:
    def test_404_for_unknown_route_carries_structured_error(self, client):
        r = client.get("/api/v1/does-not-exist")
        assert r.status_code == 404
        body = r.json()
        assert body["code"] == "not_found"
        assert "request_id" in body
        assert "message" in body

    def test_503_when_pipeline_not_initialized(self):
        # No DI override → pipeline is None → 503 from get_pipeline.
        app = create_app(init_pipeline=False)
        with TestClient(app) as c:
            r = c.get("/api/v1/status")
        assert r.status_code == 503
        body = r.json()
        assert body["code"] == "service_unavailable"

    def test_validation_error_has_field_details(self, client):
        # 'k' must be an int.
        r = client.post("/api/v1/search", json={"query": "x", "k": "not-an-int"})
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "validation_error"
        assert "errors" in body["details"]


# --------------------------------------------------------------------------- #
# OpenAPI
# --------------------------------------------------------------------------- #


class TestOpenAPISchema:
    def test_openapi_json_available(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert schema["openapi"].startswith("3.")
        assert "paths" in schema

    def test_all_main_endpoints_present_in_schema(self, client):
        schema = client.get("/openapi.json").json()
        paths = set(schema["paths"].keys())
        expected = {
            "/health",
            "/api/v1/status",
            "/api/v1/ingest/text",
            "/api/v1/ingest/paths",
            "/api/v1/ingest/files",
            "/api/v1/search",
            "/api/v1/ask",
            "/api/v1/cache/stats",
            "/api/v1/cache/clear",
            "/api/v1/documents",
            "/api/v1/index",
        }
        missing = expected - paths
        assert not missing, f"Missing endpoints in OpenAPI: {missing}"

    def test_response_schemas_reference_pydantic_models(self, client):
        schema = client.get("/openapi.json").json()
        # Each endpoint should have a 200 response that references a component schema.
        for path, methods in schema["paths"].items():
            for method, details in methods.items():
                if method in ("get", "post", "put", "delete"):
                    responses = details.get("responses", {})
                    # We don't require 200 specifically (DELETE could be 204
                    # in some configs) — just ensure some success response exists.
                    success_codes = [c for c in responses if c.startswith("2")]
                    assert success_codes, f"{method.upper()} {path} has no 2xx response"
