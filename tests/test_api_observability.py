# tests/test_api_observability.py

"""
Tests for Phase 9 observability endpoints + collector hooks.

Mirrors the test_api.py pattern: module-scoped pipeline with DI
override so the embedder loads only once.
"""

from __future__ import annotations

import threading
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_ingest_lock, get_pipeline
from src.observability import get_metrics, reset_metrics
from src.rag_pipeline import RAGPipeline


GOLDEN_DOCS = [
    "Photosynthesis converts light energy into chemical energy in plant chloroplasts.",
    "Mitochondria generate ATP via oxidative phosphorylation in eukaryotic cells.",
    "The capital of France is Paris, located on the Seine river.",
    "BM25 is a sparse keyword retrieval algorithm based on TF-IDF.",
    "Reciprocal rank fusion combines ranked lists from multiple retrievers.",
]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> RAGPipeline:
    persist = tmp_path_factory.mktemp("api_obs_pipeline")
    rag = RAGPipeline(index_dir=str(persist), enable_cache=False)
    rag.ingest_documents(GOLDEN_DOCS)
    return rag


@pytest.fixture
def client(pipeline) -> Iterator[TestClient]:
    reset_metrics()  # fresh registry per test
    app = create_app(init_pipeline=False)
    shared_lock = threading.Lock()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_ingest_lock] = lambda: shared_lock
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# /metrics/snapshot
# --------------------------------------------------------------------------- #


class TestSnapshot:
    def test_snapshot_has_canonical_shape(self, client):
        r = client.get("/api/v1/metrics/snapshot")
        assert r.status_code == 200
        body = r.json()
        for key in (
            "uptime_s", "counters", "gauges", "labeled_counters",
            "histograms", "labeled_histograms", "ring_buffer_sizes",
            "request_id",
        ):
            assert key in body

    def test_snapshot_carries_request_id_header(self, client):
        r = client.get("/api/v1/metrics/snapshot")
        assert "X-Request-ID" in r.headers
        assert r.json()["request_id"] == r.headers["X-Request-ID"]

    def test_snapshot_uptime_positive(self, client):
        r = client.get("/api/v1/metrics/snapshot")
        assert r.json()["uptime_s"] >= 0


# --------------------------------------------------------------------------- #
# Request metrics (collector hook in TimingMiddleware)
# --------------------------------------------------------------------------- #


class TestRequestMetrics:
    def test_requests_total_increments_on_each_call(self, client):
        # Hit the health endpoint a few times.
        for _ in range(3):
            client.get("/health")
        r = client.get("/api/v1/metrics/requests")
        body = r.json()
        # Find /health entries.
        health_entries = [
            e for e in body["requests_total"] if e["endpoint"].endswith("/health")
        ]
        assert health_entries, body["requests_total"]
        # 3 calls + the implicit /api/v1/metrics/requests call gets counted
        # on its OWN response status, so we just check the count is >=3.
        assert sum(e["value"] for e in health_entries) >= 3

    def test_per_endpoint_latency_histograms_populated(self, client):
        for _ in range(3):
            client.get("/health")
        r = client.get("/api/v1/metrics/requests")
        body = r.json()
        health_lat = [
            e for e in body["request_latency_ms"]
            if e["endpoint"].endswith("/health")
        ]
        assert health_lat
        # Each entry has a stats dict.
        stats = health_lat[0]["stats"]
        assert stats["count"] >= 3
        assert stats["p50"] is not None

    def test_failed_request_recorded_with_status_code(self, client):
        # POST /search without a body → 422 from FastAPI validation.
        client.post("/api/v1/search", json={})
        r = client.get("/api/v1/metrics/requests")
        body = r.json()
        codes = {e["status_code"] for e in body["requests_total"]}
        assert "422" in codes


# --------------------------------------------------------------------------- #
# Retrieval metrics (collector hook in RAGPipeline.search)
# --------------------------------------------------------------------------- #


class TestRetrievalMetrics:
    def test_retrievals_total_increments_on_search(self, client):
        client.post("/api/v1/search", json={"query": "photosynthesis", "k": 3})
        r = client.get("/api/v1/metrics/retrieval")
        body = r.json()
        assert body["retrievals_total"] >= 1

    def test_retrieval_latency_histogram_observed(self, client):
        client.post("/api/v1/search", json={"query": "photosynthesis", "k": 3})
        client.post("/api/v1/search", json={"query": "mitochondria", "k": 3})
        r = client.get("/api/v1/metrics/retrieval")
        body = r.json()
        assert body["retrieval_latency_ms"]["count"] >= 2
        assert body["retrieval_latency_ms"]["p50"] is not None

    def test_recent_traces_populated_after_search(self, client):
        client.post("/api/v1/search", json={"query": "BM25", "k": 2})
        r = client.get("/api/v1/traces/recent?limit=10")
        body = r.json()
        assert body["limit"] == 10
        assert len(body["traces"]) >= 1
        latest = body["traces"][-1]
        assert latest["original_query"] == "BM25"
        # Trace dict should be JSON-serializable end-to-end.
        assert "total_latency_ms" in latest


# --------------------------------------------------------------------------- #
# Cache metrics
# --------------------------------------------------------------------------- #


class TestCacheMetrics:
    def test_cache_endpoint_returns_backend_and_stats(self, client):
        r = client.get("/api/v1/metrics/cache")
        assert r.status_code == 200
        body = r.json()
        assert "backend" in body
        assert "process_stats" in body
        assert "retrieval_cache_hits_total" in body


# --------------------------------------------------------------------------- #
# Ingestion metrics (collector hook in ingest routes)
# --------------------------------------------------------------------------- #


class TestIngestionMetrics:
    def test_ingestion_records_metric(self, client):
        # Force a small text ingest.
        r = client.post(
            "/api/v1/ingest/text",
            json={"documents": ["a small test document"], "reset": False},
        )
        assert r.status_code == 200
        metrics = client.get("/api/v1/metrics/ingestion").json()
        modes = {e["mode"] for e in metrics["ingestions_total"]}
        assert "text" in modes
        assert metrics["chunks_ingested_total"] >= 1


# --------------------------------------------------------------------------- #
# Streams (gauge / counter exposed even when no stream is active)
# --------------------------------------------------------------------------- #


class TestStreamMetrics:
    def test_stream_metrics_shape(self, client):
        r = client.get("/api/v1/metrics/streams")
        assert r.status_code == 200
        body = r.json()
        assert "active_streams" in body
        assert "stream_sessions_total" in body
        assert "stream_duration_ms" in body


# --------------------------------------------------------------------------- #
# Recent traces
# --------------------------------------------------------------------------- #


class TestRecentTraces:
    def test_limit_is_clamped(self, client):
        r = client.get("/api/v1/traces/recent?limit=1000")
        # ge=1, le=500 → 1000 should be rejected with 422
        assert r.status_code == 422

    def test_default_limit_returns_recent(self, client):
        client.post("/api/v1/search", json={"query": "BM25", "k": 1})
        r = client.get("/api/v1/traces/recent")
        assert r.status_code == 200
        assert isinstance(r.json()["traces"], list)


# --------------------------------------------------------------------------- #
# Prometheus exporter
# --------------------------------------------------------------------------- #


class TestPrometheusEndpoint:
    def test_returns_text_plain(self, client):
        r = client.get("/api/v1/metrics/prometheus")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")

    def test_contains_expected_metric_names(self, client):
        client.post("/api/v1/search", json={"query": "photosynthesis", "k": 2})
        text = client.get("/api/v1/metrics/prometheus").text
        assert "# TYPE retrievals_total counter" in text
        assert "# TYPE requests_total counter" in text
        assert "# TYPE active_streams gauge" in text
        assert "# TYPE retrieval_latency_ms summary" in text

    def test_text_format_is_terminated(self, client):
        text = client.get("/api/v1/metrics/prometheus").text
        assert text.endswith("\n")


# --------------------------------------------------------------------------- #
# OpenAPI schema includes new endpoints
# --------------------------------------------------------------------------- #


class TestOpenAPI:
    def test_openapi_contains_observability_paths(self, client):
        schema = client.get("/openapi.json").json()
        paths = set(schema["paths"].keys())
        for p in (
            "/api/v1/metrics/snapshot",
            "/api/v1/metrics/requests",
            "/api/v1/metrics/cache",
            "/api/v1/metrics/retrieval",
            "/api/v1/metrics/ingestion",
            "/api/v1/metrics/streams",
            "/api/v1/traces/recent",
            "/api/v1/metrics/prometheus",
        ):
            assert p in paths, f"missing: {p}"


# --------------------------------------------------------------------------- #
# Concurrent collection — gauge and counters stay consistent
# --------------------------------------------------------------------------- #


class TestConcurrentCollection:
    def test_concurrent_requests_do_not_corrupt_counters(self, client):
        import concurrent.futures as cf

        def do_search(i):
            return client.post("/api/v1/search", json={"query": f"q{i}", "k": 1}).status_code

        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(do_search, range(40)))
        assert all(s == 200 for s in results)

        # All 40 searches recorded.
        r = client.get("/api/v1/metrics/retrieval").json()
        assert r["retrievals_total"] >= 40
