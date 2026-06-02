# src/api/routes/observability.py

"""
Observability endpoints (Phase 9).

Read-only access to the metrics registry plus a Prometheus scrape target.
All endpoints are cheap (snapshot copies of bounded-size structures); a
dashboard polling every 1-5 seconds adds negligible load.

Routes:

* ``GET /api/v1/metrics/snapshot``    — full structured dump
* ``GET /api/v1/metrics/requests``    — per-endpoint count + latency
* ``GET /api/v1/metrics/cache``       — cache stats (process + namespace)
* ``GET /api/v1/metrics/retrieval``   — retrieval throughput + stage latency
* ``GET /api/v1/metrics/ingestion``   — ingestion counts + latency
* ``GET /api/v1/metrics/streams``     — SSE session metrics
* ``GET /api/v1/traces/recent``       — recent RetrievalTrace dumps
* ``GET /api/v1/metrics/prometheus``  — Prometheus exposition text
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from src.api.dependencies import (
    get_index_manager,
    get_pipeline,
    get_request_id,
)
from src.observability import get_metrics, to_prometheus_text

router = APIRouter(tags=["observability"])


# --------------------------------------------------------------------------- #
# Snapshot — everything
# --------------------------------------------------------------------------- #


@router.get("/metrics/snapshot")
def metrics_snapshot(
    request_id: str = Depends(get_request_id),
) -> Dict[str, Any]:
    snap = get_metrics().snapshot()
    snap["request_id"] = request_id
    return snap


# --------------------------------------------------------------------------- #
# Per-area views
# --------------------------------------------------------------------------- #


@router.get("/metrics/requests")
def metrics_requests(request_id: str = Depends(get_request_id)) -> Dict[str, Any]:
    m = get_metrics()
    snap = m.snapshot()
    return {
        "request_id": request_id,
        "uptime_s": snap["uptime_s"],
        "requests_total": snap["labeled_counters"]["requests_total"],
        "request_latency_ms": snap["labeled_histograms"]["request_latency_ms"],
    }


@router.get("/metrics/cache")
def metrics_cache(
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
) -> Dict[str, Any]:
    m = get_metrics()
    return {
        "request_id": request_id,
        "backend": pipeline.cache.backend_name,
        "process_stats": pipeline.cache.stats.snapshot(),
        "retrieval_cache_hits_total": m.cache_hits_total.value,
        "cache_ops_total": m.snapshot()["labeled_counters"]["cache_ops_total"],
    }


@router.get("/metrics/retrieval")
def metrics_retrieval(request_id: str = Depends(get_request_id)) -> Dict[str, Any]:
    m = get_metrics()
    snap = m.snapshot()
    return {
        "request_id": request_id,
        "retrievals_total": snap["counters"]["retrievals_total"],
        "reranker_used_total": snap["counters"]["reranker_used_total"],
        "cache_hits_total": snap["counters"]["cache_hits_total"],
        "retrieval_latency_ms": snap["histograms"]["retrieval_latency_ms"],
        "retrieval_stage_latency_ms": snap["labeled_histograms"]["retrieval_stage_latency_ms"],
        "expansion_strategy_usage_total": snap["labeled_counters"]["expansion_strategy_usage_total"],
    }


@router.get("/metrics/ingestion")
def metrics_ingestion(request_id: str = Depends(get_request_id)) -> Dict[str, Any]:
    m = get_metrics()
    snap = m.snapshot()
    return {
        "request_id": request_id,
        "ingestions_total": snap["labeled_counters"]["ingestions_total"],
        "chunks_ingested_total": snap["counters"]["chunks_ingested_total"],
        "ingest_failures_total": snap["counters"]["ingest_failures_total"],
        "ingest_latency_ms": snap["histograms"]["ingest_latency_ms"],
    }


@router.get("/metrics/streams")
def metrics_streams(request_id: str = Depends(get_request_id)) -> Dict[str, Any]:
    m = get_metrics()
    snap = m.snapshot()
    return {
        "request_id": request_id,
        "active_streams": snap["gauges"]["active_streams"],
        "stream_sessions_total": snap["counters"]["stream_sessions_total"],
        "stream_duration_ms": snap["histograms"]["stream_duration_ms"],
    }


# --------------------------------------------------------------------------- #
# Phase 12 — model + index lifecycle views
# --------------------------------------------------------------------------- #


@router.get("/metrics/models")
def metrics_models(request_id: str = Depends(get_request_id)) -> Dict[str, Any]:
    """Provider + model lifecycle metrics."""
    snap = get_metrics().snapshot()
    lc = snap["labeled_counters"]
    return {
        "request_id": request_id,
        "provider_ops_total": lc.get("provider_ops_total", []),
        "provider_chat_total": lc.get("provider_chat_total", []),
        "provider_errors_total": lc.get("provider_errors_total", []),
        "model_installs_total": lc.get("model_installs_total", []),
        "model_switch_total": lc.get("model_switch_total", []),
    }


@router.get("/metrics/indexes")
def metrics_indexes(
    manager=Depends(get_index_manager),
    request_id: str = Depends(get_request_id),
) -> Dict[str, Any]:
    """Index lifecycle metrics + current active-index *state*.

    The active index name is reported as a value (never a metric label) so
    metric cardinality stays bounded regardless of how many indexes exist.
    """
    snap = get_metrics().snapshot()
    return {
        "request_id": request_id,
        "active_index": manager.registry.active_name,
        "index_count": len(manager.registry.names()),
        "index_switch_total": snap["counters"].get("index_switch_total", 0),
        "benchmark_runs_total": snap["counters"].get("benchmark_runs_total", 0),
        "index_jobs_total": snap["labeled_counters"].get("index_jobs_total", []),
        "index_builds_total": snap["labeled_counters"].get("index_builds_total", []),
        "index_build_duration_ms": snap["labeled_histograms"].get("index_build_duration_ms", []),
    }


# --------------------------------------------------------------------------- #
# Recent traces (ring buffer)
# --------------------------------------------------------------------------- #


@router.get("/traces/recent")
def traces_recent(
    limit: int = Query(default=20, ge=1, le=500),
    request_id: str = Depends(get_request_id),
) -> Dict[str, Any]:
    m = get_metrics()
    return {
        "request_id": request_id,
        "limit": limit,
        "traces": m.recent_traces.snapshot(limit=limit),
    }


# --------------------------------------------------------------------------- #
# Prometheus exposition text
# --------------------------------------------------------------------------- #


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
def metrics_prometheus() -> str:
    """
    Prometheus text-format scrape target.

    Returns ``text/plain`` (Prometheus exposition v0.0.4). Compatible
    with any Prom-text scraper without further configuration.
    """
    return to_prometheus_text(get_metrics())
