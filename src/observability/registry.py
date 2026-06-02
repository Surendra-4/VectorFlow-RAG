# src/observability/registry.py

"""
Process-local metrics registry + canonical metric catalog.

The registry is a singleton per process. Multi-worker FastAPI
deployments give each worker its own registry — operators scrape
each worker individually via the Prometheus exporter. This is a
deliberate choice for Phase 9 to avoid the complexity of a shared
metrics backend before it's needed.

Canonical metrics are defined here once so the rest of the codebase
can reference them by name without duplicating definitions.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from src.observability.primitives import (
    Counter,
    Gauge,
    Histogram,
    LabeledCounter,
    LabeledHistogram,
    RingBuffer,
)


class MetricsRegistry:
    """Process-local collection of metric primitives.

    Catalog (created at construction time so the snapshot shape is stable):

    * ``requests_total``        — LabeledCounter(endpoint, status_code)
    * ``request_latency_ms``    — LabeledHistogram(endpoint)
    * ``retrievals_total``      — Counter
    * ``retrieval_latency_ms``  — Histogram
    * ``retrieval_stage_latency_ms`` — LabeledHistogram(stage)
    * ``expansion_strategy_usage_total`` — LabeledCounter(strategy)
    * ``reranker_used_total``   — Counter
    * ``cache_ops_total``       — LabeledCounter(namespace, op)
    * ``ingestions_total``      — LabeledCounter(mode)
    * ``ingest_latency_ms``     — Histogram
    * ``chunks_ingested_total`` — Counter
    * ``ingest_failures_total`` — Counter
    * ``active_streams``        — Gauge
    * ``stream_sessions_total`` — Counter
    * ``stream_duration_ms``    — Histogram
    * ``recent_traces``         — RingBuffer
    * ``recent_errors``         — RingBuffer
    """

    def __init__(self, trace_buffer_size: int = 100, error_buffer_size: int = 100):
        self._start_time = time.monotonic()
        self._lock = threading.Lock()

        # Request-level
        self.requests_total = LabeledCounter(
            "requests_total", ("endpoint", "status_code"),
            "HTTP requests by endpoint and status code",
        )
        self.request_latency_ms = LabeledHistogram(
            "request_latency_ms", ("endpoint",),
            description="HTTP request latency per endpoint",
        )

        # Retrieval
        self.retrievals_total = Counter("retrievals_total", "Search calls")
        self.retrieval_latency_ms = Histogram(
            "retrieval_latency_ms", description="End-to-end search latency"
        )
        self.retrieval_stage_latency_ms = LabeledHistogram(
            "retrieval_stage_latency_ms", ("stage",),
            description="Per-stage latency within search",
        )
        self.expansion_strategy_usage_total = LabeledCounter(
            "expansion_strategy_usage_total", ("strategy",),
            "Times each expansion strategy was invoked",
        )
        self.reranker_used_total = Counter(
            "reranker_used_total", "Searches that ran the cross-encoder reranker"
        )
        self.cache_hits_total = Counter("cache_hits_total", "Full-retrieval cache hits")

        # Cache (namespace breakdown). Currently we attribute by the
        # SafeCache aggregate; finer-grained namespace counting will
        # land when individual wrappers emit metrics themselves.
        self.cache_ops_total = LabeledCounter(
            "cache_ops_total", ("namespace", "op"),
            "Cache operations by namespace and op",
        )

        # Ingestion
        self.ingestions_total = LabeledCounter(
            "ingestions_total", ("mode",),
            "Ingestion calls by mode (text/files/paths)",
        )
        self.ingest_latency_ms = Histogram(
            "ingest_latency_ms", description="Ingestion call latency"
        )
        self.chunks_ingested_total = Counter(
            "chunks_ingested_total", "Total chunks produced across ingestions"
        )
        self.ingest_failures_total = Counter(
            "ingest_failures_total", "Per-file ingest failures (not whole-batch)"
        )

        # Model management (Phase 12). Bounded cardinality: provider names are
        # from a fixed registry; op/status/kind are fixed enums.
        self.provider_ops_total = LabeledCounter(
            "provider_ops_total", ("provider", "op"),
            "Provider operations (list/validate/set_key/delete) by provider",
        )
        self.model_installs_total = LabeledCounter(
            "model_installs_total", ("provider", "status"),
            "Offline model install attempts by provider and outcome",
        )
        self.model_switch_total = LabeledCounter(
            "model_switch_total", ("provider", "kind"),
            "Active-model switch events by provider and model kind",
        )
        # Provider usage + failures. Bounded: provider is a fixed registry,
        # status/error_type are small fixed enums.
        self.provider_chat_total = LabeledCounter(
            "provider_chat_total", ("provider", "status"),
            "Chat generations by provider and outcome",
        )
        self.provider_errors_total = LabeledCounter(
            "provider_errors_total", ("provider", "error_type"),
            "Provider-layer errors by provider and error class",
        )

        # Index lifecycle (Phase 12h/i). Bounded: job type + terminal status,
        # and index_type is a fixed recipe enum (never the unbounded index name).
        self.index_jobs_total = LabeledCounter(
            "index_jobs_total", ("type", "status"),
            "Background jobs by type and terminal status",
        )
        self.index_builds_total = LabeledCounter(
            "index_builds_total", ("index_type", "status"),
            "Index builds by FAISS recipe and outcome",
        )
        self.index_build_duration_ms = LabeledHistogram(
            "index_build_duration_ms", ("index_type",),
            description="Index build wall-clock by recipe",
        )
        self.index_switch_total = Counter(
            "index_switch_total", "Active-index switch events"
        )
        self.benchmark_runs_total = Counter(
            "benchmark_runs_total", "Index benchmark runs"
        )

        # Streaming
        self.active_streams = Gauge("active_streams", "Currently-open SSE streams")
        self.stream_sessions_total = Counter(
            "stream_sessions_total", "Lifetime SSE sessions opened"
        )
        self.stream_duration_ms = Histogram(
            "stream_duration_ms", description="Completed SSE session duration"
        )

        # Ring buffers
        self.recent_traces = RingBuffer(
            "recent_traces", max_size=trace_buffer_size,
            description="Recent RetrievalTrace snapshots",
        )
        self.recent_errors = RingBuffer(
            "recent_errors", max_size=error_buffer_size,
            description="Recent error events",
        )

    # ------------------------------------------------------------------ #
    # Snapshot — full dump for the dashboard
    # ------------------------------------------------------------------ #

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._start_time

    def snapshot(self) -> Dict[str, Any]:
        """Full structured dump suitable for JSON serialization."""
        return {
            "uptime_s": self.uptime_s,
            "counters": {
                "retrievals_total": self.retrievals_total.value,
                "reranker_used_total": self.reranker_used_total.value,
                "cache_hits_total": self.cache_hits_total.value,
                "chunks_ingested_total": self.chunks_ingested_total.value,
                "ingest_failures_total": self.ingest_failures_total.value,
                "stream_sessions_total": self.stream_sessions_total.value,
                "index_switch_total": self.index_switch_total.value,
                "benchmark_runs_total": self.benchmark_runs_total.value,
            },
            "gauges": {
                "active_streams": self.active_streams.value,
            },
            "labeled_counters": {
                "requests_total": _labels_to_dict(
                    self.requests_total.label_names, self.requests_total.items()
                ),
                "expansion_strategy_usage_total": _labels_to_dict(
                    self.expansion_strategy_usage_total.label_names,
                    self.expansion_strategy_usage_total.items(),
                ),
                "cache_ops_total": _labels_to_dict(
                    self.cache_ops_total.label_names, self.cache_ops_total.items()
                ),
                "ingestions_total": _labels_to_dict(
                    self.ingestions_total.label_names, self.ingestions_total.items()
                ),
                "provider_ops_total": _labels_to_dict(
                    self.provider_ops_total.label_names, self.provider_ops_total.items()
                ),
                "model_installs_total": _labels_to_dict(
                    self.model_installs_total.label_names, self.model_installs_total.items()
                ),
                "model_switch_total": _labels_to_dict(
                    self.model_switch_total.label_names, self.model_switch_total.items()
                ),
                "provider_chat_total": _labels_to_dict(
                    self.provider_chat_total.label_names, self.provider_chat_total.items()
                ),
                "provider_errors_total": _labels_to_dict(
                    self.provider_errors_total.label_names, self.provider_errors_total.items()
                ),
                "index_jobs_total": _labels_to_dict(
                    self.index_jobs_total.label_names, self.index_jobs_total.items()
                ),
                "index_builds_total": _labels_to_dict(
                    self.index_builds_total.label_names, self.index_builds_total.items()
                ),
            },
            "histograms": {
                "retrieval_latency_ms": self.retrieval_latency_ms.snapshot(),
                "ingest_latency_ms": self.ingest_latency_ms.snapshot(),
                "stream_duration_ms": self.stream_duration_ms.snapshot(),
            },
            "labeled_histograms": {
                "request_latency_ms": _labeled_hist_to_dict(
                    self.request_latency_ms.label_names, self.request_latency_ms.items()
                ),
                "retrieval_stage_latency_ms": _labeled_hist_to_dict(
                    self.retrieval_stage_latency_ms.label_names,
                    self.retrieval_stage_latency_ms.items(),
                ),
                "index_build_duration_ms": _labeled_hist_to_dict(
                    self.index_build_duration_ms.label_names,
                    self.index_build_duration_ms.items(),
                ),
            },
            "ring_buffer_sizes": {
                "recent_traces": len(self.recent_traces),
                "recent_errors": len(self.recent_errors),
            },
        }


def _labels_to_dict(label_names, items) -> List[Dict[str, Any]]:
    """Materialize labeled-counter items into JSON-friendly records."""
    out: List[Dict[str, Any]] = []
    for labels, value in items:
        entry = {name: lbl for name, lbl in zip(label_names, labels)}
        entry["value"] = value
        out.append(entry)
    return out


def _labeled_hist_to_dict(label_names, items) -> List[Dict[str, Any]]:
    """Materialize labeled-histogram items into JSON-friendly records."""
    out: List[Dict[str, Any]] = []
    for labels, snap in items:
        entry = {name: lbl for name, lbl in zip(label_names, labels)}
        entry["stats"] = snap
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Singleton accessor
# --------------------------------------------------------------------------- #


_REGISTRY: "MetricsRegistry | None" = None
_REGISTRY_LOCK = threading.Lock()


def get_metrics() -> MetricsRegistry:
    """Return the process-local MetricsRegistry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = MetricsRegistry()
    return _REGISTRY


def reset_metrics() -> None:
    """Replace the singleton with a fresh registry — primarily for tests."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = MetricsRegistry()
