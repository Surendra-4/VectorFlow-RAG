# src/observability/__init__.py

"""
Lightweight observability for VectorFlow-RAG.

Provides four metric primitive types — Counter, LabeledCounter, Gauge,
Histogram — plus a bounded RingBuffer for recent-trace retention. All
primitives are thread-safe via per-instance locks; contention is
negligible at the throughputs this service targets.

Telemetry is opt-in but on by default. The registry singleton is
process-local; for multi-worker deployments each worker has its own
registry, and Phase 9 deliberately stops short of cross-worker
aggregation (operators can scrape per-worker via the Prometheus
text exporter).

No external dependencies. The text exporter format is a strict subset
of the Prometheus exposition format so any Prom-compatible scraper
works without code changes here.
"""

from src.observability.primitives import (
    Counter,
    Gauge,
    Histogram,
    LabeledCounter,
    LabeledHistogram,
    RingBuffer,
)
from src.observability.prometheus_export import to_prometheus_text
from src.observability.registry import MetricsRegistry, get_metrics, reset_metrics

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "LabeledCounter",
    "LabeledHistogram",
    "MetricsRegistry",
    "RingBuffer",
    "get_metrics",
    "reset_metrics",
    "to_prometheus_text",
]
