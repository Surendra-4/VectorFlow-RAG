# src/observability/prometheus_export.py

"""
Prometheus text-format exporter for the metrics registry.

Format reference: https://prometheus.io/docs/instrumenting/exposition_formats/

We support the subset our metric primitives need:

* counters  → ``# TYPE name counter`` + ``name{labels} value``
* gauges    → ``# TYPE name gauge``   + ``name value``
* histograms → ``# TYPE name summary`` (we report quantiles, not buckets) +
              ``name{quantile="0.5"} v`` lines + ``_count`` + ``_sum`` (when known)

No external dep — this is a deliberate ~120-line text builder so the
service can be scraped by any Prom-compatible collector without pulling
in the official client library.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from src.observability.registry import MetricsRegistry


_LABEL_VALUE_ESCAPE = str.maketrans({
    "\\": r"\\",
    "\"": r"\"",
    "\n": r"\n",
})


def _escape_label_value(v: str) -> str:
    return str(v).translate(_LABEL_VALUE_ESCAPE)


def _format_labels(label_names: Iterable[str], label_values: Iterable[str]) -> str:
    pairs = [
        f'{name}="{_escape_label_value(value)}"'
        for name, value in zip(label_names, label_values)
    ]
    if not pairs:
        return ""
    return "{" + ",".join(pairs) + "}"


def to_prometheus_text(reg: MetricsRegistry) -> str:
    """Render the registry as Prometheus exposition text."""
    lines: List[str] = []

    # ---- Plain counters ---------------------------------------------------- #
    for c in (
        reg.retrievals_total,
        reg.reranker_used_total,
        reg.cache_hits_total,
        reg.chunks_ingested_total,
        reg.ingest_failures_total,
        reg.stream_sessions_total,
        reg.index_switch_total,
        reg.benchmark_runs_total,
    ):
        if c.description:
            lines.append(f"# HELP {c.name} {c.description}")
        lines.append(f"# TYPE {c.name} counter")
        lines.append(f"{c.name} {c.value}")

    # ---- Labeled counters ------------------------------------------------- #
    for lc in (
        reg.requests_total,
        reg.expansion_strategy_usage_total,
        reg.cache_ops_total,
        reg.ingestions_total,
        # Phase 12 — model + index lifecycle (all bounded cardinality).
        reg.provider_ops_total,
        reg.provider_chat_total,
        reg.provider_errors_total,
        reg.model_installs_total,
        reg.model_switch_total,
        reg.index_jobs_total,
        reg.index_builds_total,
    ):
        if lc.description:
            lines.append(f"# HELP {lc.name} {lc.description}")
        lines.append(f"# TYPE {lc.name} counter")
        for labels, value in lc.items():
            lines.append(f"{lc.name}{_format_labels(lc.label_names, labels)} {value}")

    # ---- Gauges ----------------------------------------------------------- #
    for g in (reg.active_streams,):
        if g.description:
            lines.append(f"# HELP {g.name} {g.description}")
        lines.append(f"# TYPE {g.name} gauge")
        lines.append(f"{g.name} {g.value}")

    # ---- Histograms (rendered as summaries) ------------------------------- #
    for h in (reg.retrieval_latency_ms, reg.ingest_latency_ms, reg.stream_duration_ms):
        if h.description:
            lines.append(f"# HELP {h.name} {h.description}")
        lines.append(f"# TYPE {h.name} summary")
        snap = h.snapshot()
        if snap["count"] == 0:
            lines.append(f"{h.name}_count 0")
            continue
        for q, key in (("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")):
            lines.append(f'{h.name}{{quantile="{q}"}} {snap[key]}')
        lines.append(f"{h.name}_count {snap['count']}")
        if snap["mean"] is not None:
            lines.append(f"{h.name}_sum {snap['mean'] * snap['count']}")

    # ---- Labeled histograms (per-label summaries) ------------------------- #
    for lh in (reg.request_latency_ms, reg.retrieval_stage_latency_ms, reg.index_build_duration_ms):
        if lh.description:
            lines.append(f"# HELP {lh.name} {lh.description}")
        lines.append(f"# TYPE {lh.name} summary")
        for labels, snap in lh.items():
            label_str_base = _format_labels(lh.label_names, labels).rstrip("}")
            sep = "," if label_str_base else "{"
            if snap["count"] == 0:
                lines.append(f"{lh.name}_count{_format_labels(lh.label_names, labels)} 0")
                continue
            for q, key in (("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")):
                # build "{endpoint=..., quantile=...}"
                if label_str_base.startswith("{"):
                    quantile_part = f'{label_str_base[1:]}{sep}quantile="{q}"' + "}"
                    suffix = "{" + quantile_part if False else quantile_part
                else:
                    suffix = f'{{quantile="{q}"}}'
                # Compose cleanly using a helper to avoid the brittle string surgery.
                merged = _merge_label_with_quantile(lh.label_names, labels, q)
                lines.append(f"{lh.name}{merged} {snap[key]}")
            lines.append(
                f"{lh.name}_count{_format_labels(lh.label_names, labels)} {snap['count']}"
            )

    return "\n".join(lines) + "\n"


def _merge_label_with_quantile(
    label_names: Iterable[str], label_values: Iterable[str], quantile: str
) -> str:
    pairs = [
        f'{name}="{_escape_label_value(value)}"'
        for name, value in zip(label_names, label_values)
    ]
    pairs.append(f'quantile="{quantile}"')
    return "{" + ",".join(pairs) + "}"
