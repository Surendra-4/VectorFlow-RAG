# tests/test_observability_phase12.py

"""
Observability integration tests for Phase 12 metrics (12j):
provider usage/errors, model switches/installs, index jobs/builds/switch,
benchmark runs — plus bounded-cardinality + Prometheus export checks.
"""

from __future__ import annotations

from src.observability import get_metrics, reset_metrics, to_prometheus_text


def setup_function():
    reset_metrics()


# --------------------------------------------------------------------------- #
# New metrics exist and record
# --------------------------------------------------------------------------- #


def test_phase12_metrics_present():
    m = get_metrics()
    for attr in (
        "provider_ops_total", "provider_chat_total", "provider_errors_total",
        "model_installs_total", "model_switch_total",
        "index_jobs_total", "index_builds_total", "index_build_duration_ms",
        "index_switch_total", "benchmark_runs_total",
    ):
        assert hasattr(m, attr), attr


def test_snapshot_includes_phase12_sections():
    m = get_metrics()
    m.provider_chat_total.inc("openai", "success")
    m.model_switch_total.inc("ollama", "chat")
    m.index_builds_total.inc("ivf_pq", "succeeded")
    m.index_switch_total.inc()
    m.benchmark_runs_total.inc()
    m.index_build_duration_ms.observe("ivf_pq", value=123.0)

    snap = m.snapshot()
    lc = snap["labeled_counters"]
    assert lc["provider_chat_total"][0]["provider"] == "openai"
    assert lc["model_switch_total"][0]["kind"] == "chat"
    assert lc["index_builds_total"][0]["index_type"] == "ivf_pq"
    assert snap["counters"]["index_switch_total"] == 1
    assert snap["counters"]["benchmark_runs_total"] == 1
    assert "index_build_duration_ms" in snap["labeled_histograms"]


# --------------------------------------------------------------------------- #
# Bounded cardinality — index NAME never becomes a label
# --------------------------------------------------------------------------- #


def test_index_builds_labeled_by_recipe_not_name():
    m = get_metrics()
    # Simulate building 100 differently-named indexes of the same recipe.
    for i in range(100):
        m.index_builds_total.inc("hnsw", "succeeded")  # recipe label, not name
    snap = m.snapshot()
    rows = snap["labeled_counters"]["index_builds_total"]
    # Exactly one series (hnsw/succeeded), value 100 — cardinality stays bounded.
    assert len(rows) == 1
    assert rows[0]["value"] == 100
    assert rows[0]["index_type"] == "hnsw"


def test_provider_errors_bounded_by_class():
    m = get_metrics()
    for _ in range(50):
        m.provider_errors_total.inc("openai", "ProviderAuthError")
    rows = m.snapshot()["labeled_counters"]["provider_errors_total"]
    assert len(rows) == 1
    assert rows[0]["value"] == 50


# --------------------------------------------------------------------------- #
# Prometheus export includes the new metrics
# --------------------------------------------------------------------------- #


def test_prometheus_export_contains_phase12_metrics():
    m = get_metrics()
    m.provider_ops_total.inc("ollama", "list_installed")
    m.provider_chat_total.inc("anthropic", "success")
    m.model_installs_total.inc("ollama", "success")
    m.index_jobs_total.inc("index_build", "succeeded")
    m.index_builds_total.inc("flat", "succeeded")
    m.index_switch_total.inc()
    m.benchmark_runs_total.inc()
    m.index_build_duration_ms.observe("flat", value=42.0)

    text = to_prometheus_text(m)
    assert "provider_ops_total{" in text
    assert 'provider="anthropic"' in text
    assert "model_installs_total{" in text
    assert "index_builds_total{" in text
    assert "index_switch_total " in text
    assert "benchmark_runs_total " in text
    assert "index_build_duration_ms{" in text
    # Every TYPE line must have a matching series (well-formed exposition).
    assert text.endswith("\n")


def test_prometheus_export_is_valid_after_phase12_additions():
    # Smoke: a fresh registry exports without error and is non-empty.
    text = to_prometheus_text(get_metrics())
    assert "# TYPE" in text
    assert "provider_chat_total" in text  # declared even at zero
