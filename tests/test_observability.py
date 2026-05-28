# tests/test_observability.py

"""
Unit tests for observability primitives, registry, and Prometheus exporter.
"""

from __future__ import annotations

import concurrent.futures
import re
import threading
import time

import pytest

from src.observability import (
    Counter,
    Gauge,
    Histogram,
    LabeledCounter,
    LabeledHistogram,
    MetricsRegistry,
    RingBuffer,
    get_metrics,
    reset_metrics,
    to_prometheus_text,
)


# --------------------------------------------------------------------------- #
# Counter
# --------------------------------------------------------------------------- #


class TestCounter:
    def test_starts_at_zero(self):
        assert Counter("x").value == 0

    def test_inc_default_step(self):
        c = Counter("x")
        c.inc()
        c.inc()
        assert c.value == 2

    def test_inc_with_step(self):
        c = Counter("x")
        c.inc(n=5)
        assert c.value == 5

    def test_negative_inc_rejected(self):
        with pytest.raises(ValueError):
            Counter("x").inc(n=-1)

    def test_concurrent_increment_correct(self):
        c = Counter("x")

        def worker():
            for _ in range(1000):
                c.inc()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda _: worker(), range(8)))

        assert c.value == 8 * 1000


# --------------------------------------------------------------------------- #
# LabeledCounter
# --------------------------------------------------------------------------- #


class TestLabeledCounter:
    def test_inc_with_correct_label_count(self):
        lc = LabeledCounter("x", ("endpoint", "status"))
        lc.inc("/a", "200")
        lc.inc("/a", "200")
        lc.inc("/b", "200")
        assert dict(lc.items()) == {
            ("/a", "200"): 2,
            ("/b", "200"): 1,
        }

    def test_total(self):
        lc = LabeledCounter("x", ("k",))
        lc.inc("a", n=3)
        lc.inc("b", n=2)
        assert lc.total() == 5

    def test_wrong_label_count_raises(self):
        lc = LabeledCounter("x", ("endpoint", "status"))
        with pytest.raises(ValueError):
            lc.inc("/a")  # missing status
        with pytest.raises(ValueError):
            lc.inc("/a", "200", "extra")

    def test_concurrent_inc_per_label(self):
        lc = LabeledCounter("x", ("k",))

        def worker(name: str):
            for _ in range(500):
                lc.inc(name)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(worker, f"label{i}") for i in range(4)]
            for f in futures:
                f.result()

        items = dict(lc.items())
        for i in range(4):
            assert items[(f"label{i}",)] == 500


# --------------------------------------------------------------------------- #
# Gauge
# --------------------------------------------------------------------------- #


class TestGauge:
    def test_set_inc_dec(self):
        g = Gauge("x")
        g.set(10)
        assert g.value == 10.0
        g.inc(2.5)
        assert g.value == 12.5
        g.dec()
        assert g.value == 11.5

    def test_concurrent_inc_dec(self):
        g = Gauge("x")

        def inc_worker():
            for _ in range(1000):
                g.inc()

        def dec_worker():
            for _ in range(1000):
                g.dec()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            ex.map(lambda _: inc_worker(), range(4))
            ex.map(lambda _: dec_worker(), range(4))

        # 4*1000 ups, 4*1000 downs → 0
        # Need to wait for the threads to finish before reading
        # (ThreadPoolExecutor.__exit__ joins).
        assert g.value == 0.0


# --------------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------------- #


class TestHistogram:
    def test_empty_snapshot(self):
        snap = Histogram("x").snapshot()
        assert snap["count"] == 0
        for k in ("p50", "p95", "p99", "min", "max", "mean"):
            assert snap[k] is None

    def test_simple_percentiles(self):
        h = Histogram("x")
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            h.observe(v)
        snap = h.snapshot()
        assert snap["count"] == 10
        assert snap["min"] == 10
        assert snap["max"] == 100
        assert snap["p50"] == 60  # nearest-rank: idx = int(0.5*10) = 5 → sorted[5]=60
        assert snap["mean"] == 55.0

    def test_rolling_window_drops_old_samples(self):
        h = Histogram("x", window_s=0.2)
        h.observe(1.0)
        h.observe(2.0)
        time.sleep(0.3)
        # New observe triggers prune of the old samples.
        h.observe(3.0)
        snap = h.snapshot()
        assert snap["count"] == 1
        assert snap["min"] == 3.0

    def test_max_samples_cap(self):
        h = Histogram("x", window_s=3600, max_samples=10)
        for v in range(100):
            h.observe(float(v))
        snap = h.snapshot()
        assert snap["count"] == 10

    def test_invalid_args_rejected(self):
        with pytest.raises(ValueError):
            Histogram("x", window_s=0)
        with pytest.raises(ValueError):
            Histogram("x", max_samples=0)

    def test_concurrent_observe(self):
        h = Histogram("x", max_samples=10000)

        def worker():
            for v in range(500):
                h.observe(float(v))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(worker) for _ in range(4)]
            for f in futs:
                f.result()

        snap = h.snapshot()
        assert snap["count"] == 4 * 500


# --------------------------------------------------------------------------- #
# LabeledHistogram
# --------------------------------------------------------------------------- #


class TestLabeledHistogram:
    def test_observe_distinct_labels(self):
        lh = LabeledHistogram("x", ("stage",))
        lh.observe("retrieve", value=10)
        lh.observe("retrieve", value=20)
        lh.observe("rerank", value=100)
        items = dict(lh.items())
        assert items[("retrieve",)]["count"] == 2
        assert items[("rerank",)]["count"] == 1

    def test_wrong_label_count_raises(self):
        lh = LabeledHistogram("x", ("stage",))
        with pytest.raises(ValueError):
            lh.observe(value=1.0)


# --------------------------------------------------------------------------- #
# RingBuffer
# --------------------------------------------------------------------------- #


class TestRingBuffer:
    def test_bounded_size(self):
        rb = RingBuffer("x", max_size=3)
        for v in range(10):
            rb.append(v)
        assert len(rb) == 3
        assert rb.snapshot() == [7, 8, 9]

    def test_snapshot_limit(self):
        rb = RingBuffer("x", max_size=10)
        for v in range(10):
            rb.append(v)
        assert rb.snapshot(limit=3) == [7, 8, 9]

    def test_clear(self):
        rb = RingBuffer("x")
        rb.append("a")
        rb.clear()
        assert len(rb) == 0

    def test_invalid_size_rejected(self):
        with pytest.raises(ValueError):
            RingBuffer("x", max_size=0)

    def test_concurrent_append(self):
        rb = RingBuffer("x", max_size=10000)

        def worker(tid):
            for i in range(500):
                rb.append((tid, i))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(worker, t) for t in range(4)]
            for f in futs:
                f.result()

        assert len(rb) == 4 * 500


# --------------------------------------------------------------------------- #
# MetricsRegistry
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_singleton(self):
        reset_metrics()
        a = get_metrics()
        b = get_metrics()
        assert a is b

    def test_reset_creates_fresh(self):
        a = get_metrics()
        a.retrievals_total.inc()
        reset_metrics()
        b = get_metrics()
        assert b is not a
        assert b.retrievals_total.value == 0

    def test_snapshot_shape_stable_empty(self):
        reset_metrics()
        snap = get_metrics().snapshot()
        # Required top-level keys.
        for key in (
            "uptime_s", "counters", "gauges",
            "labeled_counters", "histograms",
            "labeled_histograms", "ring_buffer_sizes",
        ):
            assert key in snap, f"missing key: {key}"

    def test_snapshot_after_activity(self):
        reset_metrics()
        m = get_metrics()
        m.retrievals_total.inc()
        m.retrieval_latency_ms.observe(12.5)
        m.requests_total.inc("/health", "200")
        m.active_streams.inc()
        m.recent_traces.append({"q": "test"})

        snap = m.snapshot()
        assert snap["counters"]["retrievals_total"] == 1
        assert snap["histograms"]["retrieval_latency_ms"]["count"] == 1
        assert len(snap["labeled_counters"]["requests_total"]) == 1
        assert snap["gauges"]["active_streams"] == 1.0
        assert snap["ring_buffer_sizes"]["recent_traces"] == 1


# --------------------------------------------------------------------------- #
# Prometheus exporter
# --------------------------------------------------------------------------- #


class TestPrometheusExport:
    def test_renders_counters(self):
        reset_metrics()
        m = get_metrics()
        m.retrievals_total.inc(n=3)
        text = to_prometheus_text(m)
        assert "# TYPE retrievals_total counter" in text
        assert "retrievals_total 3" in text

    def test_renders_labeled_counters(self):
        reset_metrics()
        m = get_metrics()
        m.requests_total.inc("/api/v1/search", "200", n=2)
        text = to_prometheus_text(m)
        assert 'requests_total{endpoint="/api/v1/search",status_code="200"} 2' in text

    def test_renders_gauges(self):
        reset_metrics()
        m = get_metrics()
        m.active_streams.set(4)
        text = to_prometheus_text(m)
        assert "# TYPE active_streams gauge" in text
        assert "active_streams 4" in text

    def test_renders_histograms_as_summary(self):
        reset_metrics()
        m = get_metrics()
        for v in [10, 20, 30]:
            m.retrieval_latency_ms.observe(v)
        text = to_prometheus_text(m)
        assert "# TYPE retrieval_latency_ms summary" in text
        assert 'retrieval_latency_ms{quantile="0.5"}' in text
        assert "retrieval_latency_ms_count 3" in text

    def test_renders_labeled_histograms(self):
        reset_metrics()
        m = get_metrics()
        m.request_latency_ms.observe("/search", value=15.0)
        m.request_latency_ms.observe("/search", value=22.0)
        text = to_prometheus_text(m)
        assert (
            'request_latency_ms{endpoint="/search",quantile="0.5"}' in text
        ), text

    def test_label_value_escaping(self):
        reset_metrics()
        m = get_metrics()
        m.requests_total.inc('/api/with"quote', "200")
        text = to_prometheus_text(m)
        assert r'/api/with\"quote' in text

    def test_text_is_newline_terminated(self):
        reset_metrics()
        text = to_prometheus_text(get_metrics())
        assert text.endswith("\n")
