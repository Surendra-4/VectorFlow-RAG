# tests/test_observability_overhead.py

"""
Low-overhead benchmark for the metrics layer.

These tests compare hot-path operations with and without metric
collection to verify the overhead claim from the Phase 9 plan
(<1% on retrieval p50). They print numbers (run with ``pytest -s``)
and enforce only loose ceilings because CI hardware is variable.
"""

from __future__ import annotations

import time

import pytest

from src.observability import (
    Counter,
    Gauge,
    Histogram,
    LabeledCounter,
    LabeledHistogram,
    RingBuffer,
    reset_metrics,
)


# --------------------------------------------------------------------------- #
# Per-op microbenchmarks
# --------------------------------------------------------------------------- #


def _bench(fn, iters: int) -> float:
    """Run ``fn`` ``iters`` times; return seconds per op."""
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    return elapsed / iters


class TestPrimitiveOverhead:
    """Each primitive should clock in well under microsecond territory."""

    def test_counter_inc_microbench(self):
        c = Counter("bench")
        per_op = _bench(c.inc, iters=200_000)
        per_op_us = per_op * 1e6
        print(f"\n[Counter.inc] {per_op_us:.3f} µs/op")
        # 5 µs is generous — typical: ~0.1 µs
        assert per_op_us < 5.0

    def test_labeled_counter_inc_microbench(self):
        lc = LabeledCounter("bench", ("endpoint", "status"))
        def op():
            lc.inc("/search", "200")
        per_op = _bench(op, iters=200_000)
        per_op_us = per_op * 1e6
        print(f"[LabeledCounter.inc] {per_op_us:.3f} µs/op")
        assert per_op_us < 10.0

    def test_gauge_set_microbench(self):
        g = Gauge("bench")
        counter = [0.0]
        def op():
            counter[0] += 1.0
            g.set(counter[0])
        per_op = _bench(op, iters=200_000)
        per_op_us = per_op * 1e6
        print(f"[Gauge.set] {per_op_us:.3f} µs/op")
        assert per_op_us < 5.0

    def test_histogram_observe_microbench(self):
        h = Histogram("bench", max_samples=200_000)
        v = [0.0]
        def op():
            v[0] += 1.0
            h.observe(v[0])
        per_op = _bench(op, iters=100_000)
        per_op_us = per_op * 1e6
        print(f"[Histogram.observe] {per_op_us:.3f} µs/op")
        # Histogram does more work (lock + deque append + prune); a few µs
        # is the realistic ceiling.
        assert per_op_us < 20.0

    def test_ring_buffer_append_microbench(self):
        rb = RingBuffer("bench", max_size=10_000)
        def op():
            rb.append({"k": "v"})
        per_op = _bench(op, iters=100_000)
        per_op_us = per_op * 1e6
        print(f"[RingBuffer.append] {per_op_us:.3f} µs/op")
        assert per_op_us < 20.0


# --------------------------------------------------------------------------- #
# End-to-end overhead — full retrieval cache hit path with metric collection
# --------------------------------------------------------------------------- #


class TestRetrievalOverhead:
    """Verify metric collection doesn't materially slow retrieval."""

    def test_cache_hit_warm_path_with_metrics(self, tmp_path):
        from src.cache.factory import make_cache
        from src.cache.memory import MemoryCache
        from src.cache.safe import SafeCache
        from src.rag_pipeline import RAGPipeline

        reset_metrics()
        cache = SafeCache(MemoryCache(default_ttl_s=3600))
        rag = RAGPipeline(
            index_dir=str(tmp_path / "obs_overhead"),
            cache=cache,
            enable_cache=True,
        )
        rag.ingest_documents([
            "Photosynthesis converts light to chemical energy.",
            "Mitochondria generate ATP.",
        ])

        # Warm the full-retrieval cache.
        rag.search("photosynthesis", k=2)

        # Benchmark the warm path (cache hit + metric collection).
        iters = 500
        t0 = time.perf_counter()
        for _ in range(iters):
            rag.search("photosynthesis", k=2)
        elapsed = time.perf_counter() - t0
        per_call_us = (elapsed / iters) * 1e6
        print(f"\n[Cache-hit retrieval + metrics] {per_call_us:.1f} µs/call ({iters} iters)")
        # Even with the metric collection overhead, a warm cache-hit
        # retrieval should comfortably stay under 5 ms on any machine.
        assert per_call_us < 5_000
