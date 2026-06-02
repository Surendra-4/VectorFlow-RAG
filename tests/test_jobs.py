# tests/test_jobs.py

"""
Tests for the background job system (Phase 12h): registry, context, progress,
cancellation, replayable streaming, history, and concurrency.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.indexing import IndexManager, IndexProfile, IndexRegistry
from src.jobs import (
    JobRegistry,
    JobStatus,
    build_index_job,
)
from src.jobs.base import Job, JobContext


@pytest.fixture
def registry():
    reg = JobRegistry(max_workers=4, max_retained=50)
    yield reg
    reg.shutdown(wait=False)


def _wait_terminal(job: Job, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.status.is_terminal:
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------- #
# Happy path + progress
# --------------------------------------------------------------------------- #


def test_job_runs_to_success_with_progress(registry):
    seen = []

    def work(ctx: JobContext, n=5):
        for i in range(n):
            ctx.set_progress((i + 1) / n * 100, f"step {i+1}")
            seen.append(i)
        return {"done": n}

    job = registry.submit("demo", work, n=5)
    assert _wait_terminal(job)
    assert job.status == JobStatus.SUCCEEDED
    assert job.result == {"done": 5}
    assert job.progress == 100.0
    assert seen == [0, 1, 2, 3, 4]
    assert job.started_at is not None and job.finished_at is not None


def test_job_failure_is_captured(registry):
    def boom(ctx: JobContext):
        raise RuntimeError("kaboom")

    job = registry.submit("demo", boom)
    assert _wait_terminal(job)
    assert job.status == JobStatus.FAILED
    assert "RuntimeError: kaboom" in job.error
    # A failed job must not take down the pool — a subsequent job still runs.
    ok = registry.submit("demo", lambda ctx: 42)
    assert _wait_terminal(ok)
    assert ok.result == 42


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


def test_cooperative_cancellation(registry):
    started = threading.Event()

    def long_work(ctx: JobContext):
        started.set()
        for _ in range(1000):
            ctx.check_cancel()
            time.sleep(0.005)
        return "should not reach"

    job = registry.submit("demo", long_work)
    assert started.wait(2.0)
    assert registry.cancel(job.id) is True
    assert _wait_terminal(job)
    assert job.status == JobStatus.CANCELLED
    assert job.result is None


def test_cancel_unknown_or_terminal_returns_false(registry):
    assert registry.cancel("nope") is False
    job = registry.submit("demo", lambda ctx: 1)
    assert _wait_terminal(job)
    assert registry.cancel(job.id) is False  # already terminal


# --------------------------------------------------------------------------- #
# Streaming (replayable)
# --------------------------------------------------------------------------- #


def test_stream_replays_then_completes(registry):
    def work(ctx: JobContext):
        for i in range(3):
            ctx.set_progress((i + 1) * 30, f"s{i}")
            time.sleep(0.02)
        return "ok"

    job = registry.submit("demo", work)
    events = list(job.stream(heartbeat_s=1.0))  # drains until terminal
    assert len(events) >= 2
    assert events[-1].get("terminal") is True
    assert events[-1]["status"] == "succeeded"
    assert events[-1]["result"] == "ok"


def test_late_subscriber_still_sees_terminal(registry):
    job = registry.submit("demo", lambda ctx: "fast")
    assert _wait_terminal(job)
    # Subscribe AFTER completion — must still get the terminal event.
    events = list(job.stream(heartbeat_s=1.0))
    assert events[-1].get("terminal") is True
    assert events[-1]["result"] == "fast"


# --------------------------------------------------------------------------- #
# Listing + history + retention
# --------------------------------------------------------------------------- #


def test_list_jobs_newest_first_and_filtered(registry):
    a = registry.submit("typeA", lambda ctx: 1)
    b = registry.submit("typeB", lambda ctx: 2)
    _wait_terminal(a)
    _wait_terminal(b)
    all_jobs = registry.list_jobs()
    assert all_jobs[0].created_at >= all_jobs[-1].created_at
    only_a = registry.list_jobs(job_type="typeA")
    assert [j.type for j in only_a] == ["typeA"]


def test_history_included_on_request(registry):
    def work(ctx: JobContext):
        ctx.set_progress(50, "half")
        return "done"

    job = registry.submit("demo", work)
    assert _wait_terminal(job)
    d = job.to_dict(include_history=True)
    assert "history" in d
    assert any(e["message"] == "half" for e in d["history"])
    # Without the flag, no history key (keeps list payloads small).
    assert "history" not in job.to_dict()


def test_retention_evicts_oldest_terminal(registry):
    small = JobRegistry(max_workers=2, max_retained=3)
    try:
        jobs = [small.submit("demo", lambda ctx: 1) for _ in range(6)]
        for j in jobs:
            _wait_terminal(j)
        # allow eviction to settle
        remaining = small.list_jobs()
        assert len(remaining) <= 3
    finally:
        small.shutdown()


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_jobs_all_complete(registry):
    results = {}

    def work(ctx: JobContext, key=0):
        time.sleep(0.05)
        results[key] = key * 2
        return key

    jobs = [registry.submit("demo", work, key=i) for i in range(8)]
    for j in jobs:
        assert _wait_terminal(j, timeout=10)
        assert j.status == JobStatus.SUCCEEDED
    assert results == {i: i * 2 for i in range(8)}


def test_metrics_hook_fires_on_terminal(registry):
    fired = []
    registry.set_metrics_hook(lambda job: fired.append((job.type, job.status.value)))
    job = registry.submit("demo", lambda ctx: 1)
    assert _wait_terminal(job)
    time.sleep(0.05)
    assert ("demo", "succeeded") in fired


# --------------------------------------------------------------------------- #
# Index build job
# --------------------------------------------------------------------------- #


@pytest.fixture
def index_manager(temp_dir):
    reg = IndexRegistry(path=temp_dir / "reg.json")
    return IndexManager(registry=reg, indices_root=temp_dir / "named")


def test_build_index_job_with_precomputed_embeddings(registry, index_manager):
    rng = np.random.default_rng(0)
    v = rng.standard_normal((20, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    texts = [f"d{i}" for i in range(20)]
    profile = IndexProfile(name="job_built", backend="faiss", index_type="flat",
                           embedding_model="synthetic", vector_dimension=8)

    job = registry.submit(
        "index_build", build_index_job,
        manager=index_manager, profile=profile, texts=texts,
        embeddings=v, ids=[f"c{i}" for i in range(20)], make_active=True,
    )
    assert _wait_terminal(job)
    assert job.status == JobStatus.SUCCEEDED
    assert job.result["index_name"] == "job_built"
    assert job.result["num_vectors"] == 20
    assert index_manager.registry.active_name == "job_built"


def test_build_index_job_uses_embedder_in_batches(registry, index_manager):
    class FakeEmbedder:
        def encode(self, batch, show_progress=False, input_type=None):
            # deterministic 4-d vectors
            return np.array([[float(len(t)), 1.0, 2.0, 3.0] for t in batch], dtype=np.float32)

    texts = [f"chunk-{i}" for i in range(10)]
    profile = IndexProfile(name="emb_built", backend="faiss", index_type="flat",
                           embedding_model="fake")
    job = registry.submit(
        "index_build", build_index_job,
        manager=index_manager, profile=profile, texts=texts,
        embedder=FakeEmbedder(), batch_size=4,
    )
    assert _wait_terminal(job)
    assert job.status == JobStatus.SUCCEEDED
    assert job.result["num_vectors"] == 10
    assert job.result["vector_dimension"] == 4


def test_build_index_job_empty_corpus_fails(registry, index_manager):
    profile = IndexProfile(name="empty", backend="faiss", index_type="flat",
                           embedding_model="x")
    job = registry.submit("index_build", build_index_job,
                          manager=index_manager, profile=profile, texts=[], embeddings=[])
    assert _wait_terminal(job)
    assert job.status == JobStatus.FAILED
