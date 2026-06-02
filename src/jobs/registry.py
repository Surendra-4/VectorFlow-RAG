# src/jobs/registry.py

"""
JobRegistry (Phase 12h) — submit, track, stream, cancel, and retain jobs.

Backed by a small ``ThreadPoolExecutor`` so HTTP request handlers return
immediately while FAISS builds/training run off the request thread. Finished
jobs are retained (bounded) so the UI can show a history and late SSE
subscribers can still fetch a terminal result.

The registry never lets a job exception escape into the worker pool: the
wrapper records it on the job as a structured error and finalizes cleanly.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from src.jobs.base import Job, JobCancelled, JobContext, JobStatus
from src.logging_setup import get_logger

logger = get_logger(__name__)


class JobRegistry:
    """Process-local registry of background jobs."""

    def __init__(self, max_workers: int = 2, max_retained: int = 200):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="vfr-job"
        )
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.RLock()
        self._max_retained = max_retained
        self._metrics_hook: Optional[Callable[[Job], None]] = None

    def set_metrics_hook(self, hook: Callable[[Job], None]) -> None:
        """Register a callback invoked when a job reaches a terminal state."""
        self._metrics_hook = hook

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #

    def submit(self, job_type: str, fn: Callable, *, label: str = "", **kwargs) -> Job:
        """Schedule ``fn(ctx, **kwargs)`` on the pool and return its Job.

        ``fn`` receives a :class:`JobContext` as its first argument and may
        return any JSON-serializable result, which becomes ``job.result``.
        """
        job = Job(job_type, label=label)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_if_needed()
        self._executor.submit(self._run, job, fn, kwargs)
        logger.info("Submitted job %s (type=%s)", job.id, job_type)
        return job

    def _run(self, job: Job, fn: Callable, kwargs: dict) -> None:
        # Honor a cancel requested before the worker picked the job up.
        if job.cancel_requested:
            job._finalize(JobStatus.CANCELLED)
            self._fire_metrics(job)
            return
        job._mark_running()
        ctx = JobContext(job)
        try:
            result = fn(ctx, **kwargs)
            job._finalize(JobStatus.SUCCEEDED, result=result)
        except JobCancelled:
            job._finalize(JobStatus.CANCELLED)
            logger.info("Job %s cancelled", job.id)
        except Exception as exc:  # noqa: BLE001 - jobs must never crash the pool
            job._finalize(JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            logger.exception("Job %s failed", job.id)
        finally:
            self._fire_metrics(job)

    def _fire_metrics(self, job: Job) -> None:
        if self._metrics_hook is not None:
            try:
                self._metrics_hook(job)
            except Exception:  # pragma: no cover - metrics must never break jobs
                logger.debug("job metrics hook raised; ignored")

    # ------------------------------------------------------------------ #
    # Queries / control
    # ------------------------------------------------------------------ #

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: Optional[int] = None, job_type: Optional[str] = None) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if job_type is not None:
            jobs = [j for j in jobs if j.type == job_type]
        # newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        if limit is not None:
            jobs = jobs[:limit]
        return jobs

    def cancel(self, job_id: str) -> bool:
        """Request cancellation. Returns False if unknown or already terminal."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status.is_terminal:
            return False
        job.request_cancel()
        return True

    # ------------------------------------------------------------------ #
    # Retention
    # ------------------------------------------------------------------ #

    def _evict_if_needed(self) -> None:
        """Drop oldest *terminal* jobs once over the retention cap."""
        while len(self._jobs) > self._max_retained:
            # Find the oldest terminal job to evict; never evict a live one.
            victim = None
            for jid, job in self._jobs.items():
                if job.status.is_terminal:
                    victim = jid
                    break
            if victim is None:
                break  # all live — let the pool drain first
            del self._jobs[victim]

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


# --------------------------------------------------------------------------- #
# Process-wide singleton (the API holds its own on app.state, but a default is
# handy for scripts/tests).
# --------------------------------------------------------------------------- #

_REGISTRY: Optional[JobRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_job_registry() -> JobRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = JobRegistry()
    return _REGISTRY


def reset_job_registry() -> None:
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is not None:
            _REGISTRY.shutdown(wait=False)
        _REGISTRY = None
