# src/jobs/__init__.py

"""
Background job system for VectorFlow-RAG (Phase 12h).

Runs long operations (FAISS index builds/training) off the HTTP request thread
with progress tracking, cooperative cancellation, replayable SSE streaming, and
bounded job history.

Public surface:

* :class:`Job` / :class:`JobContext` / :class:`JobStatus` / :class:`JobCancelled`
* :class:`JobRegistry` / :func:`get_job_registry` / :func:`reset_job_registry`
* :func:`build_index_job`
"""

from src.jobs.base import Job, JobCancelled, JobContext, JobStatus
from src.jobs.index_jobs import benchmark_recipes_job, build_index_job
from src.jobs.registry import JobRegistry, get_job_registry, reset_job_registry

__all__ = [
    "Job",
    "JobCancelled",
    "JobContext",
    "JobRegistry",
    "JobStatus",
    "benchmark_recipes_job",
    "build_index_job",
    "get_job_registry",
    "reset_job_registry",
]
