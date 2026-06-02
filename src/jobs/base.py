# src/jobs/base.py

"""
Background job primitives (Phase 12h).

Long-running work — chiefly FAISS index builds/training — must not block HTTP
workers. A :class:`Job` is a unit of such work executed on a thread pool, with:

* **progress tracking** — ``0..100`` plus a human message,
* **cooperative cancellation** — the work function polls ``ctx.check_cancel()``,
* **replayable streaming** — a late SSE subscriber still sees the full history
  then live updates until the job reaches a terminal state,
* **terminal results** — a JSON-serializable result or a structured error.

The streaming model uses a per-job event list guarded by a Condition. Yielding
happens *outside* the lock (we copy new events under the lock, release, then
yield), so a slow SSE client can never stall the worker thread.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class JobCancelled(Exception):
    """Raised inside a job function when cancellation is requested."""


# Cap on retained progress events per job — bounds memory for chatty jobs.
_MAX_EVENTS = 2000


class Job:
    """A single background unit of work + its observable state."""

    def __init__(self, job_type: str, *, label: str = "", job_id: Optional[str] = None):
        self.id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        self.type = job_type
        self.label = label or job_type
        self.status = JobStatus.PENDING
        self.progress: float = 0.0
        self.message: str = ""
        self.result: Any = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        # Concurrency machinery (never serialized).
        self._cancel_event = threading.Event()
        self._cond = threading.Condition()
        self._events: List[Dict[str, Any]] = []
        self._terminal = False

    # ------------------------------------------------------------------ #
    # State transitions (called by the registry worker)
    # ------------------------------------------------------------------ #

    def _mark_running(self) -> None:
        with self._cond:
            self.status = JobStatus.RUNNING
            self.started_at = time.time()
        self._emit(self.progress, "started")

    def _emit(self, percent: float, message: str = "") -> None:
        """Append a progress event and wake any streamers."""
        with self._cond:
            self.progress = max(0.0, min(100.0, float(percent)))
            if message:
                self.message = message
            event = {
                "job_id": self.id,
                "status": self.status.value,
                "progress": self.progress,
                "message": self.message,
                "ts": time.time(),
            }
            self._events.append(event)
            if len(self._events) > _MAX_EVENTS:
                # Drop oldest mid-stream events; never drop the first (start).
                del self._events[1:len(self._events) - _MAX_EVENTS + 1]
            self._cond.notify_all()

    def _finalize(self, status: JobStatus, *, result: Any = None, error: Optional[str] = None) -> None:
        with self._cond:
            self.status = status
            self.result = result
            self.error = error
            self.finished_at = time.time()
            if status == JobStatus.SUCCEEDED:
                self.progress = 100.0
            self._terminal = True
            self._events.append({
                "job_id": self.id,
                "status": self.status.value,
                "progress": self.progress,
                "message": self.message,
                "result": result,
                "error": error,
                "ts": time.time(),
                "terminal": True,
            })
            self._cond.notify_all()

    # ------------------------------------------------------------------ #
    # Cancellation
    # ------------------------------------------------------------------ #

    def request_cancel(self) -> None:
        self._cancel_event.set()
        # Nudge streamers so a cancel is observed promptly even pre-start.
        with self._cond:
            self._cond.notify_all()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #

    def stream(self, heartbeat_s: float = 15.0) -> Iterator[Dict[str, Any]]:
        """Yield progress events, replaying history first, until terminal.

        Safe for a late subscriber: all buffered events replay before live
        ones. Yields outside the lock so a slow consumer never blocks workers.
        """
        idx = 0
        while True:
            with self._cond:
                while idx >= len(self._events) and not self._terminal:
                    self._cond.wait(timeout=heartbeat_s)
                new = self._events[idx:]
                idx += len(new)
                done = self._terminal and idx >= len(self._events)
            for event in new:
                yield event
            if done:
                break

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self, include_history: bool = False) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_requested,
        }
        if include_history:
            with self._cond:
                d["history"] = list(self._events)
        return d


@dataclass
class JobContext:
    """Handle passed to a job function for progress + cancellation.

    A job function has the signature ``fn(ctx: JobContext, **kwargs)`` and
    should call :meth:`set_progress` as it advances and :meth:`check_cancel`
    at safe checkpoints in long loops.
    """

    job: Job

    def set_progress(self, percent: float, message: str = "") -> None:
        self.job._emit(percent, message)

    def is_cancelled(self) -> bool:
        return self.job.cancel_requested

    def check_cancel(self) -> None:
        if self.job.cancel_requested:
            raise JobCancelled()
