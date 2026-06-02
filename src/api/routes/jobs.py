# src/api/routes/jobs.py

"""
Background jobs API (Phase 12h).

* ``GET    /jobs``              — list jobs (newest first; optional ?type=)
* ``GET    /jobs/{id}``         — one job's full state (+ history)
* ``GET    /jobs/{id}/stream``  — SSE progress until terminal
* ``POST   /jobs/{id}/cancel``  — request cooperative cancellation

Streaming replays buffered progress first, so a late subscriber still sees the
whole run, then live events until the job finishes.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_job_registry, get_request_id
from src.api.schemas import JobListResponse, JobResponse

router = APIRouter(tags=["jobs"], prefix="/jobs")


@router.get("", response_model=JobListResponse)
def list_jobs(
    type: str | None = None,
    limit: int = 50,
    registry=Depends(get_job_registry),
    request_id: str = Depends(get_request_id),
) -> JobListResponse:
    jobs = registry.list_jobs(limit=limit, job_type=type)
    return JobListResponse(jobs=[j.to_dict() for j in jobs], request_id=request_id)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    registry=Depends(get_job_registry),
    request_id: str = Depends(get_request_id),
) -> JobResponse:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
    return JobResponse(job=job.to_dict(include_history=True), request_id=request_id)


@router.get("/{job_id}/stream")
def stream_job(
    job_id: str,
    registry=Depends(get_job_registry),
    request_id: str = Depends(get_request_id),
):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

    def generate():
        for evt in job.stream():
            kind = "done" if evt.get("terminal") else "progress"
            yield _sse(kind, evt)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
    )


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(
    job_id: str,
    registry=Depends(get_job_registry),
    request_id: str = Depends(get_request_id),
) -> JobResponse:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
    registry.cancel(job_id)
    return JobResponse(job=job.to_dict(), request_id=request_id)
