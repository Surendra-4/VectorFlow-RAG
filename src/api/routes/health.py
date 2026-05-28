# src/api/routes/health.py

"""Liveness probe — never touches the pipeline."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_request_id, get_started_at
from src.api.schemas import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    request_id: str = Depends(get_request_id),
    started_at: float = Depends(get_started_at),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_version=getattr(request.app.state, "app_version", "unknown"),
        uptime_s=time.monotonic() - started_at if started_at else 0.0,
        request_id=request_id,
    )
