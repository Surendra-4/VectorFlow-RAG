# src/api/routes/config.py

"""
Runtime configuration API (Phase 12c/d).

Exposes the :class:`RuntimeConfigStore` over HTTP, keeping the live-vs-index
separation the store enforces:

* ``GET   /config/runtime``        — full snapshot (live + staged + active)
* ``PATCH /config/runtime/live``   — update live-query settings (applied now)
* ``GET   /config/runtime/index``  — staged index settings + rebuild flag
* ``PATCH /config/runtime/index``  — stage index-construction settings (NO rebuild)
* ``POST  /config/runtime/index/reset`` — discard staged → back to active

Live updates are applied to the running pipeline immediately. Index updates
are staged only; a rebuild happens exclusively through the index API (12e).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import (
    get_ingest_lock,
    get_pipeline,
    get_request_id,
    get_runtime_config,
)
from src.api.schemas import (
    LiveSettingsResponse,
    RuntimeConfigResponse,
    StagedIndexResponse,
)
from src.logging_setup import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["config"], prefix="/config")


@router.get("/runtime", response_model=RuntimeConfigResponse)
def get_runtime(
    runtime=Depends(get_runtime_config),
    request_id: str = Depends(get_request_id),
) -> RuntimeConfigResponse:
    snap = runtime.snapshot()
    return RuntimeConfigResponse(
        version=snap["version"],
        live=snap["live"],
        staged_index=snap["staged_index"],
        active_index=snap["active_index"],
        request_id=request_id,
    )


@router.patch("/runtime/live", response_model=LiveSettingsResponse)
async def patch_live(
    request: Request,
    pipeline=Depends(get_pipeline),
    runtime=Depends(get_runtime_config),
    lock=Depends(get_ingest_lock),
    request_id: str = Depends(get_request_id),
) -> LiveSettingsResponse:
    """Patch live-query settings and apply them to the running pipeline.

    The patch is a free-form JSON object (subset of LiveQuerySettings fields);
    validation happens in RuntimeConfigStore.update_live (raises
    RuntimeConfigError → 400 with the offending field).
    """
    patch = await request.json()
    if not isinstance(patch, dict):
        from src.runtime_config import RuntimeConfigError

        raise RuntimeConfigError("Request body must be a JSON object.")

    new_live = runtime.update_live(patch)
    # Apply to the live pipeline. Serialized against ingestion mutations.
    with lock:
        pipeline.apply_live_settings(new_live)
    logger.info("Live settings patched via API (request_id=%s)", request_id)
    return LiveSettingsResponse(live=new_live.to_dict(), request_id=request_id)


@router.get("/runtime/index", response_model=StagedIndexResponse)
def get_staged_index(
    runtime=Depends(get_runtime_config),
    request_id: str = Depends(get_request_id),
) -> StagedIndexResponse:
    staged = runtime.staged_index
    active = runtime.active_index
    return StagedIndexResponse(
        staged_index=staged.to_dict(),
        rebuild_required=(staged.to_dict() != active.to_dict()),
        request_id=request_id,
    )


@router.patch("/runtime/index", response_model=StagedIndexResponse)
async def patch_staged_index(
    request: Request,
    runtime=Depends(get_runtime_config),
    request_id: str = Depends(get_request_id),
) -> StagedIndexResponse:
    """Stage index-construction settings. Explicitly does NOT rebuild — the
    response's ``rebuild_required`` tells the UI a rebuild is pending."""
    patch = await request.json()
    if not isinstance(patch, dict):
        from src.runtime_config import RuntimeConfigError

        raise RuntimeConfigError("Request body must be a JSON object.")

    staged = runtime.stage_index(patch)
    active = runtime.active_index
    return StagedIndexResponse(
        staged_index=staged.to_dict(),
        rebuild_required=(staged.to_dict() != active.to_dict()),
        request_id=request_id,
    )


@router.post("/runtime/index/reset", response_model=StagedIndexResponse)
def reset_staged_index(
    runtime=Depends(get_runtime_config),
    request_id: str = Depends(get_request_id),
) -> StagedIndexResponse:
    staged = runtime.reset_staged_to_active()
    return StagedIndexResponse(
        staged_index=staged.to_dict(),
        rebuild_required=False,  # just reset → matches active
        request_id=request_id,
    )
