# src/api/routes/cache.py

"""Cache stats + clear endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline, get_request_id
from src.api.schemas import CacheClearResponse, CacheStatsResponse

router = APIRouter(tags=["observability"], prefix="/cache")


@router.get("/stats", response_model=CacheStatsResponse)
def cache_stats(
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
) -> CacheStatsResponse:
    snap = pipeline.cache.stats.snapshot()
    return CacheStatsResponse(
        backend=pipeline.cache.backend_name,
        hits=snap["hits"],
        misses=snap["misses"],
        sets=snap["sets"],
        deletes=snap["deletes"],
        errors=snap["errors"],
        hit_ratio=snap["hit_ratio"],
        request_id=request_id,
    )


@router.post("/clear", response_model=CacheClearResponse)
def cache_clear(
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
) -> CacheClearResponse:
    pipeline.cache.clear()
    return CacheClearResponse(
        backend=pipeline.cache.backend_name,
        message="Cache cleared",
        request_id=request_id,
    )
