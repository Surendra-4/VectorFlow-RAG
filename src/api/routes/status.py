# src/api/routes/status.py

"""Comprehensive pipeline state — UI status-panel data source."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_pipeline, get_request_id, get_started_at
from src.api.schemas import StatusResponse

router = APIRouter(tags=["meta"])


@router.get("/status", response_model=StatusResponse)
def status(
    request: Request,
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
    started_at: float = Depends(get_started_at),
) -> StatusResponse:
    settings = pipeline.settings
    return StatusResponse(
        app_name=settings.app.name,
        app_version=settings.app.version,
        vector_store_backend=settings.vector_store.backend,
        embedder_model=pipeline.embedder.model_name,
        embedder_dimension=pipeline.embedder.dimension,
        embedder_device=getattr(pipeline.embedder, "device", None),
        reranker_enabled=pipeline.enable_reranker,
        reranker_model=settings.reranker.model_name if pipeline.enable_reranker else None,
        expansion_enabled=pipeline.enable_expansion,
        expansion_strategies=list(settings.query_expansion.strategies) if pipeline.enable_expansion else [],
        cache_backend=pipeline.cache.backend_name,
        documents_ingested=pipeline.document_count,
        chunks_indexed=len(pipeline.corpus),
        corpus_fingerprint=pipeline.corpus_fingerprint,
        rrf_k=settings.retrieval.rrf_k,
        candidates_per_modality=settings.retrieval.candidates_per_modality,
        active_index_name=getattr(pipeline, "active_index_name", None),
        uptime_s=time.monotonic() - started_at if started_at else 0.0,
        request_id=request_id,
    )
