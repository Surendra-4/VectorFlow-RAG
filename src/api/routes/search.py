# src/api/routes/search.py

"""POST /api/v1/search — RRF/expansion/rerank retrieval."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline, get_request_id
from src.api.schemas import RetrievalResult, SearchRequest, SearchResponse

router = APIRouter(tags=["retrieval"])


def _to_result(d: Dict[str, Any]) -> RetrievalResult:
    """Coerce a pipeline result dict into the validated response model.

    Pipeline results carry top-level provenance fields plus a nested
    ``metadata`` dict; the response model preserves both. Unknown
    pipeline fields are kept under ``extra='allow'`` semantics.
    """
    return RetrievalResult(**d)


@router.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest,
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
) -> SearchResponse:
    if pipeline.hybrid_retriever is None:
        # Pipeline isn't ingested yet → empty result, not a 500.
        return SearchResponse(query=req.query, results=[], trace=None, request_id=request_id)

    if req.return_trace:
        results, trace = pipeline.search(req.query, k=req.k, return_trace=True)
        trace_dict = trace.to_dict()
    else:
        results = pipeline.search(req.query, k=req.k, return_trace=False)
        trace_dict = None

    return SearchResponse(
        query=req.query,
        results=[_to_result(r) for r in results],
        trace=trace_dict,
        request_id=request_id,
    )
