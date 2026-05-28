# src/api/routes/admin.py

"""DELETE /api/v1/index — wipe the entire retrieval index. Confirmation gated."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_ingest_lock, get_pipeline, get_request_id
from src.api.schemas import IndexResetResponse

router = APIRouter(tags=["admin"])


@router.delete("/index", response_model=IndexResetResponse)
def reset_index(
    confirm: bool = Query(False, description="Must be true to actually wipe."),
    pipeline=Depends(get_pipeline),
    lock=Depends(get_ingest_lock),
    request_id: str = Depends(get_request_id),
) -> IndexResetResponse:
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Pass ?confirm=true to actually reset the index.",
        )

    with lock:
        previous_chunks = len(pipeline.corpus)
        try:
            pipeline.vector_store.delete_collection()
        except Exception:
            # Already empty or transient — non-fatal for the admin op.
            pass
        # Rebuild a clean store and zero out the in-memory state.
        from src.vector_store import make_vector_store

        pipeline.vector_store = make_vector_store(
            persist_directory=pipeline.vector_store.persist_directory,
            backend=pipeline.settings.vector_store.backend,
        )
        pipeline.bm25_retriever = None
        pipeline.hybrid_retriever = None
        pipeline.corpus = []
        pipeline.document_count = 0
        pipeline.corpus_fingerprint = None
        # Also clear the cache so stale retrieval results don't linger.
        pipeline.cache.clear()

    return IndexResetResponse(
        cleared=True,
        previous_chunks=previous_chunks,
        request_id=request_id,
    )
