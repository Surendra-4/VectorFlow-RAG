# src/api/routes/documents.py

"""GET /api/v1/documents — list ingested documents grouped by doc_id."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline, get_request_id
from src.api.schemas import DocumentListResponse, DocumentSummary

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
) -> DocumentListResponse:
    """
    Group all known chunks by ``doc_id`` and return a per-document summary.

    Uses the BM25 retriever's metadata index as the source of truth — it
    already mirrors the canonical chunk metadata. When no documents have
    been ingested, returns an empty list (not a 404).
    """
    if pipeline.bm25_retriever is None:
        return DocumentListResponse(
            documents=[],
            total_documents=0,
            total_chunks=0,
            corpus_fingerprint=None,
            request_id=request_id,
        )

    by_doc: Dict[str, Dict] = defaultdict(lambda: {"chunk_count": 0, "document_name": None, "source_path": None})
    total_chunks = 0
    for meta in pipeline.bm25_retriever._metadatas:
        if not meta:
            continue
        doc_id = meta.get("document_id")
        if not doc_id:
            continue
        entry = by_doc[doc_id]
        entry["chunk_count"] += 1
        # First non-None wins; chunks within a doc share these.
        if entry["document_name"] is None:
            entry["document_name"] = meta.get("document_name")
        if entry["source_path"] is None:
            entry["source_path"] = meta.get("source_path")
        total_chunks += 1

    docs = [
        DocumentSummary(
            doc_id=doc_id,
            document_name=info["document_name"],
            source_path=info["source_path"],
            chunk_count=info["chunk_count"],
        )
        for doc_id, info in sorted(by_doc.items())
    ]
    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=total_chunks,
        corpus_fingerprint=pipeline.corpus_fingerprint,
        request_id=request_id,
    )
