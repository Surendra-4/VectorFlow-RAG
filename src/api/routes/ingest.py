# src/api/routes/ingest.py

"""Ingestion endpoints: text bodies, server-side paths, and multipart file upload."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from src.api.dependencies import (
    get_ingest_lock,
    get_pipeline,
    get_request_id,
    require_user_if_enabled,
)
from src.api.schemas import (
    IngestionFailure,
    IngestionResponse,
    IngestPathsRequest,
    IngestTextRequest,
)
from src.auth import service as auth_service
from src.observability import get_metrics

router = APIRouter(tags=["ingestion"], prefix="/ingest")


def _to_failures(raw) -> List[IngestionFailure]:
    return [IngestionFailure(path=f["path"], reason=f["reason"]) for f in (raw or [])]


def _record_ingest_metrics(mode: str, elapsed_ms: float, chunks: int, failure_count: int) -> None:
    """Push ingestion metrics into the registry. Never raises."""
    try:
        m = get_metrics()
        m.ingestions_total.inc(mode)
        m.ingest_latency_ms.observe(elapsed_ms)
        m.chunks_ingested_total.inc(n=chunks)
        if failure_count:
            m.ingest_failures_total.inc(n=failure_count)
    except Exception:  # pragma: no cover
        pass


@router.post("/text", response_model=IngestionResponse)
def ingest_text(
    body: IngestTextRequest,
    pipeline=Depends(get_pipeline),
    lock=Depends(get_ingest_lock),
    user=Depends(require_user_if_enabled),
    request_id: str = Depends(get_request_id),
) -> IngestionResponse:
    if body.metadatas is not None and len(body.metadatas) != len(body.documents):
        raise HTTPException(
            status_code=400,
            detail=f"metadatas length ({len(body.metadatas)}) must equal documents length ({len(body.documents)})",
        )
    t0 = time.perf_counter()
    with lock:
        pipeline.ingest_documents(body.documents, metadatas=body.metadatas, reset=body.reset)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _record_ingest_metrics("text", elapsed_ms, len(pipeline.corpus), 0)
    if user is not None:
        auth_service.record_for_user_id(
            user.id, documents_ingested=len(body.documents), chunks_ingested=len(pipeline.corpus)
        )
    return IngestionResponse(
        successes=[f"text:{i}" for i in range(len(body.documents))],
        failures=[],
        chunks=len(pipeline.corpus),
        documents_ingested=pipeline.document_count,
        corpus_fingerprint=pipeline.corpus_fingerprint,
        request_id=request_id,
    )


@router.post("/paths", response_model=IngestionResponse)
def ingest_paths(
    body: IngestPathsRequest,
    pipeline=Depends(get_pipeline),
    lock=Depends(get_ingest_lock),
    user=Depends(require_user_if_enabled),
    request_id: str = Depends(get_request_id),
) -> IngestionResponse:
    kwargs = {"reset": body.reset}
    if body.fail_fast is not None:
        kwargs["fail_fast"] = body.fail_fast
    t0 = time.perf_counter()
    with lock:
        result = pipeline.ingest_files(body.paths, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _record_ingest_metrics("paths", elapsed_ms, result["chunks"], len(result["failures"]))
    if user is not None:
        auth_service.record_for_user_id(
            user.id, documents_ingested=len(result["successes"]), chunks_ingested=result["chunks"]
        )
    return IngestionResponse(
        successes=result["successes"],
        failures=_to_failures(result["failures"]),
        chunks=result["chunks"],
        documents_ingested=pipeline.document_count,
        corpus_fingerprint=pipeline.corpus_fingerprint,
        request_id=request_id,
    )


@router.post("/files", response_model=IngestionResponse)
async def ingest_files(
    files: List[UploadFile] = File(..., description="Files to ingest."),
    reset: bool = Form(True),
    pipeline=Depends(get_pipeline),
    lock=Depends(get_ingest_lock),
    user=Depends(require_user_if_enabled),
    request_id: str = Depends(get_request_id),
) -> IngestionResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Persist uploads to a request-scoped temp dir; the dir is cleaned up
    # after ingestion. The loader registry expects on-disk files so it
    # can MIME-sniff and stat them.
    tmp_dir = Path(tempfile.mkdtemp(prefix="vfr_upload_"))
    paths: List[Path] = []
    try:
        for f in files:
            target = tmp_dir / Path(f.filename or "upload.bin").name
            with open(target, "wb") as out:
                shutil.copyfileobj(f.file, out)
            paths.append(target)

        t0 = time.perf_counter()
        with lock:
            result = pipeline.ingest_files(paths, reset=reset)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _record_ingest_metrics("files", elapsed_ms, result["chunks"], len(result["failures"]))
        if user is not None:
            auth_service.record_for_user_id(
                user.id, documents_ingested=len(result["successes"]), chunks_ingested=result["chunks"]
            )

        return IngestionResponse(
            successes=[Path(p).name for p in result["successes"]],
            failures=_to_failures(result["failures"]),
            chunks=result["chunks"],
            documents_ingested=pipeline.document_count,
            corpus_fingerprint=pipeline.corpus_fingerprint,
            request_id=request_id,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
