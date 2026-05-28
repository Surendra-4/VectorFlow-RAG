# src/api/schemas.py

"""
Pydantic request/response models for the VectorFlow-RAG HTTP API.

These are the **stable contract** with API consumers (the future Next.js
frontend, the monitoring dashboard, external integrations). Adding
fields is non-breaking; removing or renaming fields requires a major
version bump (``/api/v2/``).

Every response model carries ``request_id`` so consumers can correlate
logs across distributed callers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------------------------------- #
# Common
# --------------------------------------------------------------------------- #


class ErrorResponse(BaseModel):
    """Structured error payload returned by every failing endpoint."""

    code: str = Field(..., description="Machine-readable error code (snake_case).")
    message: str = Field(..., description="Human-readable summary.")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional structured details (field errors, etc.)."
    )
    request_id: str = Field(..., description="Correlation ID echoed from request context.")


# --------------------------------------------------------------------------- #
# Health / Status
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Always 'ok' when the service is responsive.")
    app_version: str
    uptime_s: float
    request_id: str


class StatusResponse(BaseModel):
    """Comprehensive pipeline state — drives UI status panels."""

    app_name: str
    app_version: str
    vector_store_backend: str
    embedder_model: str
    embedder_dimension: int
    embedder_device: Optional[str] = None
    reranker_enabled: bool
    reranker_model: Optional[str] = None
    expansion_enabled: bool
    expansion_strategies: List[str] = Field(default_factory=list)
    cache_backend: str
    documents_ingested: int
    chunks_indexed: int
    corpus_fingerprint: Optional[str]
    rrf_k: int
    candidates_per_modality: int
    uptime_s: float
    request_id: str


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class IngestTextRequest(BaseModel):
    documents: List[str] = Field(..., min_length=1, description="Raw text documents to ingest.")
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Per-document metadata. If provided, must match documents length.",
    )
    reset: bool = Field(
        default=True,
        description="Wipe the index before ingesting. Default True preserves the legacy behavior.",
    )


class IngestPathsRequest(BaseModel):
    paths: List[str] = Field(..., min_length=1, description="Absolute or working-tree-relative paths.")
    reset: bool = True
    fail_fast: Optional[bool] = Field(
        default=None,
        description="Override settings.ingestion.fail_fast. If None, uses config default.",
    )


class IngestionFailure(BaseModel):
    path: str
    reason: str


class IngestionResponse(BaseModel):
    successes: List[str] = Field(default_factory=list)
    failures: List[IngestionFailure] = Field(default_factory=list)
    chunks: int = Field(0, description="Total chunks added across successful files/docs.")
    documents_ingested: int
    corpus_fingerprint: Optional[str]
    request_id: str


# --------------------------------------------------------------------------- #
# Search & Q&A
# --------------------------------------------------------------------------- #


class RetrievalResult(BaseModel):
    """One chunk returned by retrieval, with full provenance."""

    model_config = ConfigDict(extra="allow")

    text: str
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    document_name: Optional[str] = None
    source_path: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    hybrid_score: float
    rrf_score: Optional[float] = None
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=100)
    return_trace: bool = False


class SearchResponse(BaseModel):
    query: str
    results: List[RetrievalResult]
    trace: Optional[Dict[str, Any]] = None
    request_id: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k_docs: int = Field(default=3, ge=1, le=20)
    return_sources: bool = True
    stream: bool = Field(
        default=False,
        description="If True, response is text/event-stream with token events.",
    )


class AskMetrics(BaseModel):
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    num_context_docs: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[RetrievalResult]] = None
    metrics: AskMetrics
    request_id: str


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


class CacheStatsResponse(BaseModel):
    backend: str
    hits: int
    misses: int
    sets: int
    deletes: int
    errors: int
    hit_ratio: float
    request_id: str


class CacheClearResponse(BaseModel):
    backend: str
    message: str
    request_id: str


# --------------------------------------------------------------------------- #
# Documents / Admin
# --------------------------------------------------------------------------- #


class DocumentSummary(BaseModel):
    doc_id: str
    document_name: Optional[str] = None
    source_path: Optional[str] = None
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]
    total_documents: int
    total_chunks: int
    corpus_fingerprint: Optional[str]
    request_id: str


class IndexResetResponse(BaseModel):
    cleared: bool
    previous_chunks: int
    request_id: str
