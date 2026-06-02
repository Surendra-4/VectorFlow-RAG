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
    # Phase 13: which named index is serving live retrieval (None = default).
    active_index_name: Optional[str] = None
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


# --------------------------------------------------------------------------- #
# Model management (Phase 12d)
# --------------------------------------------------------------------------- #


class ProviderModelSchema(BaseModel):
    """Frontend-facing model metadata (mirrors providers.ProviderModel)."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    kind: str = "chat"
    label: Optional[str] = None
    context_window: Optional[int] = None
    supports_streaming: bool = True
    supports_tools: bool = False
    multilingual: bool = False
    installed: Optional[bool] = None
    size_bytes: Optional[int] = None
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    ram_estimate_bytes: Optional[int] = None
    pricing: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class ProviderCapabilitiesSchema(BaseModel):
    """Frontend-facing provider capabilities (mirrors ProviderCapabilities)."""

    name: str
    label: str
    location: str
    requires_api_key: bool
    supports_chat: bool = True
    supports_streaming: bool = True
    supports_model_listing: bool = True
    supports_install: bool = False
    supports_embeddings: bool = False
    base_url_configurable: bool = False
    default_base_url: Optional[str] = None
    docs_url: Optional[str] = None
    notes: str = ""
    # Whether this provider currently has an API key configured (online only).
    key_configured: bool = False
    key_hint: Optional[str] = None


class ProvidersResponse(BaseModel):
    providers: List[ProviderCapabilitiesSchema]
    request_id: str


class ModelListResponse(BaseModel):
    provider: str
    models: List[ProviderModelSchema]
    request_id: str


class InstallRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Model id to pull, e.g. 'llama3.2:1b'.")
    provider: str = Field(default="ollama", description="Offline provider to install into.")


class DeleteModelResponse(BaseModel):
    provider: str
    name: str
    deleted: bool
    request_id: str


class SetApiKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=1, description="Stored backend-side only; never returned.")


class ApiKeyStatusResponse(BaseModel):
    provider: str
    configured: bool
    hint: Optional[str] = None
    request_id: str


class ConnectionValidationResponse(BaseModel):
    provider: str
    ok: bool
    message: str = ""
    models_available: Optional[int] = None
    latency_ms: Optional[float] = None
    request_id: str


class ChatModelConfigSchema(BaseModel):
    """Selection of a chat model. No api_key field — keys live backend-side."""

    model_config = ConfigDict(protected_namespaces=())

    provider: str = "ollama"
    model: str = "tinyllama"
    base_url: Optional[str] = None
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    request_timeout_s: int = Field(default=120, ge=1, le=3600)


class SelectModelRequest(BaseModel):
    """Switch the active chat model at runtime."""

    model_config = ConfigDict(protected_namespaces=())

    provider: str = Field(..., description="Provider name, e.g. 'ollama' or 'openai'.")
    model: str = Field(..., min_length=1, description="Model id within that provider.")
    base_url: Optional[str] = None
    max_tokens: Optional[int] = Field(default=None, ge=1, le=32768)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    request_timeout_s: Optional[int] = Field(default=None, ge=1, le=3600)


class ActiveModelResponse(BaseModel):
    """The chat model the pipeline is currently using."""

    model_config = ConfigDict(protected_namespaces=())

    provider: str
    model: str
    base_url: Optional[str] = None
    location: Optional[str] = None  # offline/online (from capabilities)
    request_id: str


# --------------------------------------------------------------------------- #
# Runtime configuration (Phase 12c/d)
# --------------------------------------------------------------------------- #


class RuntimeConfigResponse(BaseModel):
    """Full runtime-config snapshot (live + staged index + active index)."""

    version: int
    live: Dict[str, Any]
    staged_index: Dict[str, Any]
    active_index: Dict[str, Any]
    request_id: str


class LiveSettingsPatch(BaseModel):
    """Partial update to live-query settings. Any subset of fields.

    Sent through as a plain dict patch to RuntimeConfigStore.update_live so
    the validation lives in exactly one place (the store).
    """

    model_config = ConfigDict(extra="allow")


class IndexSettingsPatch(BaseModel):
    """Partial update to STAGED index-construction settings. Never rebuilds."""

    model_config = ConfigDict(extra="allow")


class LiveSettingsResponse(BaseModel):
    live: Dict[str, Any]
    request_id: str


class StagedIndexResponse(BaseModel):
    staged_index: Dict[str, Any]
    # True when staged differs from active — i.e. a rebuild is needed for the
    # staged changes to take effect.
    rebuild_required: bool
    request_id: str


# --------------------------------------------------------------------------- #
# Index management (Phase 12e–i)
# --------------------------------------------------------------------------- #


class RecipeListResponse(BaseModel):
    recipes: List[Dict[str, Any]]
    request_id: str


class RecipeValidateRequest(BaseModel):
    recipe: str = Field(..., description="Recipe id, e.g. 'ivf_pq'.")
    params: Optional[Dict[str, Any]] = None
    dim: int = Field(..., ge=1, le=65536, description="Vector dimension.")
    n_vectors: int = Field(default=0, ge=0)


class RecipeValidateResponse(BaseModel):
    validation: Dict[str, Any]
    request_id: str


class IndexListResponse(BaseModel):
    indexes: List[Dict[str, Any]]
    active: Optional[str] = None
    request_id: str


class IndexDetailResponse(BaseModel):
    index: Dict[str, Any]
    request_id: str


class CreateIndexRequest(BaseModel):
    """Build a new named index from the currently-ingested corpus."""

    name: str = Field(..., min_length=1, max_length=64)
    backend: str = Field(default="faiss")
    index_type: str = Field(default="hnsw", description="Recipe id or legacy type.")
    build_params: Optional[Dict[str, Any]] = None
    search_params: Optional[Dict[str, Any]] = None
    make_active: bool = False
    overwrite: bool = False
    description: str = ""


class JobAcceptedResponse(BaseModel):
    job_id: str
    type: str
    status: str
    request_id: str


class IndexActionResponse(BaseModel):
    index_name: str
    action: str
    active: Optional[str] = None
    request_id: str


class CompatibilityResponse(BaseModel):
    report: Dict[str, Any]
    request_id: str


class BenchmarkRequest(BaseModel):
    """Benchmark several FAISS recipes over the current corpus (background job)."""

    recipes: List[str] = Field(..., min_length=1, description="Recipe ids to compare.")
    k: int = Field(default=10, ge=1, le=100)
    params: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None, description="Per-recipe param overrides keyed by recipe id."
    )
    persist: bool = Field(default=True, description="Write a schema-versioned artifact.")


class JobResponse(BaseModel):
    job: Dict[str, Any]
    request_id: str


class JobListResponse(BaseModel):
    jobs: List[Dict[str, Any]]
    request_id: str
