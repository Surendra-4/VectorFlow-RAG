// src/lib/api/types.ts

/**
 * TypeScript types mirroring the Pydantic schemas in src/api/schemas.py.
 *
 * These are the only contract that binds frontend to backend. Keep them
 * tight — if a field disappears here the compiler tells you exactly which
 * components break.
 */

// ---- Common ----------------------------------------------------------- //

export interface ErrorResponse {
  code: string;
  message: string;
  details?: Record<string, unknown> | null;
  request_id: string;
}

// ---- Status ----------------------------------------------------------- //

export interface StatusResponse {
  app_name: string;
  app_version: string;
  vector_store_backend: string;
  embedder_model: string;
  embedder_dimension?: number;
  embedder_device?: string | null;
  reranker_enabled: boolean;
  reranker_model?: string | null;
  expansion_enabled: boolean;
  expansion_strategies?: string[];
  cache_backend: string;
  documents_ingested: number;
  chunks_indexed: number;
  corpus_fingerprint?: string | null;
  rrf_k?: number;
  candidates_per_modality?: number;
  uptime_s: number;
  request_id: string;
}

// ---- Retrieval ------------------------------------------------------- //

export interface RetrievalResult {
  text: string;
  chunk_id?: string | null;
  doc_id?: string | null;
  document_name?: string | null;
  source_path?: string | null;
  page_number?: number | null;
  chunk_index?: number | null;
  hybrid_score: number;
  rrf_score: number;
  vector_rank?: number | null;
  bm25_rank?: number | null;
  rerank_score?: number | null;
  metadata: Record<string, unknown>;
}

export interface SearchRequest {
  query: string;
  k?: number;
  return_trace?: boolean;
}

export interface SearchResponse {
  results: RetrievalResult[];
  trace?: Record<string, unknown> | null;
  request_id: string;
}

// ---- Ingestion ------------------------------------------------------- //

export interface IngestionFailure {
  path: string;
  reason: string;
}

export interface IngestionResponse {
  successes: string[];
  failures: IngestionFailure[];
  chunks: number;
  documents_ingested: number;
  corpus_fingerprint?: string | null;
  request_id: string;
}

export interface IngestTextRequest {
  documents: string[];
  metadatas?: Array<Record<string, unknown>> | null;
  reset?: boolean;
}

export interface IngestPathsRequest {
  paths: string[];
  reset?: boolean;
  fail_fast?: boolean | null;
}

// ---- Ask (blocking + streaming) -------------------------------------- //

export interface AskRequest {
  query: string;
  k_docs?: number;
  return_sources?: boolean;
  stream?: boolean;
}

export interface AskMetrics {
  retrieval_time_ms: number;
  generation_time_ms: number;
  total_time_ms: number;
  num_context_docs: number;
}

export interface AskResponse {
  question: string;
  answer: string;
  sources?: RetrievalResult[] | null;
  metrics: AskMetrics;
  request_id: string;
}

// SSE event payloads from /api/v1/ask?stream=true
export interface SourcesSseEvent {
  type: "sources";
  request_id: string;
  sources: RetrievalResult[];
}
export interface TokenSseEvent {
  type: "token";
  token: string;
}
export interface DoneSseEvent {
  type: "done";
  request_id: string;
  answer: string;
  metrics: AskMetrics;
}
export interface ErrorSseEvent {
  type: "error";
  message: string;
  request_id?: string;
}
export type AskSseEvent = SourcesSseEvent | TokenSseEvent | DoneSseEvent | ErrorSseEvent;

// ---- Cache ----------------------------------------------------------- //

export interface CacheStatsResponse {
  backend: string;
  hits: number;
  misses: number;
  sets: number;
  deletes: number;
  errors: number;
  hit_ratio: number;
  request_id: string;
}

// ---- Documents ------------------------------------------------------- //

export interface DocumentSummary {
  doc_id: string;
  document_name?: string | null;
  source_path?: string | null;
  chunk_count: number;
  first_seen_chunk_id?: string | null;
}

export interface DocumentsResponse {
  documents: DocumentSummary[];
  total_documents: number;
  total_chunks: number;
  request_id: string;
}

// ---- Metrics --------------------------------------------------------- //

export interface HistogramSnapshot {
  count: number;
  p50: number | null;
  p95: number | null;
  p99: number | null;
  min: number | null;
  max: number | null;
  mean: number | null;
}

export interface LabeledCounterEntry {
  value: number;
  [label: string]: number | string;
}

export interface LabeledHistogramEntry {
  stats: HistogramSnapshot;
  [label: string]: HistogramSnapshot | string;
}

export interface MetricsSnapshot {
  uptime_s: number;
  counters: Record<string, number>;
  gauges: Record<string, number>;
  labeled_counters: Record<string, LabeledCounterEntry[]>;
  histograms: Record<string, HistogramSnapshot>;
  labeled_histograms: Record<string, LabeledHistogramEntry[]>;
  ring_buffer_sizes: Record<string, number>;
  request_id: string;
}

export interface RecentTracesResponse {
  traces: Array<Record<string, unknown>>;
  limit: number;
  request_id: string;
}

// ---- Phase 12: providers & models ------------------------------------ //

export interface ProviderCapabilities {
  name: string;
  label: string;
  location: "offline" | "online";
  requires_api_key: boolean;
  supports_chat: boolean;
  supports_streaming: boolean;
  supports_model_listing: boolean;
  supports_install: boolean;
  supports_embeddings: boolean;
  base_url_configurable: boolean;
  default_base_url?: string | null;
  docs_url?: string | null;
  notes: string;
  key_configured: boolean;
  key_hint?: string | null;
}

export interface ProviderModel {
  id: string;
  kind: "chat" | "embedding" | "reranker";
  label?: string | null;
  context_window?: number | null;
  supports_streaming: boolean;
  supports_tools: boolean;
  multilingual: boolean;
  installed?: boolean | null;
  size_bytes?: number | null;
  parameter_size?: string | null;
  quantization?: string | null;
  ram_estimate_bytes?: number | null;
  pricing?: Record<string, unknown> | null;
  description?: string | null;
}

export interface ProvidersResponse {
  providers: ProviderCapabilities[];
  request_id: string;
}

export interface ModelListResponse {
  provider: string;
  models: ProviderModel[];
  request_id: string;
}

export interface ApiKeyStatusResponse {
  provider: string;
  configured: boolean;
  hint?: string | null;
  request_id: string;
}

export interface ConnectionValidationResponse {
  provider: string;
  ok: boolean;
  message: string;
  models_available?: number | null;
  latency_ms?: number | null;
  request_id: string;
}

export interface ActiveModelResponse {
  provider: string;
  model: string;
  base_url?: string | null;
  location?: string | null;
  request_id: string;
}

export interface InstallProgressEvent {
  status: string;
  total?: number | null;
  completed?: number | null;
  percent?: number | null;
  digest?: string | null;
}

// ---- Phase 12: runtime config ---------------------------------------- //

export interface RuntimeConfigResponse {
  version: number;
  live: Record<string, unknown>;
  staged_index: Record<string, unknown>;
  active_index: Record<string, unknown>;
  request_id: string;
}

export interface StagedIndexResponse {
  staged_index: Record<string, unknown>;
  rebuild_required: boolean;
  request_id: string;
}

// ---- Phase 12: indexes & recipes ------------------------------------- //

export interface RecipeParam {
  name: string;
  label: string;
  kind: string;
  default: number;
  minimum: number;
  maximum: number;
  description: string;
  group: string;
}

export interface RecipeSpec {
  id: string;
  label: string;
  mode: "basic" | "advanced";
  requires_training: boolean;
  description: string;
  params: RecipeParam[];
  latency_class: string;
  training_cost: string;
  supports_refine: boolean;
}

export interface RecipeEstimate {
  memory_bytes: number;
  bytes_per_vector: number;
  training_cost: string;
  latency_class: string;
  needs_training: boolean;
  min_training_points?: number | null;
}

export interface RecipeValidation {
  ok: boolean;
  recipe: string;
  factory_string: string;
  resolved_params: Record<string, number>;
  errors: Array<{ field: string; message: string }>;
  warnings: string[];
  estimate?: RecipeEstimate | null;
}

export interface IndexProfile {
  name: string;
  backend: string;
  index_type: string;
  embedding_model: string;
  embedding_provider: string;
  vector_dimension: number;
  build_params: Record<string, number>;
  search_params: Record<string, number>;
  corpus_fingerprint?: string | null;
  chunk_size: number;
  chunk_overlap: number;
  normalize: boolean;
  num_vectors: number;
  metrics: Record<string, unknown>;
  created_at: number;
  last_used: number;
  description: string;
  compatibility_info: Record<string, unknown>;
}

export interface IndexListResponse {
  indexes: IndexProfile[];
  active?: string | null;
  request_id: string;
}

export interface CompatibilityIssue {
  field: string;
  severity: "blocking" | "rebuild" | "info";
  message: string;
  index_value?: unknown;
  target_value?: unknown;
}

export interface CompatibilityReport {
  index_name: string;
  compatible: boolean;
  action: "reuse" | "rebuild" | "create_new";
  message: string;
  issues: CompatibilityIssue[];
}

// ---- Phase 12: jobs --------------------------------------------------- //

export interface JobState {
  id: string;
  type: string;
  label: string;
  status: "pending" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  message: string;
  result?: Record<string, unknown> | null;
  error?: string | null;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  cancel_requested: boolean;
  history?: Array<Record<string, unknown>>;
}

export interface JobAcceptedResponse {
  job_id: string;
  type: string;
  status: string;
  request_id: string;
}

export interface BenchmarkResultRow {
  index_name: string;
  recipe: string;
  factory_string: string;
  k: number;
  num_vectors: number;
  dimension: number;
  recall_at_k: number;
  mrr: number;
  latency_ms_mean: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
  queries_per_sec: number;
  build_seconds: number;
  ingest_vectors_per_sec: number;
  index_size_bytes: number;
  estimated_memory_bytes: number;
  timestamp: number;
}
