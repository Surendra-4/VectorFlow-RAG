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
  reranker_enabled: boolean;
  reranker_model?: string | null;
  expansion_enabled: boolean;
  cache_backend: string;
  documents_ingested: number;
  chunks_indexed: number;
  corpus_fingerprint?: string | null;
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
