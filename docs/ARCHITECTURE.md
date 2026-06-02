# VectorFlow-RAG — System Architecture

Local-first, provenance-aware hybrid RAG platform. This document is the
single reference for how the system is put together end to end. It is
**local documentation** — no remote publishing implied.

> Status: as of Phase 11 (multilingual). Backend: Python / FastAPI.
> Frontend: Next.js 14 (App Router). Vector backends: ChromaDB (default)
> and FAISS HNSW. All inference is local (Ollama LLM + sentence-transformers).

---

## 1. End-to-end system architecture

```
                    ┌──────────────────────────── Frontend (Next.js) ───────────────────────────┐
                    │  Chat · Search · Ingest · Documents · Dashboard · Traces · Settings        │
                    │  typed API client · SSE parser · polling hooks · dir="auto" rendering       │
                    └───────────────────────────────────┬───────────────────────────────────────┘
                                                         │ HTTP / SSE (typed contract)
                    ┌───────────────────────────────────▼───────────────────────────────────────┐
                    │                         FastAPI service (src/api)                           │
                    │  middleware: request-id + timing + metrics                                  │
                    │  routes: health status ingest search ask cache documents observability     │
                    │  structured errors · OpenAPI · dependency-injected pipeline singleton       │
                    └───────────────────────────────────┬───────────────────────────────────────┘
                                                         │
                    ┌───────────────────────────────────▼───────────────────────────────────────┐
                    │                       RAGPipeline (src/rag_pipeline.py)                      │
                    │   orchestrates: chunk → embed → index → retrieve → fuse → rerank → generate │
                    └─┬───────────┬───────────┬───────────┬───────────┬───────────┬──────────────┘
                      │           │           │           │           │           │
              ┌───────▼──┐ ┌──────▼─────┐ ┌───▼─────┐ ┌───▼──────┐ ┌──▼───────┐ ┌─▼──────────┐
              │ loaders/ │ │ chunker +  │ │ embedder│ │ vector   │ │ bm25     │ │ reranker   │
              │ registry │ │ identity   │ │ adapter │ │ store    │ │ retriever│ │ (optional) │
              │          │ │            │ │ (cache) │ │ chroma / │ │          │ │            │
              │          │ │            │ │         │ │ faiss    │ │          │ │            │
              └──────────┘ └────────────┘ └─────────┘ └──────────┘ └──────────┘ └────────────┘
                      │                        │            │            │
              ┌───────▼────────────────────────▼────────────▼────────────▼───────┐
              │ cross-cutting: config (profiles) · cache (memory/redis) ·         │
              │ query_expansion · retrieval_trace · observability · logging       │
              └──────────────────────────────────────────────────────────────────┘
                                                         │
                                            ┌────────────▼────────────┐
                                            │   Ollama (local LLM)     │
                                            │   generation + HyDE +    │
                                            │   multi-query expansion  │
                                            └──────────────────────────┘
```

**Design invariants**
- The frontend depends only on the typed API contract (`frontend/src/lib/api/types.ts` mirrors `src/api/schemas.py`). No backend imports.
- Retrieval is **read-concurrent, lock-free**; ingestion **mutation is serialized**.
- Every swappable component sits behind a Protocol in `src/interfaces.py` (`VectorStoreProtocol`, `CacheProtocol`, `RerankerProtocol`, `EmbedderProtocol`, `LLMClientProtocol`, `LoaderProtocol`, `ExpansionStrategy`).
- All configuration flows through one place: `src/config.py` (`Settings`, env prefix `VFR_`, nested delimiter `__`).

---

## 2. Retrieval lifecycle

`RAGPipeline.search(query, k, return_trace=False)` →

```
query
 ├─ sanitize_query()            strip control chars · collapse whitespace · NFC · length cap
 ├─ full-retrieval cache lookup  key = (corpus_fingerprint, rag_config, query, k)
 │     └─ HIT → return cached results (skip everything below)   [Phase 6]
 │
 ├─ [if expansion enabled]  QueryExpansionPipeline.expand()      [Phase 5]
 │     ├─ MultiQueryExpander → N LLM-generated variants (parsed defensively, deduped, length-capped)
 │     └─ HyDEExpander       → M hypothetical documents
 │        → ExpandedQuery(queries=[original, *variants], hyde_documents=[...])
 │
 ├─ per query string:  hybrid_retriever.search()
 │     ├─ embedder.encode(q, input_type="query")  → dense vector (cache-checked)
 │     ├─ vector_store.search(emb, pool)           → {documents, distances, metadatas, ids}
 │     └─ bm25.search(q, pool)                     → [{text, score, rank, chunk_id, metadata}]
 ├─ per HyDE doc: embedder.encode(doc, input_type="query") → vector-only retrieval
 │
 ├─ Reciprocal Rank Fusion (src/rrf.py)            [Phase 1]
 │     rrf_with_ranks({"vector:original":[ids], "bm25:original":[ids],
 │                     "vector:variant_1":[...], ..., "vector:hyde_0":[...]}, k=60)
 │     → unified pool joined on chunk_id  (text-join fallback when ids absent)
 │
 ├─ [if reranker enabled]  cross_encoder.rerank(query, pool, top_n=k)   [Phase 1]
 │
 ├─ populate RetrievalTrace (stages, latencies, cache_stats, from_cache)
 ├─ store results in full-retrieval cache
 └─ record metrics (retrievals_total, latency histograms, ring buffer)   [Phase 9]
     → List[RetrievalResult]  (or (results, RetrievalTrace) if return_trace)
```

**Result shape** (every result, post Phase 3.5 + 1):
`text, chunk_id, doc_id, hybrid_score (=RRF), rrf_score, vector_rank, bm25_rank,
rerank_score?, document_name, source_path, page_number, chunk_index, metadata`

**Why RRF over alpha-fusion**: rank-based fusion is scale-invariant across
heterogeneous retrievers (BM25 raw scores vs cosine), needs no per-corpus
tuning, and is intersection-aware (docs found by both modalities get boosted).

---

## 3. Ingestion lifecycle

Two entry points share one indexing core (`RAGPipeline._index_chunks`):

- `ingest_documents(documents: list[str])` — raw text
- `ingest_files(paths)` — multi-format via the loader registry  [Phase 4]

```
files                                   raw text
 │                                          │
 ├─ per file (lock-serialized):            │
 │   ├─ resolve path + size guard          │
 │   ├─ registry.find(path) → loader       │
 │   ├─ loader.load() → LoadedDocument(pages=[LoadedPage(text, page_number, metadata)])
 │   └─ per-file try/except → successes / failures (batch never poisoned)
 │                                          │
 └────────────────┬─────────────────────────┘
                  ▼
   _chunks_from_loaded_document / chunk_text():
     doc_id = compute_doc_id(content)                 stable content hash      [Phase 3.5]
     per page → chunker.chunk_text(text, doc_id, chunk_index_offset)
        chunk_id = "{doc_id}:{chunk_index}:{sha(norm(text))[:8]}"  globally unique
        metadata = provenance (document_name, source_path, page_number, chunk_index, ...)
                  ▼
   _index_chunks():
     embeddings = embedder.encode(chunk_texts, input_type="passage")  (cache-checked)
     vector_store.add_documents(texts, embeddings, metadatas, ids=chunk_ids)
     bm25 = BM25Retriever(dict-corpus with chunk_id + metadata)
     hybrid_retriever rebuilt
     corpus_fingerprint = sha(sorted(chunk_ids))   → invalidates retrieval cache
     metrics: ingestions_total, ingest_latency_ms, chunks_ingested_total
```

**Loaders** (all behind `LoaderProtocol`, dispatched by extension/MIME):
text · markdown · json · csv/tsv · xlsx (1 page/sheet) · sqlite (1 page/table) ·
pdf (1 page/page, pypdf) · docx · image OCR (pytesseract, operator-installed lang packs).

**Identity strategy**: content-derived IDs make re-ingestion idempotent and
give every retrieved chunk a stable, source-attributed identity — the
foundation for dedup, citations, and incremental indexing.

---

## 4. Caching lifecycle  [Phase 6]

```
                  ┌──────────────── SafeCache ────────────────┐
   callers ──────▶│ never raises · counts hits/misses/errors  │──▶ backend
                  └────────────────────────────────────────────┘
                       │              │                 │
                  NullCache      MemoryCache         RedisCache
                  (disabled)   (LRU + TTL, thread-   (shared across
                                safe, default)        workers, optional)
```

| Namespace | Key | Invalidation |
|---|---|---|
| `vfr:emb:v1` | model + sha(text) + input_type | never stale (content-keyed); TTL eviction |
| `vfr:exp:v1` | exp_model + strategies + params + sha(query) | never stale; TTL eviction |
| `vfr:ret:v1` | **corpus_fingerprint** + rag_config + sha(query) + k | **implicit on re-ingest** (fingerprint changes → key changes) |

- **Structural invalidation**: there is no `invalidate()` API to misuse. Re-ingestion changes the corpus fingerprint, so stale retrieval-cache keys become unreachable and TTL out.
- **Graceful degradation**: if `backend=redis` and Redis is unreachable, the factory logs once and falls back to `MemoryCache` — retrieval still works, just colder.
- **Codec**: pickle (numpy-safe). Trust model = local-first / trusted Redis.
- **Switching embedder model** auto-invalidates embedding + retrieval caches (model name is in the key).

---

## 5. Observability lifecycle  [Phase 9]

```
request → middleware (request_id, timing) → handler → pipeline
                                                │
              ┌─────────────────────────────────▼──────────────────────────────┐
              │ MetricsRegistry (process-local, thread-safe primitives)         │
              │  Counter · LabeledCounter · Gauge · Histogram(rolling window) · │
              │  RingBuffer(recent traces / errors)                             │
              └─────────────────────────────────┬──────────────────────────────┘
                                                 │
   GET /api/v1/metrics/snapshot  ──────────────▶│  structured JSON
   GET /api/v1/metrics/{requests,cache,retrieval,ingestion,streams}
   GET /api/v1/traces/recent?limit=N
   GET /api/v1/metrics/prometheus ─────────────▶  Prometheus text (no client dep)
```

- **Overhead**: ~0.3–1 µs per primitive op; ~12 µs of metric work on a cache-hit retrieval. Sub-0.1% on any path doing real work.
- **Bounded retention**: histograms use a 300 s rolling window + sample cap; ring buffers are fixed-size. Memory is bounded regardless of throughput.
- **Cardinality discipline**: labels are bounded (endpoint, status, namespace) — never request_id or full URL.
- **Multi-worker**: metrics are process-local by design; aggregation is delegated to an external Prometheus scraper (no fake in-app distributed metrics).
- `RetrievalTrace.to_dict()` is the substrate the dashboard consumes — it captures original/sanitized/expanded queries, HyDE docs, per-strategy candidates, fusion, rerank stats, cache stats, final results, per-stage latencies.

---

## 6. API / service lifecycle  [Phase 8]

```
create_app() (factory)
 ├─ startup: build RAGPipeline singleton (models load once, lazily)
 ├─ middleware stack: CorrelationId → Timing(+metrics) → CORS
 ├─ routers under /api/v1 + /health
 ├─ exception handlers → ErrorResponse{code, message, details, request_id}
 └─ shutdown: clean release
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness (no pipeline interaction) |
| `/api/v1/status` | GET | Models, backends, doc/chunk counts, fingerprint |
| `/api/v1/ingest/{text,files,paths}` | POST | Ingestion (files = multipart) |
| `/api/v1/search` | POST | RRF/expansion/rerank retrieval; optional trace |
| `/api/v1/ask` | POST | Search → LLM; SSE stream via `stream=true` |
| `/api/v1/cache/{stats,clear}` | GET/POST | Cache observability + admin |
| `/api/v1/documents` | GET | Documents grouped by doc_id |
| `/api/v1/index` | DELETE | Reset index (requires `confirm=true`) |
| `/api/v1/metrics/*`, `/api/v1/traces/recent` | GET | Observability |

- **Concurrency model**: sync `def` handlers run in FastAPI's threadpool. Retrieval is read-only and lock-free; ingestion takes a process-wide lock.
- **Streaming**: SSE (fetch + `ReadableStream`), not WebSockets. Events: `sources → token* → done | error`. `active_streams` gauge tracks live sessions.
- **Versioning**: `/api/v1` prefix; breaking changes go to `/api/v2`.

---

## 7. Frontend architecture  [Phase 10–11]

```
frontend/src/
├── app/            App Router pages (all client components): chat, search,
│                   ingest, documents, dashboard, traces, settings + error.tsx
├── components/     ui/ · layout/ · chat/ · ingest/ · search/ · citations/ · dashboard/
└── lib/
    ├── api/        typed client (client.ts, errors.ts, sse.ts, types.ts, <resource>.ts)
    ├── hooks/      useApi · usePolling · useStreamingAsk · useTheme
    └── utils/      format · cn
```

- **State**: React built-ins only — no Redux/Zustand. Server state via thin `useApi`/`usePolling` hooks with `AbortController` cleanup.
- **SSE**: custom fetch-based parser (browser `EventSource` is GET-only) → `useStreamingAsk` aggregates `sources/token/done/error`.
- **Provenance-first UX**: `SourcePanel` surfaces document name, page, chunk index, RRF score, per-modality ranks, rerank score, chunk_id.
- **Rendering (Phase 11)**: `dir="auto"` on user-content containers (LTR/RTL auto); OS-native font stack for broad script coverage; no web-font download (offline-safe). UI-string i18n intentionally deferred.
- **Deploy**: `next dev` / `next build && next start` / `OUTPUT=export` static (desktop/Tauri). Frontend never assumes a colocated backend.

---

## 8. Multilingual profile architecture  [Phase 11]

Multilingual support is an **opt-in profile**, not a default change — the
English stack remains the validated baseline and regression control.

```
Settings.profile ∈ { english (default), multilingual, multilingual_quality }
   │
   └─ profile presets fill model choices ONLY when fields are at english defaults
      (explicit VFR_ overrides always win)

 english             multilingual                       multilingual_quality
 ─────────           ─────────────                      ─────────────────────
 all-MiniLM-L6-v2    intfloat/multilingual-e5-small     BAAI/bge-m3
 384 dim             384 dim (+query:/passage: prefix)  1024 dim
 EN reranker         jina-reranker-v2-base-multilingual bge-reranker-v2-m3
 EN stemmer          no-stemmer (Unicode-safe BM25)     no-stemmer
```

**Principles honored**
- **Model-driven multilinguality** — no per-language branching in retrieval.
- **Embedder-specific logic encapsulated** — E5 `query:`/`passage:` prefixes live entirely in the embedder adapter; callers pass semantic `input_type`, never model awareness.
- **NFC normalization** (not NFKC) at every text boundary: query sanitize, BM25 corpus, loader output, identity hashing.
- **Language detection is advisory only** (`src/language.py`, `langid`) — drives an expansion prompt hint, never retrieval routing.
- **OCR languages** are operator-installed packs, never auto-downloaded.
- **Encoding-safe end-to-end** — verified across JSON, pickle cache, RetrievalTrace, Prometheus, structured logs, SSE.

**Validated**: English Recall@5 = 1.0 on both profiles (zero regression);
multilingual profile improves monolingual MRR 0.889 → 1.0. Snapshots in
`experiments/artifacts/` (JSON + CSV, versioned).

---

## 8.5 Configurable platform architecture  [Phase 12]

Phase 12 adds a runtime-configuration layer **on top of** the immutable
env-driven `Settings`, so models and indexes change without a restart while the
validated baseline stays byte-identical under defaults.

**Provider abstraction (`src/providers/`).** One `ModelProvider` interface over
offline (Ollama) + online (OpenAI/Anthropic/Gemini/Groq/OpenRouter) backends.
`generate`/`stream_generate` keep the legacy `OllamaClient` signatures, so
`pipeline.llm` can be any provider with no call-site changes. The frontend only
ever sees `ProviderCapabilities`/`ProviderModel` JSON — never URLs or SDKs. API
keys live solely in the backend `SecretStore` (Fernet-encrypted, `0600`,
redacted) and are injected at construction by the registry factory.

**Runtime config (`src/runtime_config.py`).** Splits settings into two classes:
*live-query* (provider/model, reranker, expansion, RRF/k/candidates, cache) which
apply to the running pipeline immediately, and *index-construction* (embedding
model, chunking, backend, FAISS topology) which are **staged only** — a change
here never silently rebuilds an index. Persisted to `var/runtime_config.json`.

**Named indexes (`src/indexing/`).** `IndexProfile` + `IndexRegistry` make
indexes first-class named entities (active pointer, persisted to
`var/index_registry.json`; data under `indices/named/<name>/`). `IndexManager`
does create/load/switch/delete/export/import. The FAISS **recipe** layer turns a
`(recipe, params, dim)` into a validated `index_factory` string — validated
statically *and* by constructing the empty index in FAISS — with memory/latency/
training estimates. The **compatibility validator** grades a config change
BLOCKING (different vector space → new index) / REBUILD (same vectors, rebuild) /
INFO, driving the "create a new index?" safety UX. Nothing mutates silently.

**Background jobs (`src/jobs/`).** A thread-pool `JobRegistry` runs FAISS
builds/training/benchmarks off the HTTP worker with progress, cooperative
cancellation, replayable SSE streaming, and bounded history. Index creation and
benchmarking are submitted as jobs; the API returns `202 + job_id` and the UI
streams `/jobs/{id}/stream`.

All of this is additive: new routes under `/api/v1/models`, `/config/runtime`,
`/indexes`, `/jobs`; existing routes and default behavior are unchanged.

```
Settings (immutable, env)  ──►  RuntimeConfigStore (mutable, var/)
                                   ├─ live  ──► pipeline.apply_live_settings()
                                   └─ staged index ──► IndexManager (build job)
ModelProvider registry ──► pipeline.set_chat_provider()   SecretStore (var/, encrypted)
```

---

## 9. Deployment topology

| Mode | Backend | Frontend | Cache | Notes |
|---|---|---|---|---|
| Local dev | `uvicorn --reload` :8000 | `npm run dev` :3000 | memory | default |
| Local production | `uvicorn --workers N` | `next build && start` | Redis (shared) | single host |
| Hosted lightweight | user-run backend | Vercel build | memory/Redis | per-user backend URL via Settings/localStorage |
| Desktop | Python sidecar | `OUTPUT=export` static `./out` | memory | Tauri/Electron shell |

CORS allows `http://localhost:3000` by default (`VFR_API__CORS_ORIGINS`).

---

## 10. Scaling boundaries

| Dimension | Comfortable | Needs attention | Action |
|---|---|---|---|
| Corpus size | ≤1M chunks (FAISS HNSW, 384-dim) | >1M | tune HNSW M/ef; consider IVF-PQ |
| Vector dim | 384 (CPU) | 1024 (bge-m3) | GPU/MPS recommended |
| Concurrent retrieval | threadpool-bound, lock-free | very high QPS | multiple workers + Redis |
| Ingestion | synchronous, lock-serialized | large batches / huge files | async job queue (future) |
| BM25 | rebuilt per ingest | >100k chunks frequent re-ingest | incremental BM25 (bm25s supports it) |
| Metrics retention | rolling window + ring buffer | long-horizon history | external Prometheus + JSONL trace log |
| Cache memory | LRU-bounded | many large results | Redis with maxmemory policy |

---

## 11. Known limitations

- **No auth / multi-user** — single-tenant local-first only.
- **Synchronous ingestion** blocks an HTTP worker for large corpora.
- **No rate limiting** — public deployments need hardening (file-size guard is the only current protection).
- **No conversation persistence** — chat is single-turn, in-memory.
- **CJK BM25** uses no-stemmer whitespace; character-n-gram tokenization deferred.
- **Pickle cache codec** assumes trusted Redis (unsafe with untrusted shared instances).
- **Process-local metrics** — cross-worker aggregation delegated to Prometheus.
- **Microbenchmark thresholds** assume an unloaded machine (can flake under parallel full-suite load).
- **Frontend i18n** (UI string translation) not implemented; rendering is Unicode-safe but chrome is English.
- **`multilingual_quality` preset** wired but not benchmarked here (needs GPU).

---

## 12. Future extension points

- **Async / queued ingestion** with progress events — loaders are already isolated from indexing; the architecture doesn't block it.
- **Incremental indexing & dedup** — stable chunk_ids make upsert-by-id and content-hash dedup mechanical.
- **Reranker output caching** — key builder exists; wiring is one step.
- **Per-namespace cache metrics** — emit `cache_ops_total{namespace, op}` from each wrapper.
- **Auth + multi-user** — request_id propagation and per-doc IDs already support tenant prefixing.
- **Conversation persistence** — `useStreamingAsk` state is serializable; lift into a store + `/conversations` endpoint.
- **Agentic events** — SSE taxonomy extends cleanly with `tool_call`/`tool_result`.
- **JSONL trace persistence** — wrap the recent-traces ring buffer to also append to disk.
- **CJK tokenizer plugin** + multilingual reranker benchmarking.

---

## 13. Module map (where things live)

| Concern | Module(s) |
|---|---|
| Config / profiles | `src/config.py` |
| Identity (doc/chunk IDs) | `src/identity.py` |
| Chunking | `src/chunker.py` |
| Embedding (+prefix adapter) | `src/embedder.py` |
| Vector stores | `src/vector_store.py` (Chroma), `src/faiss_store.py` (FAISS), factory in `vector_store.py` |
| BM25 | `src/bm25_retriever.py` |
| Fusion | `src/rrf.py`, `src/hybrid_retriever.py` |
| Reranking | `src/reranker.py` |
| Query expansion | `src/query_expansion/` (base, multi_query, hyde, pipeline) |
| Language detection | `src/language.py` |
| LLM client | `src/llm_client.py` |
| Model providers (offline+online) | `src/providers/` (base, ollama, online_base, openai/anthropic/gemini/groq/openrouter, registry, secrets) |
| Runtime config (live vs staged) | `src/runtime_config.py` |
| Named indexes / recipes / compat | `src/indexing/` (profile, registry, manager, recipes, compatibility, benchmark) |
| Background jobs | `src/jobs/` (base, registry, index_jobs) |
| Loaders | `src/loaders/` (base, registry, per-format) |
| Cache | `src/cache/` (memory, redis, null, safe, codec, keys, factory, caching_embedder, caching_expansion) |
| Observability | `src/observability/` (primitives, registry, prometheus_export), `src/retrieval_trace.py` |
| API | `src/api/` (app, dependencies, middleware, errors, schemas, routes/) |
| Orchestration | `src/rag_pipeline.py` |
| Contracts | `src/interfaces.py` |
| Frontend | `frontend/` |
| Benchmarks & artifacts | `experiments/multilingual_*.py`, `experiments/artifacts/` |
```
