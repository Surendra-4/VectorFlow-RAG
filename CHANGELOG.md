# Changelog

Local release summary for the VectorFlow-RAG platform build-out
(Phases 3.5 → 11). All work is local; see `docs/ARCHITECTURE.md` for the
full system reference.

Test totals quoted are from dual-backend runs (ChromaDB default + FAISS
subset) at each phase boundary. The hard gate throughout: **English
retrieval quality must not regress** — held at every phase.

---

## [0.2.0] — Foundation → Multilingual platform

### Phase 0 — Foundation
- Centralized `Settings` (pydantic-settings, `VFR_` env prefix, nested `__`).
- Structured logging (`text`/`json`, rotating file handler), `get_logger`.
- Swap-in Protocols in `src/interfaces.py` (vector store, cache, reranker, embedder, LLM).
- Cross-platform path fixes; auto device detection (cuda → mps → cpu) in the embedder.
- `.env.example` documenting every knob.

### Phase 1 — RRF + cross-encoder reranking
- Replaced alpha-weighted fusion with **Reciprocal Rank Fusion** (`src/rrf.py`, `k=60`).
- Symmetric per-modality candidate pools; results expose `rrf_score`, `vector_rank`, `bm25_rank`.
- Optional, lazy, config-driven **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2` default).
- Retrieval-quality + ranking-consistency + latency tests.

### Phase 3 — FAISS HNSW vector store
- `FAISSVectorStore` (HNSW default; flat / IVF available), persistent index + JSON sidecar, ID-first internals.
- `make_vector_store()` factory; backend selectable via `VFR_VECTOR_STORE__BACKEND`. ChromaDB retained.
- Fixed a pre-existing ChromaDB persistence-after-reload bug (empty-store guard checked the in-memory shadow).
- Recall + latency parity tests; FAISS-HNSW ~6.5× faster than ChromaDB at 2k vectors.

### Phase 3.5 — Chunk & document identity
- Content-derived `doc_id` + `chunk_id` (`src/identity.py`); NFC normalization before hashing.
- Provenance metadata on every chunk: `document_name`, `source_path`, `page_number`, `chunk_index`.
- Vector stores return `ids`; BM25 accepts dict-corpus with `chunk_id`; RRF joins on `chunk_id` (text fallback).
- Fixes duplicate-content collapse (identical boilerplate across docs now individually attributable).

### Phase 4 — Multi-format ingestion
- `src/loaders/` package + registry (extension/MIME dispatch): text, markdown, json, csv/tsv, xlsx, sqlite, pdf, docx, image OCR.
- `RAGPipeline.ingest_files()` with per-file failure tolerance, max-file-size guard, `chunk_index_offset` for multi-page docs.
- Provenance propagated end-to-end (PDF page numbers, XLSX sheet names, SQLite table names).

### Phase 5 — Query expansion & retrieval enhancement
- `src/query_expansion/`: MultiQueryExpander + HyDEExpander + composable pipeline.
- Defensive parsing/sanitization, dedup, length caps, per-strategy timeout, graceful degradation.
- `RetrievalTrace` observability substrate (original/expanded queries, HyDE docs, per-strategy candidates, rerank stats).
- Prompt-injection + multilingual-surface + integration tests.

### Phase 6 — Redis caching & performance
- Cache backends: `MemoryCache` (LRU+TTL, thread-safe), `RedisCache` (lazy import, graceful fallback), `NullCache`, all behind `SafeCache`.
- Caching wrappers: `CachingEmbedder` (per-text + intra-batch dedup), `CachingExpansionPipeline`.
- **Structural invalidation** via corpus fingerprint (no `invalidate()` to misuse).
- Full-retrieval cache measured **322× warm speedup**; cache stats surfaced in `RetrievalTrace`.

### Phase 8 — FastAPI service layer
- `src/api/`: app factory, DI pipeline singleton, correlation-id + timing middleware, structured errors, OpenAPI.
- Endpoints: health, status, ingest (text/files/paths), search, ask (+ SSE streaming), cache, documents, index admin.
- Read-concurrent retrieval; lock-serialized ingestion; versioned `/api/v1`.

### Phase 9 — Monitoring & observability
- `src/observability/`: thread-safe Counter / LabeledCounter / Gauge / Histogram (rolling window) / RingBuffer + `MetricsRegistry`.
- Collector hooks in middleware + pipeline + ingest + SSE; Prometheus text exporter (no client dependency).
- Dashboard endpoints + recent-traces buffer; ~0.3–1 µs/op overhead, bounded retention, bounded cardinality.

### Phase 10 — Next.js frontend
- Next.js 14 App Router + TypeScript + Tailwind; no global-state library, no UI kit, no cloud deps.
- Typed API client mirroring backend schemas; fetch-based SSE parser; polling hooks.
- Pages: chat (streaming + citations), search, ingest (drag-drop + text), documents, dashboard, traces, settings.
- Static-export ready (desktop packaging); provenance-first UX; dark/light/system theme.

### Phase 11 — Multilingual support (opt-in profile)
- Profiles: `english` (default), `multilingual` (e5-small + jina reranker + no-stemmer BM25), `multilingual_quality` (bge-m3 + bge-reranker-v2-m3).
- E5 `query:`/`passage:` prefixes encapsulated in the embedder adapter (`input_type` is semantic, not model-aware).
- NFC normalization at all text boundaries; CJK sentence terminators in the chunker; Unicode-safe BM25.
- OCR language config (operator-installed packs); advisory `langid` detection (never retrieval routing).
- Frontend `dir="auto"` + documented font stack for RTL/CJK rendering.
- Multilingual golden corpus (7 languages), normalization fixtures, encoding-safety audit, versioned benchmark artifacts.
- **English Recall@5 = 1.0 on both profiles (zero regression)**; multilingual monolingual MRR 0.889 → 1.0.

---

## Cleanup pass (pre-finalization)
- Removed a stale pre-fix benchmark snapshot that would corrupt regression comparison.
- Moved unused deps (`mlflow`, `dagshub`, `pandas`, `dataclasses-json`) to a commented optional block.
- Added `*.tsbuildinfo` to frontend `.gitignore`; added `.claude/` to repo `.gitignore`.
- Verified: 62 backend modules import cleanly (no cycles); no test-helper leakage into `src/`; no stray temp/debug files; backend↔frontend config consistent.

## Verification snapshot
- Backend: ~675 tests (ChromaDB full suite) + 174 (FAISS subset); 1 timing microbenchmark flakes only under parallel-suite CPU load (0.658 µs/op in isolation).
- Frontend: 62 Vitest tests; clean `tsc --noEmit`; successful production build.
- Benchmark artifacts: `experiments/artifacts/multilingual_benchmark_*.json` + `.csv` (schema-versioned).
