# Changelog

Release summary for the VectorFlow-RAG platform build-out
(Phases 0 → 14). Local-first; see `docs/ARCHITECTURE.md` for the full system
reference and `DEPLOYMENT.md` for the free hosted setup.

Test totals quoted are from dual-backend runs (ChromaDB default + FAISS
subset) at each phase boundary. The hard gate throughout: **English
retrieval quality must not regress** — held at every phase.

---

## [2.0.0] — Authentication, multi-user & free deployment (Phase 14)

Turns the single-tenant local platform into a deployable, multi-user product —
without changing default/anonymous behavior (auth is opt-in via
`VFR_AUTH__REQUIRED`).

### Phase 14 — Accounts, auth & per-user stats
- **Email/password** (bcrypt) + **Google/GitHub OAuth** (authorization-code flow
  over `requests`, no SDK); HS256 **JWT** sessions; opaque password reset.
- **SQLAlchemy** DB layer (`users`, `user_stats`) over `DATABASE_URL` — Postgres
  in prod, SQLite locally. **Ingested documents are never stored** — only
  accounts + per-user counters, with a self-service reset.
- Data endpoints gate behind `require_user_if_enabled` (only when
  `auth.required`); `get_optional_user` stays lazy, so the anonymous path is
  byte-identical to pre-Phase-14. Provider API keys live in a Fernet-encrypted
  server-side store.
- Frontend: Render-inspired split-screen auth, route gating (`AppFrame`),
  bearer-token client with 401→logout, per-user dashboard stats.

### Professional UI redesign
- Aurora design system (CSS-var palette, `next/font`, motion primitives,
  `Constellation` canvas hero), glass components — applied across every page.

### Free deployment (₹0)
- Frontend on **Vercel**, backend self-hosted behind a **free tunnel** (ngrok
  static domain / Cloudflare Quick Tunnel — no domain, no card), accounts/stats
  in free **Postgres** (Neon/Supabase) or local SQLite; answer model via local
  Ollama or a hosted key. Runbook: `DEPLOYMENT.md`.

### Post-release maintenance
- **perf**: index builds reuse ingest-time embeddings instead of re-embedding
  the whole corpus (`get_embeddings` on both vector backends).
- **fix**: macOS torch+faiss OpenMP segfault on threaded IVF/PQ training
  (`OMP_NUM_THREADS=1` on Darwin); small-corpus build/benchmark hardening.
- **fix**: streaming `ask` attaches the bearer token; ngrok browser-warning header.
- **test**: suite made hermetic against a local `.env` (auth + `DATABASE_URL`).
- **chore**: removed the legacy Streamlit app + workflow and stale committed
  index fixtures; added the missing `python-multipart` / `cryptography` / `langid`
  dependencies; de-duplicated `.gitignore`.

### Verification
- 975 backend tests (904 functions / 58 files) + 79 frontend ≈ **1,050 total**;
  dual-backend; English retrieval parity preserved. Real ablations versioned in
  `experiments/artifacts/` (FAISS recipes + multilingual profile).

---

## [0.4.0] — Live named-index switching (Phase 13)

Closes the loop on Phase 12's named indexes: an index you build/benchmark can
now actually **serve live retrieval**, switched at runtime — safely.

### Phase 13a — Pipeline activation with identity retention
- `RAGPipeline` retains chunk records (text/chunk_id/metadata) via
  `iter_chunk_records()`, so a named index can be built with *identical*
  identity + provenance — the precondition for a correct hot-swap (RRF joins on
  `chunk_id`; citations read metadata).
- `activate_named_index(name, store)` / `activate_default_index()` swap only the
  vector half of hybrid retrieval, rebuilding `HybridRetriever` over the new
  store while keeping BM25 (same `chunk_id` set). `_default_vector_store` handle
  retained at ingestion; re-ingestion resets activation. `active_index_name`
  folded into the retrieval-cache key.

### Phase 13b — Compatibility-gated switch API
- `create_index` builds from the live chunk records (real ids + metadata).
- `POST /indexes/{name}/switch` runs a switch-compatibility check (live embedder
  model + dim + corpus fingerprint vs the index); on success activates it in the
  pipeline (ingest-lock-serialized) + sets the registry active pointer; on
  mismatch returns a structured **409** (`code=index_incompatible`) with the full
  report in `details`. `POST /indexes/activate-default` reverts. `/status`
  exposes `active_index_name`.

### Phase 13c — Frontend active-index UX
- IndexesTab shows which index serves live retrieval, a "Use default retrieval"
  revert, and surfaces the 409 compatibility report inline ("create a new
  index?") instead of a raw error.

### Verification
- Default path (`active_index_name=None`) is byte-identical — English retrieval
  parity preserved. New tests: 6 pipeline-activation + 3 switch/compat API + 4
  frontend component. Frontend suite 74 passed; tsc clean; build OK.

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

## [0.3.0] — Configurable retrieval platform (Phase 12)

Transforms the app into a runtime-configurable platform: switch chat models
(local **or** API) without a restart, manage offline models, and build/compare/
switch named FAISS indexes safely. **Default settings behave identically to
0.2.0** — every new subsystem is opt-in and the English-parity gate held.

### Phase 12a–b — Model provider abstraction
- `src/providers/`: `ModelProvider` ABC + `ProviderModel`/`ProviderCapabilities`/
  `ConnectionStatus` metadata; `ChatModelConfig`/`EmbeddingModelConfig`/
  `RerankerModelConfig` (no `api_key` field by design); typed `ProviderError`
  hierarchy. `OllamaProvider` (chat delegates to `OllamaClient` for byte-parity)
  + offline catalog/install/delete. Online providers over `requests` (no SDKs):
  OpenAI/Groq/OpenRouter (OpenAI-compatible base), Anthropic, Gemini.
- `SecretStore`: backend-only, Fernet-encrypted at rest (graceful fallback),
  `0600` perms, redaction — keys never reach the frontend or logs.

### Phase 12c–d — Runtime config + model-management API
- `RuntimeConfigStore`: mutable layer over the immutable env `Settings`,
  separating **live-query** settings (applied immediately) from
  **index-construction** settings (staged — never a silent rebuild).
- `RAGPipeline.set_chat_provider()` / `apply_live_settings()` hot-swap the
  provider + live knobs; settings deep-copied lazily so the singleton/baseline
  is untouched.
- API: `/models/*` (providers, offline installed/catalog/install-SSE/delete,
  online key mgmt + validate + list, active, select), `/config/runtime/*`.
  `ProviderError` → structured 401/404/503/502; keys never echoed.

### Phase 12e–g — Named indexes, recipes, compatibility
- `IndexProfile` + `IndexRegistry` (named indexes, active pointer, persisted)
  + `IndexManager` (create/load/switch/delete/export/import; zip-slip guarded).
- FAISS recipe catalog (Flat/HNSW/IVF/PQ + IVF-PQ/IVF-HNSW/HNSW-PQ/OPQ-IVF-PQ/
  IMI/IndexRefineFlat/Multi-D-ADC) with validated params; `validate_recipe`
  checks statically **and** by constructing in FAISS; memory/latency/training
  estimates. `FAISSVectorStore` gains `factory_string` + nprobe (legacy path
  unchanged). `check_compatibility` grades changes BLOCKING/REBUILD/INFO and
  drives the "create a new index?" UX — never mutates silently.

### Phase 12h–i — Background jobs + benchmarking
- `src/jobs/`: `Job`/`JobContext`/`JobRegistry` over a thread pool — progress,
  cooperative cancellation, replayable SSE streaming, bounded history. FAISS
  builds/training run off the HTTP worker.
- Benchmarking: Recall@K/MRR/latency/QPS/size vs an exact Flat reference;
  multi-recipe compare; schema-versioned artifacts. API: `/indexes/*`
  (recipes/validate/list/create-job/switch/delete/compatibility/benchmark) +
  `/jobs/*` (list/get/stream/cancel).

### Phase 12j — Observability sweep
- New bounded-cardinality metrics: provider ops/chat/errors, model installs/
  switches, index jobs/builds (+duration), index switches, benchmark runs.
  Active-index name reported as a **state value**, never a label. Prometheus
  exporter + `/metrics/models` + `/metrics/indexes` views.

### Phase 12k — Frontend settings dashboard
- Typed client (`models`/`runtimeConfig`/`indexes`/`jobs` + SSE consumer),
  `useJobProgress` hook, UI primitives (Select/Toggle/Tabs/ProgressBar).
- Tabs: **Models** (local Ollama install/select OR closed-source via API key),
  **Retrieval** (live toggles + fusion knobs), **Indexes** (named-index list +
  health, Basic/Advanced builder with live validation, compatibility warnings,
  build progress, benchmark comparison). Theme/responsive/a11y preserved.

### Verification
- Backend: full suite on ChromaDB + FAISS subset; ~190 new Phase-12 tests.
  **English Recall@5 unchanged under default settings.** Existing API contracts
  unchanged (additive routes only).
- Frontend: 70 Vitest tests (8 new); clean `tsc --noEmit`; production build OK.

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
