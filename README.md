# VectorFlow-RAG

**A production-grade, local-first Retrieval-Augmented Generation platform** — hybrid retrieval with Reciprocal Rank Fusion, cross-encoder reranking, LLM query expansion, multi-format ingestion, provenance-rich answers, multilingual support, pluggable offline **and** online model providers, named/benchmarkable FAISS indexes with live switching, multi-user authentication, full observability, and a typed Next.js frontend. Runs end-to-end on your own machine — **and deploys for ₹0** (your Mac + a free tunnel + Vercel).

![Python](https://img.shields.io/badge/python-3.11-blue)
![Frontend](https://img.shields.io/badge/frontend-Next.js%2014-black)
![Tests](https://img.shields.io/badge/tests-1050%2B-success)
![Backends](https://img.shields.io/badge/vector%20backends-Chroma%20%2B%20FAISS-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Style](https://img.shields.io/badge/code%20style-black-000000)

> **TL;DR for reviewers:** this is not a notebook demo. It's a layered system — a typed FastAPI service, two interchangeable vector backends (ChromaDB / FAISS) behind one factory, an 11-recipe FAISS index lab with validation + benchmarking + compatibility-gated **live index switching**, a model-provider abstraction spanning **local Ollama and 5 hosted APIs** (keys encrypted at rest, never exposed to the browser), background jobs with streaming progress, a cache layer with structural invalidation, query-expansion + reranking stages, end-to-end provenance (stable content-derived chunk/document IDs), an opt-in multilingual profile that *provably* doesn't regress English retrieval, **email/password + Google/GitHub auth with per-user stats**, and a React frontend that depends only on the public API contract. **1,050+ automated tests**, dual-backend validated. Deep rationale in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md); free deployment in [`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## Table of contents
- [Why this exists](#why-this-exists)
- [Feature matrix](#feature-matrix)
- [Architecture at a glance](#architecture-at-a-glance)
- [The retrieval pipeline](#the-retrieval-pipeline)
- [Technology choices & reasoning](#technology-choices--reasoning)
- [Measured results & ablations](#measured-results--ablations)
- [Engineering trade-offs](#engineering-trade-offs-made-deliberately)
- [Quickstart](#quickstart)
- [Deployment (₹0)](#deployment-0)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Limitations & future work](#limitations--future-work)

---

## Why this exists

Keyword search misses synonyms; vector search misses exact terms; most "RAG in 50 lines" demos are black boxes you can't debug, can't deploy, and can't trust. Real retrieval systems must juggle **recall, precision, latency, provenance, multi-tenancy, and operability at the same time** — without shipping your documents to a third-party API.

VectorFlow-RAG combines **BM25 lexical matching** and **dense vector search**, fuses them with **Reciprocal Rank Fusion**, optionally refines with a **cross-encoder reranker** and **LLM query expansion**, and returns answers grounded in **source-attributed chunks**. The answer model can be a fully local Ollama model **or** a hosted provider — your choice, switchable at runtime. Every retrieval is observable down to per-stage latency and per-modality rank, and every account's activity is tracked independently.

---

## Feature matrix

| Capability | What it does | Where |
|---|---|---|
| **Hybrid retrieval (RRF)** | Rank-based fusion of BM25 + dense vectors; scale-invariant, no per-corpus tuning | `src/rrf.py`, `src/hybrid_retriever.py` |
| **Cross-encoder reranking** | Optional second-stage precision refinement over the fused pool | `src/reranker.py` |
| **Query expansion** | Multi-query rewriting + HyDE (hypothetical document embeddings) via LLM | `src/query_expansion/` |
| **Dual vector backends** | ChromaDB (default) and FAISS, behind one factory; switch by config | `src/vector_store.py`, `src/faiss_store.py` |
| **Named index lab** | 11 FAISS recipes (Flat/HNSW/IVF/PQ/OPQ/IMI…), validated `index_factory` strings, memory/latency/training estimates, in-app **benchmarking** | `src/indexing/` |
| **Live index switching** | Promote a named index to serve live retrieval at runtime — **compatibility-gated** (same embedder + corpus), preserves chunk-id join + provenance | `src/indexing/`, `src/rag_pipeline.py` |
| **Model providers** | One interface over **offline Ollama** + **online OpenAI / Anthropic / Gemini / Groq / OpenRouter**; keys Fernet-encrypted server-side, never sent to the browser | `src/providers/` |
| **Background jobs** | Thread-pool job registry: index builds/benchmarks off the HTTP worker, streaming progress (SSE), cooperative cancellation | `src/jobs/` |
| **Authentication & multi-user** | Email/password (bcrypt) + Google/GitHub OAuth (auth-code flow), JWT sessions, per-user statistics with self-service reset | `src/auth/`, `src/db/` |
| **Stable identity & provenance** | Content-derived `doc_id`/`chunk_id`; every result carries document name, page, chunk index, per-modality ranks | `src/identity.py` |
| **Multi-format ingestion** | PDF, DOCX, TXT, Markdown, CSV/TSV, XLSX, SQLite, JSON, images (OCR) | `src/loaders/` |
| **Caching** | Memory / Redis / null behind a fail-safe wrapper; structural invalidation via corpus fingerprint | `src/cache/` |
| **Observability** | Thread-safe metrics registry, per-stage retrieval traces, Prometheus export, dashboard endpoints | `src/observability/`, `src/retrieval_trace.py` |
| **Multilingual (opt-in)** | `english` / `multilingual` / `multilingual_quality` profiles; English baseline provably preserved | `src/config.py`, `src/language.py` |
| **FastAPI service** | Versioned REST + SSE streaming, structured errors, correlation IDs, OpenAPI | `src/api/` |
| **Next.js frontend** | Auth screens, chat (streaming), search, ingest (drag-drop), documents, dashboard, traces, settings (models/indexes/jobs) | `frontend/` |

---

## Architecture at a glance

```
Next.js on Vercel ──HTTPS/SSE (typed contract, JWT)──► FastAPI on your Mac (/api/v1)
        │                                                   │  middleware: request-id + timing + metrics
   (free tunnel: ngrok)                                     │  auth gate (JWT) · CORS
                                                            ▼
                                                    RAGPipeline (orchestrator)
   chunk → embed → index → retrieve → fuse(RRF) → rerank? → generate(Ollama | hosted API)
        │         │          │            │           │              │
   loaders/   embedder   vector store  bm25      reranker      model provider
   (9 fmts)   (+cache)   chroma/faiss  retriever  (optional)   (offline/online)
        └─── cross-cutting: config(profiles) · cache · query_expansion · retrieval_trace ·
             observability · named-index registry · background jobs · auth/DB · logging
```

Design invariants:
- **Frontend ↔ backend coupling is exactly one typed schema file.** The UI never imports backend code.
- **Retrieval is lock-free and concurrent; ingestion is serialized.**
- **Every swappable component sits behind a `Protocol`** (`src/interfaces.py`) — vector store, cache, reranker, embedder, LLM/provider, loader, expansion strategy.
- **One configuration entry point** (`src/config.py`, env prefix `VFR_`), with a runtime-config layer on top for hot model/index changes.
- **Secrets never reach the client** — provider API keys live only in an encrypted server-side store; the browser sees `configured: bool` + a redacted hint.

Full diagram + per-lifecycle walkthroughs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## The retrieval pipeline

```
query
 ├─ sanitize (NFC normalize · strip control chars · length cap)
 ├─ full-retrieval cache lookup ── HIT ─► return (skips everything below)
 ├─ query expansion (optional): multi-query variants + HyDE docs
 ├─ per query/HyDE input → { dense vector search , BM25 search }
 ├─ Reciprocal Rank Fusion  RRF(d) = Σ 1/(k + rank_i(d)),  k=60   (joins on chunk_id)
 ├─ cross-encoder rerank (optional) over the fused pool
 └─ results: text + chunk_id + doc_id + document_name + page_number
            + hybrid/RRF score + vector_rank + bm25_rank + rerank_score
```

**Why RRF over the more common α-weighted fusion** (`score = α·vec + (1−α)·bm25`): RRF is *rank-based*, so it needs no score normalization across two retrievers whose scores live on totally different scales (unbounded BM25 vs. cosine ∈ [0,1]). It has one well-studied constant (`k=60`) instead of a per-corpus `α` to tune, and it naturally rewards documents found by *both* modalities. The project began on α-fusion and migrated to RRF — documented in the changelog.

---

## Technology choices & reasoning

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| Answer model | **Ollama (local) or 5 hosted providers** | Local = zero cost + full privacy + offline; hosted = quality when you want it. One interface, switch at runtime. |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2` default) | 384-dim, ~80 MB, CPU/MPS-viable. Auto-detects CUDA→MPS→CPU. |
| Sparse retrieval | **BM25 (`bm25s`)** | Battle-tested lexical baseline; catches exact/rare terms vectors miss. |
| Fusion | **Reciprocal Rank Fusion** | Scale-invariant, tuning-light, intersection-aware (see above). |
| Reranker | **cross-encoder/ms-marco-MiniLM-L-6-v2** (default) | ~80 MB, biggest precision lever per MB. Optional. |
| Vector store | **FAISS** + **ChromaDB** | FAISS = 11 tunable recipes + benchmarking; Chroma = zero-setup default. One factory, swap by config. |
| Auth | **bcrypt + JWT + manual OAuth 2.0** | No heavyweight auth SDK; OAuth code flow over `requests`; keys/tokens never leave the server. |
| Database | **SQLAlchemy → Postgres / SQLite** | Accounts + per-user stats only (never your documents). SQLite locally, free Postgres in prod. |
| API | **FastAPI + Uvicorn** | Async-friendly, OpenAPI out of the box, Pydantic schemas double as the frontend contract. |
| Frontend | **Next.js 14 + TypeScript + Tailwind** | App Router, typed API client, custom design system, no global-state library needed. |
| Streaming | **SSE (fetch + ReadableStream)** | Simpler and more debuggable than WebSockets for one-shot token streaming. |
| Config | **pydantic-settings + runtime-config store** | Typed env baseline, plus hot live/staged changes without a restart. |

---

## Measured results & ablations

> **Integrity note:** every number below was measured on this codebase (Apple Silicon, CPU/MPS). Golden retrieval sets are intentionally small and diverse — **regression gates and directional signals, not BEIR-grade leaderboard scores**. Large-scale benchmark replay (MS-MARCO/BEIR) is listed under future work. Nothing here is extrapolated; artifacts are versioned in [`experiments/artifacts/`](experiments/artifacts/).

### Test suite
- **975 backend tests** collected (pytest; 904 test functions across 58 files), validated on **both** vector backends (ChromaDB + FAISS).
- **79 frontend tests** (Vitest + Testing Library), clean `tsc --noEmit`, successful production build.
- **Total: 1,050+ automated tests.**

### Ablation 1 — FAISS index recipes (real run, n = 3,489 chunks, dim = 384, k = 10)
The in-app benchmark scores each recipe against an exact-Flat ground truth — the classic recall/latency/memory trade-off, on a real corpus (two ML textbooks). Generated by the Settings ▸ Indexes ▸ Benchmark panel:

| Recipe | Recall@10 | MRR | p50 latency | QPS | Index size | Notes |
|---|---|---|---|---|---|---|
| **flat** (exact) | **1.000** | 1.000 | 0.132 ms | 6,140 | 5.5 MB | exact baseline / ground truth |
| **hnsw** | 0.998 | 0.990 | 0.092 ms | 7,654 | 6.5 MB | near-exact, graph index, no training |
| **ivf** | 0.954 | 1.000 | 0.045 ms | 17,942 | 5.7 MB | ~3× the QPS for ~5% recall |
| **ivf_pq** | 0.536 | 0.913 | 0.041 ms | 14,569 | **0.75 MB** | ~7× smaller; recall trade for memory |

Reading: **HNSW reproduces exact search** (0.998 R@10) with no training step; **IVF** triples throughput for a small recall cost; **IVF-PQ** compresses the index ~7× (5.5 MB → 0.75 MB) — the lever you pull when memory, not recall, is the constraint. `efSearch`/`nprobe` are live query-time knobs to move along this curve without rebuilding.

### Ablation 1b — scaling to 100k vectors (384-dim, k = 10, NDCG@K added)
Same harness on a **100,000-vector** corpus (clustered synthetic, 200 queries) — the index layer stays **sub-millisecond**:

| Recipe | Recall@10 | NDCG@10 | p95 latency | QPS | Index size |
|---|---|---|---|---|---|
| **HNSW** | 0.974 | **0.984** | **0.14 ms** | 7,744 | 185 MB |
| **IVF** (nprobe-tuned) | **1.000** | **1.000** | 0.37 ms | 3,049 | 159 MB |
| flat (exact) | 1.000 | 1.000 | 3.99 ms | 283 | 158 MB |

HNSW serves 100k vectors at **p95 ≈ 0.14 ms / NDCG@10 0.98**, ~28× faster than exact at ~97% recall; IVF matches exact quality at ~10× the throughput. (IVF-PQ on this distribution needs param tuning — its default recall is poor, an honest artifact in the JSON.)

### Ablation 2 — multilingual profile (golden corpus, 7 languages, k = 5)
The hard gate: *English retrieval must not regress when multilingual support is added.* It held.

| Profile | Embedder | English R@5 | Monolingual MRR | Cross-lingual MRR | Query p50 |
|---|---|---|---|---|---|
| `english` (default) | all-MiniLM-L6-v2 | **1.00** | 0.889 | 1.00 | 34.3 ms |
| `multilingual` | multilingual-e5-small | **1.00** | **1.00** | 1.00 | 21.0 ms |

English recall identical; the multilingual model *improves* monolingual ranking (MRR 0.889 → 1.00) across a mixed en/fr/de/es/zh/ar/ru corpus while preserving the English control.

### Cache impact (golden corpus, full-retrieval cache)
Cold retrieval **12.4 ms** → warm cache hit **0.04 ms** → **~322× speedup** on repeat queries — the biggest single latency win once query expansion (which adds LLM calls) is enabled.

### Observability overhead (per-op, isolated)
`Counter.inc` 0.34 µs · `LabeledCounter.inc` 1.07 µs · `Gauge.set` 0.27 µs · `Histogram.observe` 0.66 µs · `RingBuffer.append` 0.24 µs — telemetry adds <0.1% to any path doing real retrieval work.

---

## Engineering trade-offs (made deliberately)

| Decision | Trade-off accepted | Why |
|---|---|---|
| Self-host backend behind a **free tunnel** vs. paid cloud | Backend is up only while your machine is | The ML stack (PyTorch + sentence-transformers + FAISS) needs ~1 GB RAM — more than free cloud tiers; your hardware is free and already there |
| Multilingual as **opt-in profile**, not default | Users flip a config flag for non-English | Cross-lingual spaces can weaken monolingual precision; keeping English as the validated control eliminates regression risk |
| **RRF** over tuned α-fusion | Gives up per-corpus score weighting | Robustness and zero-tuning beat marginal tuned gains across heterogeneous corpora |
| **Cap `OMP_NUM_THREADS=1` on macOS** | Single-threaded OpenMP on Darwin | PyTorch + faiss each bundle an OpenMP runtime; training a FAISS IVF/PQ index from a job thread with torch loaded *segfaults* on macOS. Capping avoids it at zero cost to MPS (GPU) embedding; Linux keeps full threading |
| **Synchronous ingestion** (lock-serialized) | Large batches block an HTTP worker | Correctness/simplicity now; loaders are isolated from indexing so async/queued ingestion is a non-breaking add |
| **Process-local metrics** | No built-in cross-worker aggregation | Avoids a fake in-app distributed-metrics system; delegates to a real Prometheus scrape (export endpoint provided) |
| **Pickle** cache codec | Unsafe with untrusted Redis | Local-first / trusted-Redis assumption; numpy-lossless and fast; swappable codec abstraction in place |

---

## Quickstart

**Prerequisites:** Python 3.11, Node 18+. Optional: [Ollama](https://ollama.com) (local answer model), Tesseract (image OCR), Redis (shared cache).

### Backend
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.api                       # → http://localhost:8000  (OpenAPI at /docs)
```
> Factory-only by design (no import-time model load). `python -m src.api` is the foolproof launcher; the explicit equivalent is
> `uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000`.
> Search and ingest work immediately; **Ask** needs an answer model — `ollama serve && ollama pull llama3.2`, or paste a hosted-provider key in Settings.

### Frontend
```bash
cd frontend
cp .env.example .env.local              # NEXT_PUBLIC_API_BASE_URL defaults to http://localhost:8000
npm install
npm run dev                             # → http://localhost:3000
```

### Smoke test
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{"documents":["VectorFlow-RAG is a local-first hybrid RAG platform."],"reset":true}'
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' -d '{"query":"what is VectorFlow?","k":3}'
```

---

## Deployment (₹0)

The heavy ML stack runs on **your machine**; only the lightweight frontend is hosted. No paid servers, no domain, no credit card.

- **Frontend** → **Vercel** free tier (root directory = `frontend`, set `NEXT_PUBLIC_API_BASE_URL` to your tunnel URL).
- **Backend** → your Mac, exposed through a **free tunnel** — an **ngrok** free *static domain* (stable URL, no card) or a Cloudflare Quick Tunnel.
- **Database** → free **Postgres** (Neon / Supabase) for accounts + stats, or local **SQLite**.
- **Answer model** → local **Ollama** (free) or a hosted-provider key.
- **Container** → the backend ships a production `Dockerfile` (slim Python 3.11, non-root, healthcheck, tesseract for OCR): `docker build -t vectorflow-rag . && docker run -p 8000:8000 --env-file .env vectorflow-rag` — for any host or a rented ≥2 GB instance.

Auth, CORS, and OAuth callbacks are all environment-driven and HTTPS-aware. The full step-by-step (which env vars change per tunnel, OAuth app setup, smoke test, troubleshooting) is in **[`DEPLOYMENT.md`](DEPLOYMENT.md)**. A component-by-component analysis of what does and doesn't need a *stable* hostname (email/password & CORS don't; OAuth & the public frontend link do) is included there.

---

## Configuration

All settings are environment-driven (prefix `VFR_`, nested delimiter `__`), with a `.env` file supported. Highlights:

```bash
VFR_PROFILE=multilingual                          # english (default) | multilingual | multilingual_quality
VFR_VECTOR_STORE__BACKEND=faiss                   # chromadb | faiss
VFR_RERANKER__ENABLED=true                        # cross-encoder reranking
VFR_QUERY_EXPANSION__ENABLED=true                 # multi-query + HyDE
VFR_CACHE__BACKEND=redis                          # none | memory | redis
VFR_LLM__MODEL=llama3.2                           # any Ollama model (or pick a hosted provider in Settings)

# Multi-user / deployment
VFR_AUTH__REQUIRED=true                           # gate data endpoints behind login (prod)
VFR_AUTH__JWT_SECRET=<long-random>                # HS256 signing secret
VFR_AUTH__PUBLIC_BASE_URL=https://you.ngrok-free.app   # builds OAuth callbacks; sets cookie Secure
VFR_AUTH__FRONTEND_URL=https://you.vercel.app
VFR_API__CORS_ORIGINS=["https://you.vercel.app"]  # must include the deployed frontend origin
DATABASE_URL=postgresql://…                       # Postgres in prod; unset → local SQLite
VFR_SECRET_KEY=<fernet-key>                        # encrypts stored provider API keys
```

Profiles are presets that fill model/tokenizer choices **only** where you haven't overridden them — explicit env vars always win. Full list in [`.env.example`](.env.example); production template in [`.env.production.example`](.env.production.example).

---

## Testing

```bash
# Backend (ChromaDB default)
pytest -m "not integration and not slow"

# Backend on FAISS
VFR_VECTOR_STORE__BACKEND=faiss pytest tests/test_rag_pipeline.py tests/test_retrieval_quality.py

# Multilingual retrieval (loads the e5 model)
pytest tests/test_multilingual_retrieval.py -m slow

# Frontend
cd frontend && npm run test && npm run typecheck

# Index benchmark / multilingual benchmark (write versioned artifacts)
python -m experiments.multilingual_benchmark
```

Coverage spans: identity, chunking, both vector backends, RRF math, reranking, all 9 loaders, query expansion, caching (incl. `fakeredis`), API contracts, **auth (DB, JWT, OAuth, per-user stats)**, **model providers + secret store**, **index recipes/validation/benchmarking + compatibility + jobs**, **the macOS FAISS-thread-safety guard**, observability + overhead, encoding-safety, normalization fixtures, and end-to-end multilingual retrieval.

---

## Project structure

```
src/
├── api/              FastAPI service (app, routes, schemas, middleware, errors)
├── auth/             password hashing, JWT, OAuth (Google/GitHub), reset emails, service
├── db/               SQLAlchemy models (User, UserStats), session/engine
├── providers/        model providers (Ollama + OpenAI/Anthropic/Gemini/Groq/OpenRouter) + encrypted SecretStore
├── indexing/         named index registry, FAISS recipes, validation, benchmarking, compatibility
├── jobs/             thread-pool job registry (index build/benchmark, SSE progress, cancel)
├── cache/            memory/redis/null backends, SafeCache, caching wrappers
├── loaders/          9 format loaders + registry
├── observability/    metrics primitives, registry, Prometheus export
├── query_expansion/  multi-query + HyDE + pipeline
├── config.py · runtime_config.py · identity.py · rag_pipeline.py · rrf.py
├── hybrid_retriever.py · vector_store.py · faiss_store.py · bm25_retriever.py
├── embedder.py · reranker.py · llm_client.py · retrieval_trace.py · language.py
frontend/             Next.js 14 app (auth, pages, components, typed API client, tests)
experiments/          benchmark harnesses + versioned artifacts
docs/ARCHITECTURE.md  full system reference     TECHNICAL_REPORT.md  deep dive
DEPLOYMENT.md         ₹0 deployment runbook      CHANGELOG.md         phase-by-phase history
```

---

## Limitations & future work

Honest about what's *not* done:
- **Synchronous ingestion** — large corpora block an HTTP worker; async/queued ingestion is architecturally unblocked but not built.
- **No rate limiting** — public deployments rely on app-level auth + a max-file-size guard; add a reverse-proxy limiter for hardening.
- **No conversation persistence** — chat is single-turn (the streaming state is serializable, so lifting into a store + `/conversations` endpoint is small).
- **Monitoring dashboard** doesn't yet collect token-usage or system-resource (CPU/RAM) metrics; embedding time is folded into retrieval latency.
- **CJK BM25** is whitespace-tokenized (no segmentation); character-n-gram tokenization is a documented future opt-in.
- **`multilingual_quality`** preset (BGE-M3) is wired but benchmarked only at small scale (needs GPU).
- **Large-scale retrieval ablation** (MS-MARCO / BEIR with NDCG/MRR@10) is future work — current numbers are honest small-corpus regression gates.

Roadmap and extension points are detailed in [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).

---

## Documentation map
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — end-to-end architecture, every subsystem lifecycle, deployment topology, scaling boundaries, module map.
- **[`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)** — exhaustive design rationale, algorithms, alternatives considered, failure modes, scaling math.
- **[`DEPLOYMENT.md`](DEPLOYMENT.md)** — the ₹0 deploy runbook (Vercel + ngrok/Cloudflare tunnel + free Postgres + OAuth).
- **[`CHANGELOG.md`](CHANGELOG.md)** — phase-by-phase build history with test counts.
- **[`frontend/README.md`](frontend/README.md)** — frontend-specific run/build/deploy.

## License
MIT.
</content>
