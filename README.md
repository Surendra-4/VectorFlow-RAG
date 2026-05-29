# VectorFlow-RAG

**A production-grade, local-first Retrieval-Augmented Generation platform** — hybrid retrieval with Reciprocal Rank Fusion, cross-encoder reranking, query expansion, multi-format ingestion, provenance-rich answers, multilingual support, full observability, and a typed Next.js frontend. Zero paid APIs; everything runs on your machine.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Frontend](https://img.shields.io/badge/frontend-Next.js%2014-black)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-750%2B-success)
![Style](https://img.shields.io/badge/code%20style-black-000000)

> **TL;DR for reviewers:** this is not a notebook demo. It's a layered system — typed FastAPI service, two interchangeable vector backends (ChromaDB / FAISS-HNSW), a cache layer with structural invalidation, query-expansion + reranking stages, end-to-end provenance (stable chunk/document IDs), an opt-in multilingual profile that provably does not regress English retrieval, and a React frontend that depends only on the public API contract. ~750 automated tests, dual-backend validated. Deep design rationale lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).

---

## Table of contents
- [Why this exists](#why-this-exists)
- [Feature matrix](#feature-matrix)
- [Architecture at a glance](#architecture-at-a-glance)
- [The retrieval pipeline](#the-retrieval-pipeline)
- [Technology choices & reasoning](#technology-choices--reasoning)
- [Measured results](#measured-results)
- [Engineering trade-offs](#engineering-trade-offs-made-deliberately)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Limitations & future work](#limitations--future-work)

---

## Why this exists

Keyword search misses synonyms; vector search misses exact terms; most "RAG in 50 lines" demos are black boxes you can't debug, can't deploy, and can't trust. Real retrieval systems must juggle **recall, precision, latency, provenance, and operability at the same time** — and do it without shipping your documents to a third-party API.

VectorFlow-RAG combines **BM25 lexical matching** and **dense vector search**, fuses them with **Reciprocal Rank Fusion**, optionally refines with a **cross-encoder reranker** and **LLM query expansion**, and returns answers grounded in **source-attributed chunks** — all running locally via Ollama and sentence-transformers. Every retrieval is observable down to per-stage latency and per-modality rank.

---

## Feature matrix

| Capability | What it does | Where |
|---|---|---|
| **Hybrid retrieval (RRF)** | Rank-based fusion of BM25 + dense vectors; scale-invariant, no per-corpus tuning | `src/rrf.py`, `src/hybrid_retriever.py` |
| **Cross-encoder reranking** | Optional second-stage precision refinement over the fused pool | `src/reranker.py` |
| **Query expansion** | Multi-query rewriting + HyDE (hypothetical document embeddings) via local LLM | `src/query_expansion/` |
| **Dual vector backends** | ChromaDB (default) and FAISS-HNSW, behind one factory; switch by config | `src/vector_store.py`, `src/faiss_store.py` |
| **Stable identity & provenance** | Content-derived `doc_id`/`chunk_id`; every result carries document name, page, chunk index, per-modality ranks | `src/identity.py` |
| **Multi-format ingestion** | PDF, DOCX, TXT, Markdown, CSV/TSV, XLSX, SQLite, JSON, images (OCR) | `src/loaders/` |
| **Caching** | Memory / Redis / null behind a fail-safe wrapper; structural invalidation via corpus fingerprint | `src/cache/` |
| **Observability** | Thread-safe metrics registry, per-stage retrieval traces, Prometheus export, dashboard endpoints | `src/observability/`, `src/retrieval_trace.py` |
| **Multilingual (opt-in)** | `english` / `multilingual` / `multilingual_quality` profiles; English baseline provably preserved | `src/config.py`, `src/language.py` |
| **FastAPI service** | Versioned REST + SSE streaming, structured errors, correlation IDs, OpenAPI | `src/api/` |
| **Next.js frontend** | Chat (streaming), search, ingest (drag-drop), documents, dashboard, traces, settings | `frontend/` |

---

## Architecture at a glance

```
Next.js frontend  ──HTTP/SSE (typed contract)──►  FastAPI service (/api/v1)
                                                       │  middleware: request-id + timing + metrics
                                                       ▼
                                               RAGPipeline (orchestrator)
   chunk → embed → index → retrieve → fuse(RRF) → rerank? → generate(Ollama)
        │         │          │            │           │
   loaders/   embedder   vector store  bm25      reranker
   (9 fmts)   (+cache)   chroma/faiss  retriever  (optional)
        └──────────── cross-cutting: config(profiles) · cache(memory/redis) ·
                      query_expansion · retrieval_trace · observability · logging
```

Design invariants:
- **Frontend ↔ backend coupling is exactly one typed schema file.** The UI never imports backend code.
- **Retrieval is lock-free and concurrent; ingestion is serialized.**
- **Every swappable component sits behind a `Protocol`** (`src/interfaces.py`) — vector store, cache, reranker, embedder, LLM, loader, expansion strategy.
- **One configuration entry point** (`src/config.py`, env prefix `VFR_`).

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

**Why RRF over the more common α-weighted fusion** (`score = α·vec + (1−α)·bm25`): RRF is *rank-based*, so it needs no score normalization across two retrievers whose scores live on totally different scales (unbounded BM25 vs. cosine ∈ [0,1]). It has one well-studied constant (`k=60`) instead of a per-corpus `α` to tune, and it naturally rewards documents found by *both* modalities. The project began on α-fusion and migrated to RRF — the migration is documented in the changelog.

---

## Technology choices & reasoning

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| LLM runtime | **Ollama** (local) | Zero API cost, full privacy, offline-capable. Swappable model via one env var. |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2` default) | 384-dim, ~80 MB, CPU-viable. Auto-detects CUDA→MPS→CPU. |
| Sparse retrieval | **BM25 (`bm25s`)** | Battle-tested lexical baseline; catches exact/rare terms vectors miss. |
| Fusion | **Reciprocal Rank Fusion** | Scale-invariant, tuning-light, intersection-aware (see above). |
| Reranker | **cross-encoder/ms-marco-MiniLM-L-6-v2** (default) | ~80 MB, ~10 ms/pair CPU; biggest precision lever per MB. Optional. |
| Vector store | **FAISS-HNSW** + **ChromaDB** | HNSW = no training step + high recall + log-time search; Chroma = zero-setup default. One factory, swap by config. |
| Cache | **Redis** (opt) + in-memory LRU | Redis shares state across workers; memory cache is the zero-infra default; both behind a never-throws wrapper. |
| API | **FastAPI + Uvicorn** | Async-friendly, OpenAPI out of the box, Pydantic schemas double as the frontend contract. |
| Frontend | **Next.js 14 + TypeScript + Tailwind** | App Router, static-export-capable (desktop), no global-state library needed. |
| Streaming | **SSE (fetch + ReadableStream)** | Simpler and more debuggable than WebSockets for one-shot token streaming. |
| Config | **pydantic-settings** | Typed, env-driven, profile presets, one source of truth. |

---

## Measured results

> **Integrity note:** every number below was measured during development on this codebase (Apple Silicon, CPU/MPS). Golden-corpus retrieval sets are intentionally small and diverse — they're **regression gates and directional signals, not BEIR-grade leaderboard scores**. Large-scale benchmark replay (MS-MARCO/BEIR) is listed under future work. Nothing here is extrapolated.

### Test suite
- **~696 backend tests** collected (pytest). In a standard run **675 pass**; ~20 integration/slow tests are deselected (they require a live Ollama). Validated on **both** vector backends (ChromaDB + FAISS).
- **62 frontend tests** (Vitest + Testing Library), clean `tsc --noEmit`, successful production build.
- Total: **750+ automated tests.**

### Multilingual profile benchmark (golden corpus, 7 languages, k=5)
The hard gate: *English retrieval must not regress when multilingual support is added.* It held.

| Profile | Embedder | English R@5 | Monolingual MRR | Cross-lingual MRR | Code-switch MRR | Query p50 |
|---|---|---|---|---|---|---|
| `english` (default) | all-MiniLM-L6-v2 | **1.00** | 0.889 | 1.00 | 1.00 | 34.3 ms |
| `multilingual` | multilingual-e5-small | **1.00** | **1.00** | 1.00 | 1.00 | 21.0 ms |

English recall identical; the multilingual model *improves* monolingual ranking (MRR 0.889 → 1.00) on the mixed-language corpus. Artifacts are versioned in [`experiments/artifacts/`](experiments/artifacts/) (JSON + CSV, schema-versioned for future regression diffs).

### Vector backend latency (synthetic, n=2000, dim=128, p50)
| Backend | p50 | vs ChromaDB |
|---|---|---|
| ChromaDB | 0.39 ms | 1× |
| FAISS-Flat (exact) | 0.03 ms | ~13× faster |
| **FAISS-HNSW** | 0.06 ms | **~6.5× faster** |

### HNSW recall vs. search depth (synthetic, 1000×128-dim, vs. exact baseline)
| `efSearch` | Recall@10 |
|---|---|
| 8 | 0.62 |
| 64 (default) | passes ≥0.85 gate |
| 128 | 1.00 |

`efSearch` is a live query-time knob — trade recall for latency without rebuilding the index.

### Cache impact (golden corpus, full-retrieval cache)
- Cold retrieval: **12.4 ms** → warm (cache hit): **0.04 ms** → **~322× speedup** on repeat queries. The biggest single latency win once query expansion (which adds LLM calls) is enabled.

### Observability overhead (per-op, measured in isolation)
| Primitive | Cost |
|---|---|
| `Counter.inc` | 0.34 µs |
| `LabeledCounter.inc` | 1.07 µs |
| `Gauge.set` | 0.27 µs |
| `Histogram.observe` | 0.66 µs |
| `RingBuffer.append` | 0.24 µs |

Telemetry adds <0.1% to any path that does real retrieval work.

---

## Engineering trade-offs (made deliberately)

| Decision | Trade-off accepted | Why |
|---|---|---|
| Multilingual as **opt-in profile**, not default | Users must flip a config flag for non-English | Cross-lingual embedding spaces can slightly weaken monolingual precision; keeping English as the validated control eliminates regression risk |
| **RRF** over tuned α-fusion | Gives up per-corpus score weighting | Robustness and zero-tuning beat marginal tuned gains across heterogeneous corpora |
| **Pickle** cache codec | Unsafe with untrusted Redis | Local-first / trusted-Redis assumption; numpy-lossless and fast. Documented; swappable codec abstraction in place |
| **Synchronous ingestion** (lock-serialized) | Large batches block an HTTP worker | Correctness and simplicity now; loaders are isolated from indexing so async/queued ingestion is a non-breaking future add |
| **Process-local metrics** | No built-in cross-worker aggregation | Avoids a fake in-app distributed-metrics system; delegates aggregation to a real Prometheus scrape (export endpoint provided) |
| **No global state lib** on frontend | Manual hook composition | The data dependencies are simple; Redux/Zustand would be weight without benefit |
| **CJK BM25 = no-stemmer whitespace** | Sub-optimal for Chinese/Japanese word segmentation | Correct level of complexity for this phase; character-n-gram tokenization is a documented future opt-in |

---

## Quickstart

**Prerequisites:** Python 3.10–3.12, Node 18+. Optional: [Ollama](https://ollama.com) (for the `ask`/chat LLM), Tesseract (image OCR), Redis (shared cache).

### Backend
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.api                       # → http://localhost:8000  (docs at /docs)
```
> The app is factory-only by design (no import-time model load). `python -m src.api` is the foolproof launcher; the explicit equivalent is
> `uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000`.
> Search and ingest work immediately; only **Ask** needs Ollama (`ollama serve && ollama pull tinyllama`).

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

## Configuration

All settings are environment-driven (prefix `VFR_`, nested delimiter `__`), with a `.env` file supported. Highlights:

```bash
VFR_PROFILE=multilingual                        # english (default) | multilingual | multilingual_quality
VFR_VECTOR_STORE__BACKEND=faiss                 # chromadb | faiss
VFR_RERANKER__ENABLED=true                      # cross-encoder reranking
VFR_QUERY_EXPANSION__ENABLED=true               # multi-query + HyDE
VFR_CACHE__BACKEND=redis                         # none | memory | redis
VFR_LLM__MODEL=llama3.2:1b                        # any Ollama model
VFR_API__CORS_ORIGINS=["https://your-frontend"]  # must include the deployed frontend origin
```

Profiles are presets that fill model/tokenizer choices **only** where you haven't overridden them — explicit env vars always win. Full list in [`.env.example`](.env.example).

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

# Multilingual benchmark (writes versioned artifacts)
python -m experiments.multilingual_benchmark
```

Test layout covers: identity, chunking, both vector backends, RRF math, reranking, all 9 loaders, query expansion, caching (incl. fakeredis), API contracts, observability + overhead, encoding-safety, normalization fixtures, and end-to-end multilingual retrieval.

---

## Project structure

```
src/
├── api/              FastAPI service (app, routes, schemas, middleware, errors)
├── cache/            memory/redis/null backends, SafeCache, caching wrappers
├── loaders/          9 format loaders + registry
├── observability/    metrics primitives, registry, Prometheus export
├── query_expansion/  multi-query + HyDE + pipeline
├── config.py         Settings + profiles          identity.py     stable IDs
├── rag_pipeline.py   orchestrator                  rrf.py          fusion
├── hybrid_retriever.py · vector_store.py · faiss_store.py · bm25_retriever.py
├── embedder.py · reranker.py · llm_client.py · retrieval_trace.py · language.py
frontend/             Next.js 14 app (pages, components, typed API client, tests)
experiments/          multilingual benchmark harness + versioned artifacts
docs/ARCHITECTURE.md  full system reference          TECHNICAL_REPORT.md  deep dive
CHANGELOG.md          phase-by-phase build history
```

---

## Limitations & future work

Honest about what's *not* done:
- **No auth / multi-user** — single-tenant local-first only.
- **Monitoring dashboard covers 5 of 7 originally-scoped metrics** — token usage and system-resource (CPU/RAM) metrics are not yet collected; embedding time is folded into retrieval latency rather than surfaced standalone.
- **Synchronous ingestion** — large corpora block an HTTP worker; async/queued ingestion is architecturally unblocked but not built.
- **No rate limiting** — public deployments need hardening (a max-file-size guard is the only current protection).
- **No conversation persistence** — chat is single-turn.
- **`multilingual_quality` preset** (BGE-M3) is wired but benchmarked only at small scale (needs GPU).
- **Large-scale retrieval ablation** (MS-MARCO / BEIR with NDCG/MRR@10) is future work — current numbers are honest small-corpus regression gates.

Roadmap and extension points (async ingestion, dedup, reranker caching, auth, BEIR replay) are detailed in [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).

---

## Documentation map
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — end-to-end architecture, every subsystem lifecycle, deployment topology, scaling boundaries, module map.
- **[`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)** — exhaustive design rationale, algorithms, alternatives considered, failure modes, scaling math.
- **[`CHANGELOG.md`](CHANGELOG.md)** — phase-by-phase build history with test counts.
- **[`frontend/README.md`](frontend/README.md)** — frontend-specific run/build/deploy.

## License
MIT.
