# VectorFlow-RAG — Technical Report

The exhaustive engineering reference for this system: every subsystem, the
design decisions behind it, the alternatives that were rejected, the
trade-offs accepted, the algorithms and data structures used, the failure
modes, and the scaling math. Where the README sells, this document explains.

**Audience:** the maintainer (you) and any engineer who needs to extend,
debug, or operate the system with full understanding. It assumes you've
skimmed [`README.md`](README.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Contents**
1. [System model & layering](#1-system-model--layering)
2. [Configuration & profiles](#2-configuration--profiles)
3. [Identity model](#3-identity-model-doc_id--chunk_id)
4. [Ingestion subsystem](#4-ingestion-subsystem)
5. [Embedding subsystem](#5-embedding-subsystem)
6. [Sparse retrieval (BM25)](#6-sparse-retrieval-bm25)
7. [Vector stores](#7-vector-stores)
8. [Fusion: Reciprocal Rank Fusion](#8-fusion-reciprocal-rank-fusion)
9. [Reranking](#9-reranking)
10. [Query expansion](#10-query-expansion)
11. [Caching](#11-caching)
12. [Observability](#12-observability)
13. [RetrievalTrace](#13-retrievaltrace)
14. [API service](#14-api-service)
15. [Frontend](#15-frontend)
16. [Multilingual architecture](#16-multilingual-architecture)
17. [Concurrency & thread-safety](#17-concurrency--thread-safety)
18. [Performance & scaling math](#18-performance--scaling-math)
19. [Failure modes & graceful degradation](#19-failure-modes--graceful-degradation)
20. [Testing strategy](#20-testing-strategy)
21. [Limitations & future work](#21-limitations--future-work)

---

## 1. System model & layering

The system is a **single Python process** (the FastAPI service) holding a
**`RAGPipeline` singleton**, plus a **separate Next.js process** that talks to
it over HTTP/SSE. Local model inference (embeddings, reranker) runs in-process;
LLM generation is delegated to a local **Ollama** HTTP server.

Layering, innermost → outermost:

```
interfaces.py (Protocols)         ← contracts; nothing imports upward
  ↑
config.py · identity.py · logging_setup.py   ← foundation, no heavy deps
  ↑
embedder · chunker · bm25_retriever · vector_store/faiss_store · rrf · reranker
  ↑
query_expansion · cache · loaders · retrieval_trace · observability
  ↑
rag_pipeline.py (orchestrator)    ← imports everything above
  ↑
api/ (FastAPI)                    ← imports the pipeline
  ↑
frontend/ (typed client)          ← imports nothing Python; only the schema shape
```

The dependency direction is strict: foundation modules never import the
orchestrator. This is why `rag_pipeline.py` is the *last* commit in the git
history — it's the only module that legitimately depends on all subsystems.

**Why a singleton pipeline.** Models are expensive to load (embedder ~80 MB,
reranker ~80 MB). Loading once at process start and sharing across requests is
the only sane option. The pipeline is constructed via FastAPI dependency
injection (`src/api/dependencies.py`) and reused for every request.

---

## 2. Configuration & profiles

**One source of truth:** `src/config.py` defines a `Settings` pydantic-settings
model with nested sub-models (`embedder`, `chunker`, `vector_store`, `bm25`,
`llm`, `retrieval`, `reranker`, `query_expansion`, `cache`, `api`, `ingestion`,
`logging`). Env prefix `VFR_`, nested delimiter `__`, `.env` supported.

- `VFR_EMBEDDER__MODEL_NAME=foo` → `settings.embedder.model_name`
- A cached singleton via `get_settings()`; `reset_settings_cache()` exists for tests.

**Profiles.** `Settings.profile ∈ {english, multilingual, multilingual_quality}`.
A profile is a **preset** that fills in model/tokenizer choices, but only for
fields still at their English defaults — **explicit env overrides always win**.
This is implemented as a post-construction resolution: if you set
`VFR_PROFILE=multilingual` but also `VFR_EMBEDDER__MODEL_NAME=my-model`, your
model wins. The English profile is the default precisely so the validated
baseline is never silently changed.

**Why pydantic-settings over a dict/yaml loader:** typed access (the compiler
and IDE know `settings.retrieval.rrf_k` is an int), validation at load, env +
file + programmatic override precedence handled for free, and the same models
serialize into the API's `/status` response.

---

## 3. Identity model (`doc_id` / `chunk_id`)

This is the backbone that makes provenance, dedup, and cache-invalidation
possible. Implemented in `src/identity.py`.

### Document ID
```
doc_id = "doc_" + sha256(normalize(content)).hexdigest()[:16]
```
- 16 hex chars = 64 bits. Collision probability at 1M documents ≈ 1.8 × 10⁻¹⁹ (birthday bound). Negligible.
- **Content-derived, not random** → idempotent. Re-ingesting identical content yields the same `doc_id`. This is what makes future dedup mechanical.

### Chunk ID
```
chunk_id = f"{doc_id}:{chunk_index}:{sha256(normalize(chunk_text)).hexdigest()[:8]}"
```
Three components, each load-bearing:
- `doc_id` — **source attribution is baked in.** Two byte-identical chunks in two different documents get *different* chunk_ids. This is the right call for RAG: a citation must know which document it came from.
- `chunk_index` — preserves order; survives the multi-page `chunk_index_offset` mechanism (see §4).
- 8-hex text hash (32 bits, namespaced under the doc) — detects content change; if you re-chunk with different parameters, chunk_ids change.

### Normalization (before hashing only)
```
1. Unicode NFC normalize       (compose: a + ◌́  →  á)
2. Collapse whitespace runs → single space
3. Strip leading/trailing whitespace
   (case is PRESERVED — case is semantically meaningful)
```
The **raw** text is what's stored and displayed; normalization is applied only
to the hash input. Two chunks differing only in whitespace collide
intentionally (they're effectively duplicates).

**NFC, not NFKC.** NFKC folds compatibility characters (ﬁ → fi, fullwidth →
ASCII), which would alter citation-visible text and risk identity drift on
exotic inputs. NFC is the conservative, reversible-enough choice. (Future opt-in
NFKC is possible but off by default.)

**Why text-as-join-key was abandoned.** The original system fused BM25 and
vector results by string-equality on chunk text. Identical boilerplate (page
footers, NDAs, table-of-contents) across documents collapsed into one retrieval
entry, destroying source attribution. Switching the RRF join key to `chunk_id`
fixed this — verified by a test that ingests the same disclaimer sentence in two
documents and asserts both remain individually retrievable.

---

## 4. Ingestion subsystem

Two entry points share one indexing core:
- `RAGPipeline.ingest_documents(list[str])` — raw text
- `RAGPipeline.ingest_files(paths)` — any registered format

### Loaders (`src/loaders/`)
A `LoaderProtocol` (runtime-checkable) with a `LoadedDocument`/`LoadedPage` data
model. The `LoaderRegistry` dispatches by file extension / MIME type. Nine
loaders ship:

| Loader | Format | Page model |
|---|---|---|
| `text` | .txt, .md (Markdown is text), .log | 1 page |
| `json_loader` | .json | 1 page, canonicalized (sorted keys) for stable hashing |
| `csv_loader` | .csv, .tsv | 1 page, rows flattened |
| `xlsx_loader` | .xlsx | **1 page per sheet** (sheet name in metadata) |
| `sqlite_loader` | .db/.sqlite | **1 page per table** (schema + rows) |
| `pdf_loader` | .pdf (pypdf) | **1 page per PDF page** |
| `docx_loader` | .docx (python-docx) | 1 page |
| `image_ocr_loader` | .png/.jpg/… (pytesseract) | 1 page; operator-installed lang packs |

Each `LoadedPage` carries canonical provenance: `document_name`, `source_path`,
`page_number`, `mime_type`, `loader`. Loaders never import retrieval code —
ingestion is fully isolated from indexing, which is what keeps future
async/streaming ingestion a non-breaking change.

### Failure handling
`ingest_files` wraps each file in try/except: a corrupt PDF, missing file, or
unknown extension produces a `failures` entry, never poisons the batch. A
configurable max-file-size guard (`VFR_INGESTION__MAX_FILE_SIZE_BYTES`, default
100 MB) fails fast before a parser is invoked. `fail_fast` mode is opt-in.

### Chunking & the `chunk_index_offset` mechanism
`TextChunker.chunk_text(text, metadata, doc_id, chunk_index_offset)` splits on
sentence terminators — extended in Phase 11 to include CJK terminators
(`。！？`) — accumulating sentences up to `chunk_size` with `overlap`.

The subtlety: a multi-page PDF is chunked **page by page** so each chunk inherits
its page's `page_number`. But chunk_id must be globally unique within the
document. Solution: the pipeline passes a running `chunk_index_offset` so page 2's
chunks continue numbering where page 1 left off. Without this, page 1 chunk 0 and
page 2 chunk 0 with identical text would collide on chunk_id.

### Corpus fingerprint
After indexing: `corpus_fingerprint = sha256("|".join(sorted(chunk_ids)))[:16]`.
This single value changes whenever the corpus changes and is the mechanism for
retrieval-cache invalidation (§11).

---

## 5. Embedding subsystem

`src/embedder.py` wraps sentence-transformers. Two design points:

**Device auto-detection.** On construction, probes `cuda → mps → cpu` and picks
the best available, overridable via `VFR_EMBEDDER__DEVICE`. This is why it runs
on Apple Silicon (MPS) without configuration.

**The `input_type` adapter (multilingual encapsulation).** Some models (the E5
family) require asymmetric prefixes: queries get `"query: "`, passages get
`"passage: "`. The crucial design rule from the multilingual phase:
**no caller may know which embedder is active.** So `encode(texts, input_type=...)`
takes a *semantic* hint (`"query"` / `"passage"` / `None`) — the embedder
decides whether to apply prefixes. The English MiniLM model ignores `input_type`
entirely, so English behavior is byte-identical regardless of what callers pass.
The E5 adapter applies the prefix internally. The caller states intent, never
model awareness.

**Caching** is a *wrapper* (`src/cache/caching_embedder.py`), not baked into the
embedder. `CachingEmbedder` decorates any embedder: it looks up each text by
`(model_name, sha(text), input_type)`, batches the cache *misses* into a single
underlying `encode` call, deduplicates repeated texts within a batch, and
stitches results back into original order. This keeps the hit rate high even
when callers always pass batches (ingestion).

---

## 6. Sparse retrieval (BM25)

`src/bm25_retriever.py` wraps the `bm25s` library. Two evolutions:

- **Dict-corpus (Phase 3.5).** Accepts either `list[str]` (legacy) or
  `list[dict]` with `text` + `chunk_id` + `metadata`. When given IDs, search
  results carry `chunk_id` so RRF can join on identity rather than text.
- **No-stemmer multilingual mode (Phase 11).** English uses a Snowball stemmer;
  the multilingual profile switches to Unicode-aware whitespace tokenization with
  no stemming. Rationale: an English stemmer mangles German morphology and is
  meaningless for scripts without spaces; no-stemming is the safe,
  language-agnostic default. Per-language stemmers remain available via config
  for known monolingual corpora.

**Known weakness:** for CJK, whitespace tokenization treats a whole sentence as
one token (no word boundaries). BM25's contribution there is weak; dense
retrieval carries those queries. Character-n-gram tokenization is the documented
future fix, deliberately out of scope to avoid over-engineering.

---

## 7. Vector stores

Two backends behind `VectorStoreProtocol`, selected by
`make_vector_store(backend=...)`:

### ChromaDB (default)
Zero-setup embedded store. The Phase 3 work fixed a real bug: the original
`search()` short-circuited on an in-memory `self.documents` shadow list, so a
freshly-reloaded store returned empty results even though the persisted
collection had data. The fix checks `collection.count()` instead.

### FAISS-HNSW
`IndexHNSWFlat(dim, M=32, METRIC_INNER_PRODUCT)` with `efConstruction=200`,
`efSearch=64` (all config-tunable). Cosine similarity via inner product on
normalized embeddings.

- **Why HNSW over IVF/IVF-PQ:** no training step (critical for incremental
  ingestion — IVF needs a representative fit and drifts as you add), high recall
  out of the box, log-time search, and a *query-time* recall/latency knob
  (`efSearch`) that needs no rebuild.
- **Persistence:** binary FAISS index + a JSON sidecar (`documents`,
  `metadatas`, `ids`, `dim`, `index_type`). Atomic-ish save via tmp-file rename.
  On reload, missing/inconsistent sidecar → fresh empty index.
- **ID-first internals:** `_id_to_pos` / `_pos_to_id` maps mean the store is
  identity-aware from the ground up; `search()` returns `ids` aligned with
  `documents`.
- **Deletes:** HNSW doesn't support deletion; the supported path is full rebuild
  (the pipeline already does delete-then-add on `reset=True`).

Measured: HNSW ~6.5× faster than ChromaDB at 2k vectors (synthetic); recall
tunable from 0.62 (efSearch=8) to 1.00 (efSearch=128).

---

## 8. Fusion: Reciprocal Rank Fusion

`src/rrf.py`. The core formula (Cormack et al., 2009):

```
RRF(d) = Σ_i  1 / (k + rank_i(d))
```
where `rank_i(d)` is document d's 1-indexed rank in modality i (vector, BM25,
and each expanded query/HyDE input), and `k = 60`. Documents absent from a
modality contribute nothing for that modality.

**Why rank-based fusion.**
- *Scale-invariance.* BM25 scores are unbounded (~0–20+); cosine is [0,1]. A
  linear combo `α·vec + (1−α)·bm25` is dominated by whichever scale is larger
  unless you normalize per-corpus. RRF sidesteps this entirely — only ranks
  matter.
- *Tuning-light.* One constant `k=60` (well-studied, robust across datasets)
  versus a per-corpus `α` that needs re-tuning whenever the corpus changes.
- *Intersection-aware.* A document ranked modestly by *both* modalities beats a
  document ranked #1 by only one. Example: at k=60, a doc at rank 5 in both
  lists scores 2/65 ≈ 0.0308, beating a doc at rank 1 in one list (1/61 ≈
  0.0164). This is exactly the behavior you want from hybrid search.

**The join key.** `rrf_with_ranks()` fuses on `chunk_id` when both modalities
expose IDs, falling back to text otherwise (legacy/string-corpus path). This is
what makes the fusion provenance-correct (see §3).

**Determinism.** Ties are broken by first-appearance order across input lists,
so the same inputs always produce the same ranking — important for cache
consistency and reproducible tests.

**Trade-off acknowledged.** RRF discards the *magnitude* of relevance (how much
better #1 is than #2). When a downstream consumer needs calibrated scores, the
cross-encoder reranker (§9) provides them.

---

## 9. Reranking

`src/reranker.py` — an optional, lazy, config-driven second stage. A
cross-encoder jointly encodes `(query, candidate)` pairs and emits a relevance
score per pair. This is far more accurate than the bi-encoder cosine of the
dense retriever (which encodes query and document independently), at O(N) cost
per query where N = candidate-pool size.

- **Default model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB, ~10 ms/pair
  CPU). Multilingual presets swap to `jina-reranker-v2-base-multilingual` or
  `bge-reranker-v2-m3`.
- **Lazy + optional:** disabled by default; the model loads on first use only.
  The pipeline retrieves a wider pool (e.g. 10–20) and the reranker truncates to
  the requested `k`.
- **When it earns its cost:** queries with subtle semantic distinctions or weak
  keyword overlap. On the Phase 1 comparison, the cross-encoder produced
  well-separated scores (+10.21 for the relevant doc vs −7.44 for the runner-up
  on a Paris query) — useful not just for ordering but for future
  confidence-thresholding ("no good answer").
- **Cost:** ~150–500 ms on CPU for a 20-candidate pool; this dominates retrieval
  latency when enabled. Mitigations: GPU/MPS device, smaller pool, or rely on
  the full-retrieval cache for repeat queries.

---

## 10. Query expansion

`src/query_expansion/` — improves recall on short/ambiguous queries by
generating additional retrieval inputs via the local LLM. Two strategies, both
optional and composable:

- **Multi-query** (`multi_query.py`): asks the LLM for N alternative phrasings.
  The parser is defensive — strips numbering (`1.`, `-`, `*`), surrounding
  quotes, blank lines; dedupes; length-caps each variant; drops variants equal
  to the original. Subsumes "synonym expansion" (the LLM naturally produces
  synonym-rich rewrites) without a dedicated synonym module.
- **HyDE** (`hyde.py`): generates M hypothetical answer documents and embeds them
  as additional query vectors. Bridges vocabulary gaps the bi-encoder misses
  ("powerhouse of the cell" → retrieves "mitochondria").

All variants and HyDE docs feed the *same* RRF call as labeled rank lists
(`vector:original`, `bm25:original`, `vector:variant_1`, …, `vector:hyde_0`).

**Prompt-injection defenses (tested):** queries are sanitized (control chars
stripped, length-capped) *before* reaching the LLM; LLM output is treated as
plain strings, never reinterpreted as instructions; per-variant length caps
prevent output bombs; per-strategy wall-clock timeout; a strategy that crashes
or times out degrades to "no expansion" — the original query still retrieves.

**Honest precision note:** multi-query can *shift* top-1 when a generated variant
introduces a generic term ("algorithm") that strongly matches an unrelated doc
via BM25. RRF dampens but doesn't always reverse this. The cross-encoder
reranker (scoring against the *original* query) is the corrective. This is the
classic recall/precision trade-off of query expansion, documented in tests.

**Caching implication:** expansion is LLM-bound (1–7 s). It's cached by
`(expansion_model, strategies, params, sha(query))` so repeat queries skip the
LLM entirely.

---

## 11. Caching

`src/cache/`. The most operationally impactful subsystem.

### Backends + the SafeCache boundary
Three backends conform to `CacheProtocol`: `NullCache` (no-op, default when
caching off), `MemoryCache` (thread-safe LRU + per-entry TTL), `RedisCache`
(shared across workers, lazy `redis` import). All are wrapped by `SafeCache` at
the factory boundary, which:
- **never raises** — any backend error degrades to a cache miss + logged warning;
- counts hits/misses/sets/errors for observability.

This means no caller (CachingEmbedder, expansion cache, retrieval cache) ever
has to defend against a Redis blip — at worst they recompute.

### MemoryCache internals
`OrderedDict` for O(1) LRU (`move_to_end` on access, `popitem(last=False)` on
overflow). Per-entry TTL checked lazily on read; a small bounded sweep purges
expired entries on write so memory stays bounded between LRU evictions. Guarded
by a single `RLock` — sufficient for the read-heavy access pattern.

### Cache-key design
Namespaced and schema-versioned so format changes can't read stale data:
```
vfr:emb:v1:{model}:{sha(text)}:{input_type}        → embedding (numpy)
vfr:exp:v1:{exp_model}:{strategies}:{params}:{sha(query)}   → ExpandedQuery
vfr:ret:v1:{corpus_fingerprint}:{rag_config}:{sha(query)}:k{k}  → final results
```

### Structural invalidation (the key idea)
There is **no `invalidate()` API to misuse.** Correctness is structural:
- Embeddings are content-keyed → can never be stale; TTL is purely for eviction.
- Expansion outputs are input-keyed → same.
- **Retrieval results embed the corpus fingerprint in their key.** Re-ingesting
  changes the fingerprint → old keys become unreachable → they TTL out. There is
  no window in which a stale retrieval result can be served after the index
  changes.

Switching the embedder model also auto-invalidates embedding + retrieval caches
because the model name is in the key.

### Codec & trust model
Values are pickled (`PickleCodec`) — numpy-lossless and ~3× faster than
JSON+base64 for float arrays. Pickle is unsafe with untrusted data; the
local-first / trusted-Redis assumption makes this acceptable, and the codec is a
swappable abstraction if shared-tenant Redis ever appears.

Measured: full-retrieval cache delivers ~322× warm-vs-cold speedup on the golden
corpus.

---

## 12. Observability

`src/observability/`. Process-local, thread-safe, low-overhead, Prometheus-
compatible — without taking a Prometheus dependency.

### Primitives
`Counter`, `LabeledCounter`, `Gauge`, `Histogram`, `LabeledHistogram`,
`RingBuffer` — each lock-guarded. Measured per-op cost: 0.24–1.07 µs.

- **Histogram = rolling time window (300 s) + hard sample cap (5000).** Memory is
  bounded regardless of throughput; percentiles (p50/p95/p99) computed on read by
  sorting the live window.
- **RingBuffer** (fixed-size `deque`) holds the most-recent N RetrievalTraces and
  errors for the dashboard.

### Cardinality discipline
Labels are bounded by construction — endpoint paths (route templates, not URLs),
HTTP status codes, cache namespaces, expansion strategy names. Never `request_id`
or full query text, which would explode cardinality.

### Collector hooks
Middleware records `requests_total{endpoint,status}` + `request_latency_ms`;
the pipeline records retrieval counters/histograms + pushes traces; ingest
endpoints record ingestion metrics; the SSE route tracks `active_streams` (gauge)
and `stream_duration_ms`.

### Export
`/api/v1/metrics/snapshot` (structured JSON) for the dashboard;
`/api/v1/metrics/prometheus` (text exposition format, hand-rolled, zero deps) for
any Prom-compatible scraper. **Cross-worker aggregation is deliberately delegated
to a real Prometheus server** rather than building a fake in-app distributed
metrics layer.

### Gaps (honest)
Token usage and system-resource (CPU/RAM) metrics are not collected; embedding
time is captured inside retrieval latency rather than as a standalone stage
metric. These are the two unbuilt sub-items of the originally-scoped dashboard.

---

## 13. RetrievalTrace

`src/retrieval_trace.py` — the structured record of a single retrieval, and the
substrate the dashboard consumes. Captures: original + sanitized query, expanded
queries, HyDE docs, strategies used, expansion errors + latency, per-strategy
candidate lists (labeled by modality + source query), fused pool size + fusion
latency, reranker stats (used / input size / output size / latency), final
results, total latency, `from_cache` flag, and a cache-stats snapshot.

`to_dict()` is JSON-serializable (UTF-8-safe, verified by the encoding-safety
audit). `search(..., return_trace=True)` returns `(results, RetrievalTrace)`;
the default returns just the results list (backward-compatible).

This is what lets you answer "why did this query return these chunks, how long
did each stage take, and was it a cache hit?" — per request, after the fact.

---

## 14. API service

`src/api/`. FastAPI app via a **factory** (`create_app()`), deliberately with no
module-level `app` instance so importing the module never triggers a model load
(important for tests and multi-worker). The foolproof launcher is
`python -m src.api`; the explicit form needs `--factory`.

- **Middleware stack:** correlation-id (assigns/echoes `X-Request-ID`) →
  timing (records `X-Process-Time-Ms` + metrics) → CORS.
- **Structured errors:** one exception handler maps domain errors to
  `ErrorResponse{code, message, details, request_id}`; loader errors → 422,
  value errors → 400, unknown → 500, plus Starlette's 404. Every error logs the
  request_id.
- **Routes** under `/api/v1` (status, ingest×3, search, ask, cache, documents,
  admin, observability); `/health` at root; `/docs` + `/openapi.json` auto-served.
- **SSE streaming:** `/api/v1/ask?stream=true` emits `sources → token* → done |
  error` via `StreamingResponse`. Chosen over WebSockets — simpler, debuggable,
  sufficient for one-shot token streaming.
- **Concurrency model:** sync `def` handlers run in FastAPI's threadpool.
  Retrieval is read-only and lock-free; ingestion takes a process-wide lock.
  Versioned `/api/v1` so breaking changes go to `/api/v2`.

---

## 15. Frontend

`frontend/` — Next.js 14 App Router, TypeScript, Tailwind. **No global-state
library** (React hooks suffice for these data dependencies), **no UI kit**, **no
cloud deps**.

- **One contract:** `src/lib/api/types.ts` mirrors the backend Pydantic schemas.
  The UI imports nothing else from the backend. A schema change surfaces as a TS
  error at the call site.
- **SSE without EventSource:** the browser `EventSource` is GET-only and can't
  send a JSON body, so `src/lib/api/sse.ts` is a custom parser over `fetch` +
  `ReadableStream`; `useStreamingAsk` aggregates events into React state with
  cancel/reset.
- **State hooks:** `useApi` (load + AbortController cleanup), `usePolling`
  (dashboard, pauses when tab hidden), `useTheme` (light/dark/system).
- **Rendering (multilingual):** user-content containers use `dir="auto"` so the
  browser picks LTR/RTL per content; OS-native font stack covers CJK/RTL without
  web-font downloads (offline-safe). UI-string i18n is intentionally deferred —
  Unicode-safe *rendering* matters more than translated chrome.
- **Deployment:** `next dev` / `next build && start` / `OUTPUT=export` static
  (desktop/Tauri). Runtime backend-URL override (localStorage) lets a
  Vercel-hosted UI point at any backend without rebuild.

---

## 16. Multilingual architecture

Treated as **two problems**: retrieval quality *and* systems behavior. The
governing constraint: **English retrieval must not regress** — so multilingual
is an **opt-in profile**, never a default change.

- **Model-driven multilinguality.** No per-language branching in retrieval. The
  `multilingual` profile swaps to `multilingual-e5-small` (384-dim, same
  footprint as the English default) + a multilingual reranker + no-stemmer BM25.
  `multilingual_quality` swaps to BGE-M3 (1024-dim, GPU-recommended).
- **Encapsulated model quirks.** E5's `query:`/`passage:` prefixes live entirely
  in the embedder adapter (§5).
- **NFC everywhere.** Query sanitization, BM25 corpus, loader output, and
  identity hashing all NFC-normalize. Validated by mixed-normalization fixtures
  (NFC/NFD composed vs decomposed accents, fullwidth punctuation, zero-width
  spaces, RTL punctuation).
- **Language detection is advisory only.** `src/language.py` (langid) detects the
  *query* language to add a same-language hint to the expansion prompt. It
  **never** routes retrieval — that would be brittle. Per-chunk detection is
  opt-in and off by default.
- **OCR languages** are operator-installed Tesseract packs, never auto-downloaded.

**Evaluation methodology.** A tiered golden set: Tier 0 = existing English
goldens (regression gate, must stay green); Tier 1 = per-language monolingual
(7 languages); Tier 2 = cross-lingual (English query → French doc, etc.); Tier 3
= code-switching. Metrics: Recall@k, MRR per tier. Result: English R@5 = 1.00 on
both profiles; multilingual monolingual MRR improved 0.889 → 1.00. Artifacts are
schema-versioned in `experiments/artifacts/` for future diffs.

**Encoding-safety audit.** Multilingual systems often fail in the observability
layer before the retrieval layer. A dedicated test suite asserts non-ASCII
(CJK/RTL/Cyrillic/emoji/combining marks) survives JSON serialization, the pickle
cache codec, RetrievalTrace.to_dict(), Prometheus export, structured logs, and
SSE encoding.

---

## 17. Concurrency & thread-safety

| Component | Concurrency story |
|---|---|
| `RAGPipeline.search` | Read-only after ingestion → safe for concurrent calls, no lock |
| `RAGPipeline.ingest_*` | Mutates vector store / BM25 / fingerprint → process-wide lock serializes it |
| `MemoryCache` / `SafeCache` | `RLock`-guarded |
| `RedisCache` | redis-py connection pool is thread-safe |
| Embedder / reranker (`model.encode` / `.predict`) | PyTorch inference is thread-safe (no autograd state) |
| Vector stores (`query`/`search`) | Read-only, safe |
| Observability primitives | Each lock-guarded |
| FastAPI handlers | Sync `def` → threadpool; the above guarantees make this safe |

**Multi-worker:** each Uvicorn worker has its own pipeline + models (no shared
memory). A shared **Redis** cache lets workers reuse each other's embeddings /
expansion / retrieval results — the only cross-worker shared state, and it's
optional.

---

## 18. Performance & scaling math

### Model memory
| Config | Embedder | Reranker (if on) | Vector dim |
|---|---|---|---|
| English default | ~80 MB | ~80 MB | 384 |
| `multilingual` | ~118 MB | ~278 MB (jina) | 384 |
| `multilingual_quality` | ~568 MB | ~568 MB (bge-m3) | 1024 |

### Vector store growth (HNSW, float32)
- Per vector: `dim × 4 bytes` + graph overhead (~1.5–2×).
- 384-dim: ~1.5 KB/vector raw → 100k chunks ≈ ~60 MB (incl. graph); 1M ≈ ~600 MB.
- 1024-dim (bge-m3): ~2.6× that. This is the expensive part of the quality preset.

### Latency budget (per retrieval, CPU, expansion off)
- Query embed: ~10–30 ms (or <0.1 ms cached)
- Hybrid retrieve: ~20–60 ms
- Rerank (if on, 20 candidates): ~150–500 ms
- Full-retrieval cache hit: ~0.04 ms (skips all of the above)

With expansion on, the LLM calls (1–7 s) dominate cold latency — which is
precisely why the expansion + full-retrieval caches matter most there.

### Scale thresholds
| Corpus | Recommendation |
|---|---|
| <1k chunks | ChromaDB or FAISS-Flat — either is fine |
| 1k–1M | FAISS-HNSW (the sweet spot; default knobs sized for this) |
| 1M–10M | FAISS-HNSW with larger M; ~6 GB at 384-dim |
| >10M | Switch to IVF-PQ (compression) — config-selectable index type |

---

## 19. Failure modes & graceful degradation

| Failure | Behavior |
|---|---|
| Ollama down | `ask` returns HTTP 200 with sources + metrics + an error string in `answer`; search/ingest unaffected |
| Redis unreachable | Factory logs once, falls back to MemoryCache; retrieval continues (colder) |
| Cache backend error mid-op | SafeCache treats it as a miss; recompute |
| Corrupt / unreadable file | Per-file failure entry; batch continues |
| Oversized file | Rejected before parsing (size guard) |
| Expansion LLM timeout/crash | Strategy degrades to "no expansion"; original query still retrieves |
| Tesseract not installed | Image loader raises a clear install-hint error; other formats unaffected |
| FAISS sidecar corrupt | Fresh empty index (logged), no crash |
| Embedding-model HF download fails (first run, offline) | Startup error — needs internet once, then fully local |

The system's posture: **a dependency being unavailable degrades the affected
feature, never the whole service.**

---

## 20. Testing strategy

- **~696 backend tests** (pytest), **62 frontend** (Vitest). Dual-backend
  validation (ChromaDB + FAISS) for retrieval/identity/loaders/expansion/cache.
- **Unit:** RRF math, identity hashing, cache primitives, normalization fixtures,
  format loaders, observability primitives.
- **Integration:** end-to-end pipeline (retrieval quality, identity propagation,
  loader→pipeline, expansion, cache, multilingual retrieval), API contracts.
- **Safety:** prompt-injection defenses, encoding-safety across all serialization
  boundaries, Redis-unavailable fallback (via `fakeredis`).
- **Performance:** observability-overhead microbenchmarks (the one test that
  flakes under parallel-suite CPU load — passes at 0.66 µs/op in isolation).
- **Markers:** `integration` and `slow` tests (live Ollama / real model loads)
  are deselected in standard runs.

---

## 21. Limitations & future work

**Current limitations** (also in the README, expanded here):
- **No auth / multi-user.** Single-tenant local-first. Request-id propagation and
  per-doc IDs already support tenant prefixing when this is built.
- **Synchronous ingestion.** Large corpora block an HTTP worker. Loaders are
  isolated from indexing and `_chunks_from_loaded_document` is pure, so an
  async/queued ingestor wraps the current API without restructuring.
- **No rate limiting / hardening** for public deployment.
- **No conversation persistence** — chat is single-turn; `useStreamingAsk` state
  is serializable, so lifting into a store + `/conversations` endpoint is small.
- **Monitoring gaps:** token usage, system resources, standalone embedding-time
  metric.
- **CJK BM25** is whitespace-tokenized (no segmentation).
- **`multilingual_quality`** benchmarked only at small scale (needs GPU).
- **No large-scale retrieval ablation** (MS-MARCO/BEIR) yet — current numbers are
  honest small-corpus regression gates.

**Extension points (architecturally unblocked):**
- Async / queued ingestion with progress events.
- Incremental indexing & content-hash dedup (stable IDs make upsert mechanical).
- Reranker output caching (key builder already exists).
- Per-namespace cache metrics; token-usage + system-resource collectors.
- Runtime model-switch endpoint (settings page is read-only today).
- Agentic SSE event types (`tool_call`/`tool_result`) — the taxonomy extends
  cleanly.
- JSONL trace persistence (wrap the recent-traces ring buffer).
- CJK n-gram tokenizer; multilingual reranker benchmarking.

---

*For the system diagram, per-lifecycle walkthroughs, deployment topology, and
the module map, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For the
phase-by-phase build history with test counts, see [`CHANGELOG.md`](CHANGELOG.md).*
