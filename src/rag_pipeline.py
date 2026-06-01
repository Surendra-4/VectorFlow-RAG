# src/rag_pipeline.py

"""
Complete RAG pipeline integrating all components.

Provides end-to-end Retrieval-Augmented Generation system combining:
- Text embedding with sentence-transformers
- Hybrid retrieval (BM25 + vector search)
- LLM-based text generation with Ollama
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.bm25_retriever import BM25Retriever
from src.cache.caching_embedder import CachingEmbedder
from src.cache.caching_expansion import CachingExpansionPipeline
from src.cache.factory import make_cache
from src.cache.keys import CacheKeys
from src.cache.safe import SafeCache
from src.chunker import TextChunker
from src.config import Settings, get_settings
from src.embedder import Embedder
from src.hybrid_retriever import HybridRetriever
from src.interfaces import RerankerProtocol
from src.llm_client import OllamaClient
from src.logging_setup import configure_from_settings, get_logger
from src.observability import get_metrics
from src.vector_store import VectorStore, make_vector_store

sys.path.append(os.path.dirname(__file__))

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end RAG system combining retrieval and generation."""

    def __init__(
        self,
        index_dir: Optional[str] = None,
        alpha: Optional[float] = None,
        llm_model: Optional[str] = None,
        settings: Optional[Settings] = None,
        reranker: Optional[RerankerProtocol] = None,
        enable_reranker: Optional[bool] = None,
        expansion_pipeline: Optional[Any] = None,
        enable_expansion: Optional[bool] = None,
        cache: Optional[SafeCache] = None,
        enable_cache: Optional[bool] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            index_dir: Directory for storing indices.
                       Defaults to ``indices/rag_system``.
            alpha: Retained for backward compatibility. RRF replaced
                   alpha-fusion; the value is stored as metadata only.
            llm_model: Ollama model name. Defaults to ``settings.llm.model``.
            settings: Optional preconfigured ``Settings``; otherwise pulled
                      from the cached singleton. All other arguments override
                      the resolved settings.
            reranker: Optional pre-built reranker. Useful for tests or to
                      share an instance across pipelines. If ``None`` and
                      reranking is enabled, a ``CrossEncoderReranker`` is
                      lazy-constructed on first use.
            enable_reranker: Override ``settings.reranker.enabled``.
            cache: Optional pre-built SafeCache. Inject your own for tests.
                   If ``None``, a cache is built from settings when
                   ``enable_cache`` resolves to True.
            enable_cache: Override caching enable/disable. ``None`` (default)
                          enables caching iff ``settings.cache.backend != "none"``.
        """
        self.settings = settings or get_settings()
        configure_from_settings(self.settings)

        # Resolve persistence path with cross-platform handling.
        if index_dir is None:
            index_dir_path = Path("indices") / "rag_system"
        else:
            # Tolerate Windows-style separators in user input.
            index_dir_path = Path(str(index_dir).replace("\\", "/"))

        self.index_dir = str(index_dir_path)

        # Concrete values — explicit args win over settings.
        resolved_alpha = self.settings.retrieval.alpha if alpha is None else alpha
        resolved_llm_model = llm_model or self.settings.llm.model

        print("=" * 70)
        print("Initializing VectorFlow-RAG Pipeline")
        print("=" * 70)
        logger.info("Pipeline init: index_dir=%s alpha=%s llm=%s", self.index_dir, resolved_alpha, resolved_llm_model)

        # ----- Cache setup ----------------------------------------------- #
        # Resolved before the embedder so we can wrap it on the spot.
        cache_explicit = cache is not None
        if enable_cache is None:
            cache_on = self.settings.cache.backend != "none"
        else:
            cache_on = bool(enable_cache)
        self.enable_cache = cache_on

        if cache_explicit:
            self.cache = cache
        elif cache_on:
            self.cache = make_cache(self.settings.cache)
        else:
            self.cache = make_cache(self.settings.cache.model_copy(update={"backend": "none"}))

        print("\n[1/5] Loading embedding model...")
        # Construct the embedder from THIS pipeline's settings (not the global
        # singleton) so an injected Settings (e.g. profile=multilingual) is
        # honored. For the default profile these coincide.
        ecfg = self.settings.embedder
        raw_embedder = Embedder(
            model_name=ecfg.model_name,
            device=ecfg.device,
            normalize=ecfg.normalize,
            query_prefix=ecfg.query_prefix,
            passage_prefix=ecfg.passage_prefix,
        )
        # Wrap with CachingEmbedder when caching is on; otherwise use the
        # bare embedder so we avoid a function-call layer.
        self.embedder = (
            CachingEmbedder(raw_embedder, self.cache) if cache_on else raw_embedder
        )

        print("[2/5] Initializing text chunker...")
        self.chunker = TextChunker(
            chunk_size=self.settings.chunker.chunk_size,
            overlap=self.settings.chunker.overlap,
        )

        print("[3/5] Setting up vector store...")
        # Backend-specific subdir keeps each backend's persisted state isolated,
        # so switching VFR_VECTOR_STORE__BACKEND doesn't try to load an
        # incompatible index.
        backend = self.settings.vector_store.backend
        vector_path = str(index_dir_path / backend)
        self.vector_store = make_vector_store(persist_directory=vector_path, backend=backend)

        print("[4/5] Connecting to Ollama LLM...")
        self.llm = OllamaClient(model=resolved_llm_model)

        print("[5/5] Finalizing setup...")
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.corpus: List[str] = []
        self.alpha = resolved_alpha
        self.document_count = 0

        # Reranker is lazy: constructed on first use only if enabled.
        self.enable_reranker = (
            self.settings.reranker.enabled if enable_reranker is None else enable_reranker
        )
        self._reranker: Optional[RerankerProtocol] = reranker

        # Query-expansion is lazy: built on first use only if enabled and
        # caller didn't inject a custom pipeline (useful for tests).
        self.enable_expansion = (
            self.settings.query_expansion.enabled if enable_expansion is None else enable_expansion
        )
        self._expansion_pipeline = expansion_pipeline

        # Corpus fingerprint — set on ingestion; drives retrieval-cache invalidation.
        self.corpus_fingerprint: Optional[str] = None

        print("\n✓ RAG Pipeline Ready!")
        print(f"  - Hybrid alpha: {self.alpha}  (kept for BC; RRF active)")
        print(f"  - RRF k:        {self.settings.retrieval.rrf_k}")
        print(f"  - Candidates:   {self.settings.retrieval.candidates_per_modality} per modality")
        print(f"  - Reranker:     {'on (' + self.settings.reranker.model_name + ')' if self.enable_reranker else 'off'}")
        if self.enable_expansion:
            print(f"  - Expansion:    on (strategies={self.settings.query_expansion.strategies})")
        else:
            print("  - Expansion:    off")
        print(f"  - Cache:        {self.cache.backend_name}")
        print(f"  - LLM model:    {resolved_llm_model}")
        print("=" * 70 + "\n")

    def ingest_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        reset: bool = True,
    ):
        """
        Ingest and index raw text documents.

        Args:
            documents: List of text documents to index
            metadatas: Optional metadata for each document
            reset: Whether to reset existing indices

        For multi-format file ingestion, prefer :meth:`ingest_files`.
        """
        print(f"\n{'='*70}")
        print(f"INGESTING {len(documents)} DOCUMENTS")
        print(f"{'='*70}\n")
        logger.info("Ingesting %d documents (reset=%s)", len(documents), reset)

        print("[Step 1/4] Chunking documents...")
        all_chunks: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents):
            base_meta: Dict[str, Any] = dict(metadatas[i]) if metadatas else {}
            base_meta.setdefault("document_name", f"document_{i}")
            chunks = self.chunker.chunk_text(doc, metadata=base_meta)
            all_chunks.extend(chunks)
            print(f"  ✓ Document {i+1}: {len(chunks)} chunks created")

        print(f"\n  Total chunks created: {len(all_chunks)}")
        self._index_chunks(all_chunks, num_documents=len(documents), reset=reset)

    def ingest_files(
        self,
        paths: List[Any],
        reset: bool = True,
        registry: Optional[Any] = None,
        max_file_size_bytes: Optional[int] = None,
        fail_fast: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Load and ingest files of any registered format.

        Args:
            paths: Iterable of file paths (str or pathlib.Path).
            reset: Reset existing indices before ingesting (default ``True``).
            registry: Optional ``LoaderRegistry`` to use. Defaults to the
                package's :func:`default_registry`.
            max_file_size_bytes: Per-file size cap. Defaults to
                ``settings.ingestion.max_file_size_bytes``.
            fail_fast: If True, raise on the first failing file. Defaults to
                ``settings.ingestion.fail_fast`` (False) — failures are
                collected and reported in the return value.

        Returns:
            ``{"successes": [paths…], "failures": [(path, reason)…],
              "chunks": <int>}`` summary.
        """
        from src.loaders import default_registry as _default_registry
        from src.loaders.base import LoadedDocument, LoaderError

        from pathlib import Path as _Path

        reg = registry or _default_registry()
        ing_cfg = self.settings.ingestion
        size_cap = max_file_size_bytes if max_file_size_bytes is not None else ing_cfg.max_file_size_bytes
        do_fail_fast = ing_cfg.fail_fast if fail_fast is None else fail_fast

        path_list = [_Path(p) for p in paths]

        print(f"\n{'='*70}")
        print(f"INGESTING {len(path_list)} FILE(S)")
        print(f"{'='*70}\n")
        logger.info("Ingesting %d files (reset=%s, fail_fast=%s)", len(path_list), reset, do_fail_fast)

        successes: List[str] = []
        failures: List[Dict[str, str]] = []
        all_chunks: List[Dict[str, Any]] = []

        for path in path_list:
            try:
                # Path existence + size guard before invoking any parser.
                resolved = path.expanduser().resolve(strict=True)
                size = resolved.stat().st_size
                if size > size_cap:
                    raise LoaderError(
                        f"{resolved.name} is {size} bytes; exceeds max_file_size_bytes={size_cap}"
                    )

                loader = reg.find(resolved)
                logger.info("Loading %s with %s", resolved.name, loader.name)
                document: LoadedDocument = loader.load(resolved)

                chunks = self._chunks_from_loaded_document(document)
                all_chunks.extend(chunks)
                successes.append(str(resolved))
                print(f"  ✓ {resolved.name}: {len(document.pages)} page(s), {len(chunks)} chunk(s)")
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                logger.error("Failed to ingest %s: %s", path, reason)
                print(f"  ✗ {path}: {reason}")
                failures.append({"path": str(path), "reason": reason})
                if do_fail_fast:
                    raise

        if all_chunks:
            self._index_chunks(all_chunks, num_documents=len(successes), reset=reset)
        else:
            logger.warning("ingest_files: no chunks produced (all files failed or empty)")
            print("  (warning) no chunks produced; index untouched")

        return {
            "successes": successes,
            "failures": failures,
            "chunks": len(all_chunks),
        }

    def _chunks_from_loaded_document(self, document) -> List[Dict[str, Any]]:
        """
        Chunk a LoadedDocument page-by-page, preserving global chunk_index
        across pages so chunk_ids stay unique within the doc.
        """
        from src.identity import compute_doc_id

        doc_id = compute_doc_id(document.total_text)
        offset = 0
        all_chunks: List[Dict[str, Any]] = []
        for page in document.pages:
            if not page.text or not page.text.strip():
                # Empty pages contribute nothing but the chunk_index offset
                # stays unchanged (they didn't emit any chunk_ids).
                continue
            page_meta = dict(page.metadata)
            page_chunks = self.chunker.chunk_text(
                page.text,
                metadata=page_meta,
                doc_id=doc_id,
                chunk_index_offset=offset,
            )
            offset += len(page_chunks)
            all_chunks.extend(page_chunks)
        return all_chunks

    def _tag_chunk_languages(self, all_chunks: List[Dict[str, Any]]) -> None:
        """
        Annotate each chunk's metadata with an advisory ``language`` code.

        Opt-in (settings.ingestion.detect_language). Advisory only — the tag
        is never read by retrieval; it exists for filtering UIs and debugging.
        Detection failure leaves ``language`` as ``None``.
        """
        from src.language import get_language_detector

        detector = get_language_detector()
        for chunk in all_chunks:
            meta = chunk.get("metadata")
            if isinstance(meta, dict):
                meta["language"] = detector.detect(chunk.get("text", ""))

    def _index_chunks(
        self,
        all_chunks: List[Dict[str, Any]],
        num_documents: int,
        reset: bool,
    ) -> None:
        """
        Common indexing path used by both :meth:`ingest_documents` and
        :meth:`ingest_files`. Embeds, indexes in the vector store, and
        builds the BM25 + hybrid retriever.
        """
        # Optional advisory per-chunk language tagging (off by default).
        # Pure metadata — NEVER affects which chunks are retrieved.
        if self.settings.ingestion.detect_language:
            self._tag_chunk_languages(all_chunks)

        chunk_texts = [chunk["text"] for chunk in all_chunks]
        chunk_metas = [chunk["metadata"] for chunk in all_chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]

        print("\n[Step 2/4] Generating embeddings...")
        # input_type="passage" → asymmetric embedders (e5) apply the passage
        # prefix; symmetric models ignore it (English path unchanged).
        embeddings = self.embedder.encode(chunk_texts, show_progress=True, input_type="passage")
        print(f"  ✓ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

        print("\n[Step 3/4] Building vector index...")
        if reset:
            try:
                self.vector_store.delete_collection()
                print("  ✓ Old collection deleted successfully")
            except Exception as e:
                logger.warning("Could not delete collection: %s", e)
                print(f"  (warning) could not delete collection: {e}")

        self.vector_store = make_vector_store(
            persist_directory=self.vector_store.persist_directory,
            backend=self.settings.vector_store.backend,
        )
        self.vector_store.add_documents(
            texts=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=chunk_metas,
            ids=chunk_ids,
        )
        print(f"  ✓ Vector index built with {len(chunk_texts)} chunks")

        print("\n[Step 4/4] Building BM25 lexical index...")
        bm25_corpus = [
            {"text": text, "chunk_id": cid, "metadata": meta}
            for text, cid, meta in zip(chunk_texts, chunk_ids, chunk_metas)
        ]
        # Honor this pipeline's settings (profile may disable the stemmer).
        self.bm25_retriever = BM25Retriever(
            corpus=bm25_corpus,
            language=self.settings.bm25.language,
            use_stemmer=self.settings.bm25.use_stemmer,
        )
        self.corpus = chunk_texts
        print("  ✓ BM25 index built")

        self.hybrid_retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            alpha=self.alpha,
        )
        self.document_count = num_documents

        # Corpus fingerprint changes on every ingestion. Retrieval-cache
        # entries keyed on the old fingerprint become unreachable
        # immediately; they TTL out without an explicit invalidate call.
        self.corpus_fingerprint = CacheKeys.corpus_fingerprint(chunk_ids)
        logger.info("Corpus fingerprint set to %s", self.corpus_fingerprint)

        print(f"\n{'='*70}")
        print("✓ INGESTION COMPLETE!")
        print(f"  - Documents: {num_documents}")
        print(f"  - Chunks: {len(chunk_texts)}")
        print(f"  - Fingerprint: {self.corpus_fingerprint}")
        print("  - Ready for search and Q&A")
        print(f"{'='*70}\n")
        logger.info("Ingestion complete: %d documents, %d chunks", num_documents, len(chunk_texts))

    def _get_reranker(self) -> Optional[RerankerProtocol]:
        """Lazy-construct the reranker on first use, if enabled."""
        if not self.enable_reranker:
            return None
        if self._reranker is None:
            # Local import keeps the cross-encoder dependency optional —
            # users with reranking disabled never trigger it.
            from src.reranker import CrossEncoderReranker

            # Honor this pipeline's settings (profile selects the reranker model).
            rcfg = self.settings.reranker
            self._reranker = CrossEncoderReranker(
                model_name=rcfg.model_name,
                device=rcfg.device,
                top_n=rcfg.top_n,
            )
        return self._reranker

    def _get_expansion_pipeline(self):
        """Lazy-construct the expansion pipeline if enabled."""
        if not self.enable_expansion:
            return None
        if self._expansion_pipeline is None:
            # Local imports — expansion deps only loaded when needed.
            from src.query_expansion import (
                HyDEExpander,
                MultiQueryExpander,
                QueryExpansionPipeline,
            )

            qcfg = self.settings.query_expansion
            strategies = []
            for name in qcfg.strategies:
                if name == "multi_query":
                    strategies.append(MultiQueryExpander())
                elif name == "hyde":
                    strategies.append(HyDEExpander())
                else:
                    logger.warning("Unknown expansion strategy in config: %s", name)
            base_pipeline = QueryExpansionPipeline(
                strategies=strategies,
                max_query_length=qcfg.max_query_length,
            )
            self._expansion_pipeline = self._maybe_wrap_expansion(base_pipeline)
        return self._expansion_pipeline

    def _maybe_wrap_expansion(self, pipeline):
        """Wrap the expansion pipeline with caching when caching is enabled."""
        if not self.enable_cache or self.cache.backend_name == "null":
            return pipeline
        qcfg = self.settings.query_expansion
        cache_params = {
            "n_variants": qcfg.multi_query_count,
            "n_hyde": qcfg.hyde_count,
            "strategies": sorted(qcfg.strategies),
            "max_query_length": qcfg.max_query_length,
        }
        model_for_key = qcfg.expansion_model or self.settings.llm.model
        return CachingExpansionPipeline(
            pipeline,
            cache=self.cache,
            cache_key_model=model_for_key,
            cache_key_params=cache_params,
            ttl_s=self.settings.cache.ttl_expansion_s,
        )

    def _retrieval_cache_key(self, query: str, k: int) -> Optional[str]:
        """Build a stable retrieval-cache key for ``(query, k, current config)``."""
        if not self.enable_cache or self.cache.backend_name == "null":
            return None
        if self.corpus_fingerprint is None:
            return None
        # Anything that materially affects the output of search() goes here.
        # Adding a new knob? Add it to this dict OR bump the schema version
        # in CacheKeys.retrieval to invalidate older entries.
        qcfg = self.settings.query_expansion
        config = {
            "candidates_per_modality": self.settings.retrieval.candidates_per_modality,
            "rrf_k": self.settings.retrieval.rrf_k,
            "expansion_enabled": self.enable_expansion,
            "expansion_strategies": sorted(qcfg.strategies) if self.enable_expansion else [],
            "expansion_multi_query_count": qcfg.multi_query_count if self.enable_expansion else 0,
            "expansion_hyde_count": qcfg.hyde_count if self.enable_expansion else 0,
            "reranker_enabled": self.enable_reranker,
            "reranker_model": self.settings.reranker.model_name if self.enable_reranker else None,
            "reranker_top_n": self.settings.reranker.top_n if self.enable_reranker else None,
        }
        return CacheKeys.retrieval(self.corpus_fingerprint, config, query, k)

    def search(
        self,
        query: str,
        k: int = 5,
        return_trace: bool = False,
    ):
        """
        Retrieve the top-``k`` chunks for ``query``.

        Three modes, in order of preference based on config:

        1. **Expansion + RRF + (optional) rerank** — when expansion is enabled.
           Multiple query variants and optional HyDE documents all feed RRF
           together; the unified pool is optionally reranked.
        2. **Hybrid + rerank** — single query, retrieve a wider pool, rerank.
        3. **Hybrid only** — single query, RRF over vector+bm25.

        Args:
            query: user query
            k: final result count
            return_trace: if True, return ``(results, RetrievalTrace)``.
                Default False preserves the legacy ``List[Dict]`` return.
        """
        if self.hybrid_retriever is None:
            raise ValueError("❌ No documents ingested! Call ingest_documents() first.")

        from src.retrieval_trace import RetrievalTrace

        print(f"\n🔍 Searching for: '{query}'")
        t0 = time.time()
        trace = RetrievalTrace(original_query=query, sanitized_query=query)

        # --- Full-retrieval cache check ------------------------------------ #
        # Single cheapest path on a hit: skip expansion + retrieval + rerank
        # entirely. Cache key includes corpus_fingerprint so re-ingesting
        # invalidates implicitly.
        cache_key = self._retrieval_cache_key(query, k)
        if cache_key is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                trace.from_cache = True
                trace.final_results = list(cached)
                trace.total_latency_ms = (time.time() - t0) * 1000
                trace.cache_stats = self.cache.stats.snapshot()
                logger.debug("search cache HIT key=%s", cache_key)
                print(f"✓ Found {len(cached)} results (cached)\n")
                self._record_search_metrics(trace)
                if return_trace:
                    return cached, trace
                return cached

        expansion = self._get_expansion_pipeline()
        reranker = self._get_reranker()

        if expansion is not None:
            results = self._expanded_search(query, k=k, reranker=reranker, trace=trace)
        elif reranker is not None:
            candidate_k = max(k, self.settings.retrieval.candidates_per_modality)
            candidates = self.hybrid_retriever.search(query, k=candidate_k)
            trace.fused_pool = list(candidates)
            trace.reranker_used = True
            trace.rerank_input_size = len(candidates)
            t_rr = time.perf_counter()
            results = reranker.rerank(query, candidates, top_n=k)
            trace.rerank_latency_ms = (time.perf_counter() - t_rr) * 1000
            trace.rerank_output_size = len(results)
        else:
            results = self.hybrid_retriever.search(query, k=k)
            trace.fused_pool = list(results)

        trace.final_results = list(results)
        trace.total_latency_ms = (time.time() - t0) * 1000

        # Populate cache stats snapshot before any post-store mutations.
        trace.cache_stats = self.cache.stats.snapshot()

        # Store final results in cache for next call.
        if cache_key is not None:
            self.cache.set(cache_key, list(results), ttl=self.settings.cache.ttl_retrieval_s)

        logger.debug(
            "search query=%r k=%d returned=%d expansion=%s reranked=%s",
            query, k, len(results),
            expansion is not None,
            reranker is not None,
        )
        print(f"✓ Found {len(results)} results\n")

        self._record_search_metrics(trace)

        if return_trace:
            return results, trace
        return results

    def _record_search_metrics(self, trace) -> None:
        """
        Feed metrics from a completed RetrievalTrace.

        Centralized here so cache-hit and cache-miss paths agree on what
        gets emitted. Metric path is defensive — any exception here is
        swallowed so observability never breaks a response.
        """
        try:
            m = get_metrics()
            m.retrievals_total.inc()
            m.retrieval_latency_ms.observe(trace.total_latency_ms)

            if trace.from_cache:
                m.cache_hits_total.inc()

            # Per-stage histograms only when the stage actually ran.
            if trace.expansion_latency_ms > 0:
                m.retrieval_stage_latency_ms.observe(
                    "expansion", value=trace.expansion_latency_ms
                )
            if trace.fusion_latency_ms > 0:
                m.retrieval_stage_latency_ms.observe(
                    "fusion", value=trace.fusion_latency_ms
                )
            if trace.reranker_used:
                m.reranker_used_total.inc()
                if trace.rerank_latency_ms > 0:
                    m.retrieval_stage_latency_ms.observe(
                        "rerank", value=trace.rerank_latency_ms
                    )

            for strategy in trace.strategies_used:
                m.expansion_strategy_usage_total.inc(strategy)

            # Cache namespace ops — attribute the delta in the per-request
            # snapshot to namespace="full_retrieval". Finer-grained
            # namespace metrics would require per-wrapper emission; this
            # gives us "did the cache help on THIS request" which is
            # what dashboards typically want.
            if trace.from_cache:
                m.cache_ops_total.inc("full_retrieval", "hit")
            elif trace.cache_stats:
                # Best-effort: surface the process-wide stat shape.
                pass

            # Push trace itself into the ring buffer for /traces/recent.
            m.recent_traces.append(trace.to_dict())
        except Exception as exc:  # pragma: no cover — metric path must never fail responses
            logger.debug("metric recording suppressed an exception: %s", exc)

    def _expanded_search(self, query: str, k: int, reranker, trace) -> List[Dict[str, Any]]:
        """
        Expanded-retrieval path: multi-query + HyDE → unified RRF → optional rerank.

        Each retrieval input (original query, variant query, HyDE document)
        produces ranked candidate lists keyed by ``chunk_id`` for vector
        retrieval, and by chunk_id for BM25. All lists are RRF-fused in a
        single call, exploiting the same fusion infrastructure used in
        single-query mode.
        """
        from src.retrieval_trace import StrategyCandidates
        from src.rrf import rrf_with_ranks

        expansion = self._get_expansion_pipeline()
        assert expansion is not None  # guarded by caller

        # Stage 1: expansion
        expanded = expansion.expand(query)
        trace.sanitized_query = expanded.original
        trace.expanded_queries = list(expanded.queries)
        trace.hyde_documents = list(expanded.hyde_documents)
        trace.strategies_used = list(expanded.strategies_used)
        trace.expansion_errors = list(expanded.errors)
        trace.expansion_latency_ms = expanded.expansion_latency_ms

        # Stage 2: per-input retrieval
        pool_size = max(
            self.settings.retrieval.candidates_per_modality,
            k,
        )
        # ID-keyed maps to hydrate the final result objects.
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        ranked_lists: Dict[str, List[str]] = {}

        def _ingest_candidates(label: str, modality: str, source: str, items: List[Dict[str, Any]]):
            ids = []
            for item in items:
                cid = item.get("chunk_id")
                if not cid:
                    continue
                ids.append(cid)
                if cid not in chunk_lookup:
                    chunk_lookup[cid] = item
            ranked_lists[label] = ids
            trace.per_strategy.append(StrategyCandidates(
                label=label, modality=modality, source_query=source, candidates=items,
            ))

        # Per query string: run hybrid retrieval (which internally does its
        # own small RRF over vector+bm25). Then re-fuse across all queries.
        # We approximate per-modality ranked lists by inspecting the
        # vector_rank / bm25_rank fields the hybrid retriever attaches.
        for idx, q in enumerate(expanded.queries):
            label_prefix = "original" if idx == 0 else f"variant_{idx}"
            t = time.perf_counter()
            hits = self.hybrid_retriever.search(q, k=pool_size)
            latency_ms = (time.perf_counter() - t) * 1000

            # Split into per-modality ranked lists for downstream RRF.
            vector_hits = sorted(
                [h for h in hits if h.get("vector_rank") is not None],
                key=lambda h: h["vector_rank"],
            )
            bm25_hits = sorted(
                [h for h in hits if h.get("bm25_rank") is not None],
                key=lambda h: h["bm25_rank"],
            )
            _ingest_candidates(f"vector:{label_prefix}", "vector", q, vector_hits)
            _ingest_candidates(f"bm25:{label_prefix}", "bm25", q, bm25_hits)
            trace.per_strategy[-1].latency_ms = latency_ms
            trace.per_strategy[-2].latency_ms = latency_ms

        # HyDE documents: vector-only side input, embedded then searched.
        # A HyDE doc stands in for the query, so embed it with the query
        # input_type (matters only for asymmetric embedders like e5).
        for idx, hyde_doc in enumerate(expanded.hyde_documents):
            t = time.perf_counter()
            emb = self.embedder.encode(hyde_doc, show_progress=False, input_type="query")[0]
            raw = self.vector_store.search(emb, n_results=pool_size)
            latency_ms = (time.perf_counter() - t) * 1000
            # Rehydrate to the shape hybrid_retriever produces.
            hyde_hits: List[Dict[str, Any]] = []
            for cid, doc, dist, meta in zip(
                raw.get("ids", []),
                raw.get("documents", []),
                raw.get("distances", []),
                raw.get("metadatas", []),
            ):
                hyde_hits.append({
                    "text": doc,
                    "chunk_id": cid,
                    "doc_id": (meta or {}).get("document_id") if meta else None,
                    "hybrid_score": None,
                    "vector_rank": None,
                    "bm25_rank": None,
                    "metadata": meta or {},
                    "document_name": (meta or {}).get("document_name"),
                    "source_path": (meta or {}).get("source_path"),
                    "page_number": (meta or {}).get("page_number"),
                    "chunk_index": (meta or {}).get("chunk_index"),
                })
            _ingest_candidates(f"vector:hyde_{idx}", "vector", hyde_doc[:80] + "…", hyde_hits)
            trace.per_strategy[-1].latency_ms = latency_ms

        # Stage 3: unified RRF over every ranked list we collected.
        t_fuse = time.perf_counter()
        fused = rrf_with_ranks(ranked_lists, k=self.settings.retrieval.rrf_k)
        trace.fusion_latency_ms = (time.perf_counter() - t_fuse) * 1000

        # Rehydrate the fused IDs into full result dicts, preserving provenance.
        results: List[Dict[str, Any]] = []
        for entry in fused:
            cid = entry["item"]
            base = chunk_lookup.get(cid, {})
            results.append({
                **base,
                "chunk_id": cid,
                "hybrid_score": float(entry["score"]),
                "rrf_score": float(entry["score"]),
                "vector_rank": base.get("vector_rank"),
                "bm25_rank": base.get("bm25_rank"),
            })

        trace.fused_pool = list(results)

        # Stage 4: optional reranking on the unified pool.
        if reranker is not None:
            trace.reranker_used = True
            trace.rerank_input_size = len(results)
            t_rr = time.perf_counter()
            results = reranker.rerank(query, results, top_n=k)
            trace.rerank_latency_ms = (time.perf_counter() - t_rr) * 1000
            trace.rerank_output_size = len(results)
        else:
            results = results[:k]

        return results

    def ask(
        self,
        question: str,
        k_docs: int = 3,
        return_sources: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """RAG: retrieve relevant documents and generate an answer."""
        start_time = time.time()

        if verbose:
            print(f"\n{'='*70}")
            print("RAG QUERY")
            print(f"{'='*70}")
            print(f"Question: {question}\n")

        if verbose:
            print("[1/2] Retrieving relevant context...")

        results = self.search(question, k=k_docs)
        retrieval_time = time.time() - start_time

        if verbose:
            print("Retrieved documents:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. (score: {r['hybrid_score']:.3f}) {r['text'][:80]}...")

        context_docs = [r["text"] for r in results]

        if verbose:
            print("\n[2/2] Generating answer with LLM...")

        gen_start = time.time()
        answer = self.llm.generate(prompt=question, context=context_docs, max_tokens=512, temperature=0.7)
        generation_time = time.time() - gen_start
        total_time = time.time() - start_time

        if verbose:
            print(f"\n✓ Answer generated in {generation_time:.2f}s")
            print(f"{'='*70}\n")

        response = {
            "question": question,
            "answer": answer,
            "metrics": {
                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                "generation_time_ms": round(generation_time * 1000, 2),
                "total_time_ms": round(total_time * 1000, 2),
                "num_context_docs": len(context_docs),
                "alpha": self.alpha,
            },
        }

        if return_sources:
            response["sources"] = results

        return response

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "documents_ingested": self.document_count,
            "chunks_indexed": len(self.corpus),
            "embedding_dimension": self.embedder.dimension,
            "alpha": self.alpha,  # legacy metadata; RRF is active
            "rrf_k": self.settings.retrieval.rrf_k,
            "candidates_per_modality": self.settings.retrieval.candidates_per_modality,
            "reranker_enabled": self.enable_reranker,
            "reranker_model": self.settings.reranker.model_name if self.enable_reranker else None,
            "model": self.embedder.model_name,
            "vector_store_backend": self.settings.vector_store.backend,
            "chat_provider": getattr(self.llm, "name", "ollama"),
            "chat_model": getattr(self.llm, "model", None),
        }

    # ------------------------------------------------------------------ #
    # Runtime mutation surface (Phase 12c)
    # ------------------------------------------------------------------ #

    def _ensure_owns_settings(self) -> None:
        """Detach from a shared ``Settings`` singleton before any mutation.

        We deep-copy lazily — the only allocations are when a runtime change is
        first applied, so default deployments stay byte-identical.
        """
        if not getattr(self, "_owns_settings", False):
            self.settings = self.settings.model_copy(deep=True)
            self._owns_settings = True

    def set_chat_provider(self, chat_cfg, secret_store=None) -> None:
        """Replace the active chat provider live, without restarting.

        ``chat_cfg`` is a :class:`src.providers.ChatModelConfig`. The provider
        is built through :func:`make_chat_provider` so the API-key lookup,
        capability gating, and registry checks are uniform.
        """
        from src.providers import make_chat_provider

        new_llm = make_chat_provider(chat_cfg, secret_store=secret_store)
        # Drop the cached expansion pipeline — it embeds the previous LLM in
        # its strategies. Rebuild lazily on next use.
        self._expansion_pipeline = None

        self._ensure_owns_settings()
        self.settings.llm.model = chat_cfg.model
        if chat_cfg.base_url:
            self.settings.llm.base_url = chat_cfg.base_url
        self.settings.llm.max_tokens = chat_cfg.max_tokens
        self.settings.llm.temperature = chat_cfg.temperature
        self.settings.llm.request_timeout_s = chat_cfg.request_timeout_s

        self.llm = new_llm
        logger.info(
            "Active chat provider switched: provider=%s model=%s",
            chat_cfg.provider, chat_cfg.model,
        )

    def apply_live_settings(self, live) -> None:
        """Apply a :class:`LiveQuerySettings` snapshot without rebuilding the index.

        Only the safe / live-query knobs flow through here. Anything that would
        change embedding dimensions / chunking / vector topology lives in
        ``IndexConstructionSettings`` and is applied via the IndexManager.
        """
        self._ensure_owns_settings()

        # ---- Retrieval knobs ---- #
        self.settings.retrieval.k_default = live.retrieval_k_default
        self.settings.retrieval.candidates_per_modality = live.retrieval_candidates_per_modality
        self.settings.retrieval.rrf_k = live.retrieval_rrf_k

        # ---- Reranker ---- #
        # If the model changed (or it was off and is now on), drop the lazy
        # instance so the next query rebuilds with the new config.
        prev_model = self.settings.reranker.model_name
        self.settings.reranker.enabled = live.reranker.enabled
        self.settings.reranker.model_name = live.reranker.model
        self.settings.reranker.top_n = live.reranker.top_n
        self.settings.reranker.device = live.reranker.device
        self.enable_reranker = live.reranker_enabled
        if prev_model != live.reranker.model:
            self._reranker = None

        # ---- Query expansion ---- #
        prev_strategies = sorted(self.settings.query_expansion.strategies)
        self.settings.query_expansion.enabled = live.expansion_enabled
        self.settings.query_expansion.strategies = list(live.expansion_strategies)
        self.settings.query_expansion.multi_query_count = live.expansion_multi_query_count
        self.settings.query_expansion.hyde_count = live.expansion_hyde_count
        self.enable_expansion = live.expansion_enabled
        if prev_strategies != sorted(live.expansion_strategies):
            self._expansion_pipeline = None

        logger.info(
            "Live settings applied: reranker=%s expansion=%s rrf_k=%d candidates=%d",
            self.enable_reranker, self.enable_expansion,
            self.settings.retrieval.rrf_k,
            self.settings.retrieval.candidates_per_modality,
        )


# =============================================================================
# DEMO: Test the complete pipeline
# =============================================================================


def demo():
    """Run a complete demo of VectorFlow-RAG."""

    print("\n" + "=" * 70)
    print(" VectorFlow-RAG DEMO ".center(70, "="))
    print("=" * 70)

    documents = [
        (
            "VectorFlow-RAG is a production-grade, modular semantic retrieval and RAG framework. "
            "It is designed for machine learning engineers and researchers who want full transparency and control over their search pipeline. "
            "Unlike black-box solutions like LangChain or ElasticSearch, VectorFlow exposes every internal layer: embedding, indexing, ranking, and generation. "
            "The system is built to be reproducible, swappable, and benchmarkable."
        ),
        (
            "The embedding layer uses state-of-the-art models from sentence-transformers. "
            "By default, it uses all-MiniLM-L6-v2, a lightweight 80MB model with 384 dimensions. "
            "These models convert text into high-dimensional vectors that capture semantic meaning. "
            "The embeddings are then stored in efficient vector databases like ChromaDB and FAISS. "
            "Users can easily swap embedding models through YAML configuration files."
        ),
        (
            "Hybrid search is the core retrieval strategy in VectorFlow-RAG. "
            "It combines BM25 lexical matching with vector similarity search. "
            "BM25 is excellent at exact keyword matches and handles rare terms well, while vector search captures semantic similarity and understands context. "
            "The fusion parameter alpha controls the balance between the two approaches, where alpha=0.5 gives equal weight to both methods."
        ),
        (
            "For text generation, VectorFlow-RAG uses Ollama to run large language models locally. "
            "This approach ensures complete privacy and eliminates API costs. "
            "Popular models include TinyLlama (600MB), Llama 3.2 (1.3GB), and Mistral (4GB), which can all run on consumer hardware. "
            "The system supports any Ollama model through simple configuration changes. "
            "Generation is combined with retrieved context to produce grounded, factual answers."
        ),
        (
            "The framework includes comprehensive experiment tracking using MLflow and DagsHub. "
            "All experiments are automatically logged with metrics like NDCG, MRR, recall@k, and latency. "
            "This makes VectorFlow-RAG suitable for research applications and systematic ablation studies. "
            "Users can compare different embedding models, chunking strategies, and alpha parameters. "
            "The system is fully reproducible with versioned configurations and deterministic results."
        ),
        (
            "VectorFlow-RAG is built with modern ML engineering practices. "
            "It uses FastAPI for REST API endpoints, Docker for containerization, and GitHub Actions for CI/CD. "
            "The system can be deployed to free platforms like Streamlit Community Cloud, Render, or Hugging Face Spaces. "
            "All components are modular and can be replaced independently. "
            "The codebase includes comprehensive tests and follows production-ready design patterns."
        ),
    ]

    rag = RAGPipeline(
        index_dir=str(Path("indices") / "demo_rag"),
        alpha=0.5,
        llm_model="tinyllama",
    )

    rag.ingest_documents(documents)

    print("\nPipeline Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n\n" + "=" * 70)
    print(" TEST 1: SEARCH ONLY ".center(70, "="))
    print("=" * 70)

    search_query = "What embedding models does VectorFlow use?"
    search_results = rag.search(search_query, k=3)

    print("Top 3 Results:")
    for i, result in enumerate(search_results, 1):
        print(f"\n{i}. Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"   Text: {result['text'][:150]}...")

    print("\n\n" + "=" * 70)
    print(" TEST 2: RAG (RETRIEVAL + GENERATION) ".center(70, "="))
    print("=" * 70)

    questions = [
        "What is VectorFlow-RAG and what makes it different?",
        "How does hybrid search work in this system?",
        "What LLM options are available?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n\n--- Question {i} ---")
        response = rag.ask(question, k_docs=2, verbose=True)

        print("ANSWER:")
        print("-" * 70)
        print(response["answer"])
        print("-" * 70)

        print("\nPERFORMANCE METRICS:")
        for metric, value in response["metrics"].items():
            print(f"  {metric}: {value}")

        print("\nSOURCE DOCUMENTS:")
        for j, source in enumerate(response["sources"], 1):
            print(f"  {j}. (score: {source['hybrid_score']:.3f}) {source['text'][:80]}...")

    print("\n\n" + "=" * 70)
    print(" DEMO COMPLETE! ".center(70, "="))
    print("=" * 70)
    print("\n✓ VectorFlow-RAG is working perfectly!")
    print("✓ You can now use this for your own documents")
    print("✓ Next: Build a Streamlit UI for interactive demos\n")


if __name__ == "__main__":
    demo()
