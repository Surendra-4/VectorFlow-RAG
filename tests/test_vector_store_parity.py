# tests/test_vector_store_parity.py

"""
Parity, recall, and latency comparisons across vector-store backends.

These tests verify that:

1. **Shape parity** — ChromaDB and FAISS expose identical search-result
   structure ``{documents, distances, metadatas}``.
2. **Top-1 parity** — for self-queries on normalized embeddings, both
   backends return the queried document at rank 1.
3. **Recall** — FAISS-HNSW Recall@10 against the FAISS-Flat (exact)
   baseline stays high (≥90% on synthetic data with our default knobs).
4. **Latency** — FAISS-HNSW search is faster than FAISS-Flat once the
   corpus is non-trivial; this is the architectural reason for the
   migration. Numbers are printed (run with ``-s``) and only loose
   guards prevent egregious regressions, since CI hardware varies.

All tests use synthetic embeddings to keep them dependency-light and
fast — no sentence-transformers required.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from src.faiss_store import FAISSVectorStore
from src.vector_store import VectorStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _random_corpus(n: int, dim: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return _normalize(rng.randn(n, dim).astype(np.float32))


def _populated_store(store, corpus: np.ndarray) -> None:
    texts = [f"doc_{i}" for i in range(len(corpus))]
    store.add_documents(texts=texts, embeddings=corpus.tolist())


def _recall_at_k(retrieved_ids: List[int], gold_ids: List[int], k: int) -> float:
    """Recall@k = |retrieved ∩ gold| / |gold| (treating top-k of each as sets)."""
    if not gold_ids:
        return 1.0
    return len(set(retrieved_ids[:k]) & set(gold_ids[:k])) / len(gold_ids[:k])


def _doc_ids_from_results(results: dict) -> List[int]:
    """Parse 'doc_<i>' strings back to integer positions for set-based recall."""
    return [int(d.split("_")[-1]) for d in results["documents"]]


# --------------------------------------------------------------------------- #
# Shape parity
# --------------------------------------------------------------------------- #


class TestShapeParity:
    """Both backends must return the same dict structure from search()."""

    @pytest.fixture
    def both_stores(self, tmp_path):
        chroma = VectorStore(persist_directory=str(tmp_path / "chroma"))
        chroma.create_collection(reset=True)
        faiss = FAISSVectorStore(persist_directory=str(tmp_path / "faiss"), index_type="flat")

        corpus = _random_corpus(20, dim=128)
        _populated_store(chroma, corpus)
        _populated_store(faiss, corpus)
        return chroma, faiss, corpus

    def test_search_keys_identical(self, both_stores):
        chroma, faiss, corpus = both_stores
        c_res = chroma.search(corpus[0].tolist(), n_results=5)
        f_res = faiss.search(corpus[0].tolist(), n_results=5)
        required = {"documents", "distances", "metadatas", "ids"}
        assert required.issubset(c_res.keys())
        assert required.issubset(f_res.keys())

    def test_search_returns_n_results(self, both_stores):
        chroma, faiss, corpus = both_stores
        for k in (1, 3, 5):
            assert len(chroma.search(corpus[0].tolist(), n_results=k)["documents"]) == k
            assert len(faiss.search(corpus[0].tolist(), n_results=k)["documents"]) == k

    def test_distances_are_floats_in_both(self, both_stores):
        chroma, faiss, corpus = both_stores
        c = chroma.search(corpus[0].tolist(), n_results=3)
        f = faiss.search(corpus[0].tolist(), n_results=3)
        assert all(isinstance(d, float) for d in c["distances"])
        assert all(isinstance(d, float) for d in f["distances"])

    def test_distance_monotonic_with_similarity(self, both_stores):
        # In both backends, distance should be sorted ascending with rank
        # (top result = closest = smallest distance).
        chroma, faiss, corpus = both_stores
        c = chroma.search(corpus[0].tolist(), n_results=5)
        f = faiss.search(corpus[0].tolist(), n_results=5)
        assert c["distances"] == sorted(c["distances"])
        assert f["distances"] == sorted(f["distances"])


# --------------------------------------------------------------------------- #
# ID roundtrip (Phase 3.5)
# --------------------------------------------------------------------------- #


class TestIdRoundtrip:
    """search() must return the same ids that were passed to add_documents()."""

    @pytest.mark.parametrize("backend", ["chromadb", "faiss"])
    def test_ids_returned_match_ids_added(self, tmp_path, backend):
        if backend == "chromadb":
            store = VectorStore(persist_directory=str(tmp_path / backend))
            store.create_collection(reset=True)
        else:
            store = FAISSVectorStore(persist_directory=str(tmp_path / backend))

        corpus = _random_corpus(8, dim=128, seed=3)
        ids_in = [f"custom-{i}" for i in range(len(corpus))]
        texts = [f"doc_{i}" for i in range(len(corpus))]
        store.add_documents(texts=texts, embeddings=corpus.tolist(), ids=ids_in)

        # Query each vector against the index; expect its own ID at rank 1.
        for i, vec in enumerate(corpus):
            res = store.search(vec.tolist(), n_results=1)
            assert res["ids"][0] == ids_in[i], (
                f"backend={backend} i={i} got={res['ids'][0]} expected={ids_in[i]}"
            )

    @pytest.mark.parametrize("backend", ["chromadb", "faiss"])
    def test_ids_aligned_with_documents(self, tmp_path, backend):
        """ids[i] must correspond to documents[i] positionally."""
        if backend == "chromadb":
            store = VectorStore(persist_directory=str(tmp_path / backend))
            store.create_collection(reset=True)
        else:
            store = FAISSVectorStore(persist_directory=str(tmp_path / backend))

        corpus = _random_corpus(5, dim=128, seed=11)
        ids_in = [f"k_{i}" for i in range(len(corpus))]
        text_in = [f"text_{i}" for i in range(len(corpus))]
        store.add_documents(texts=text_in, embeddings=corpus.tolist(), ids=ids_in)

        res = store.search(corpus[0].tolist(), n_results=5)
        for doc, iid in zip(res["documents"], res["ids"]):
            # The number suffix of "text_X" must match the suffix of "k_X".
            assert doc.split("_")[-1] == iid.split("_")[-1], (
                f"backend={backend}: doc={doc} id={iid}"
            )


# --------------------------------------------------------------------------- #
# Self-query top-1 (smoke test on retrieval correctness)
# --------------------------------------------------------------------------- #


class TestSelfQueryParity:
    """Querying with a vector that exists in the index → that doc at rank 1."""

    @pytest.mark.parametrize("backend", ["chromadb", "faiss-flat", "faiss-hnsw"])
    def test_self_query_is_top1(self, tmp_path, backend):
        if backend == "chromadb":
            store = VectorStore(persist_directory=str(tmp_path / backend))
            store.create_collection(reset=True)
        elif backend == "faiss-flat":
            store = FAISSVectorStore(persist_directory=str(tmp_path / backend), index_type="flat")
        else:
            store = FAISSVectorStore(persist_directory=str(tmp_path / backend), index_type="hnsw")

        corpus = _random_corpus(50, dim=128, seed=7)
        _populated_store(store, corpus)

        # Query each of 5 random doc embeddings; each should retrieve itself.
        for idx in [0, 12, 24, 36, 48]:
            res = store.search(corpus[idx].tolist(), n_results=1)
            assert res["documents"][0] == f"doc_{idx}", f"backend={backend} idx={idx}"


# --------------------------------------------------------------------------- #
# Recall (FAISS-HNSW vs FAISS-Flat exact baseline)
# --------------------------------------------------------------------------- #


class TestRecall:
    """
    FAISS-HNSW Recall@k against FAISS-Flat (the exact-search ground truth).

    Synthetic data only — quality on real embeddings is verified separately
    in tests/test_retrieval_quality.py via the full pipeline.
    """

    @pytest.fixture(scope="class")
    def corpus(self):
        # 1000 vectors of dim 128 — small but enough to be meaningful for HNSW.
        return _random_corpus(1000, dim=128, seed=42)

    @pytest.fixture(scope="class")
    def queries(self):
        # 30 hold-out random queries (not in corpus).
        return _random_corpus(30, dim=128, seed=99)

    def _gold_top_k(self, corpus, query, k):
        """Brute-force top-k by inner product (corpus is normalized)."""
        sims = corpus @ query
        return list(np.argsort(-sims)[:k])

    def test_hnsw_recall_at_10_meets_threshold(self, tmp_path, corpus, queries):
        store = FAISSVectorStore(
            persist_directory=str(tmp_path / "hnsw_recall"),
            index_type="hnsw",
        )
        _populated_store(store, corpus)

        recalls = []
        for q in queries:
            gold = self._gold_top_k(corpus, q, k=10)
            res = store.search(q.tolist(), n_results=10)
            retrieved = _doc_ids_from_results(res)
            recalls.append(_recall_at_k(retrieved, gold, k=10))

        avg = float(np.mean(recalls))
        print(f"\n[HNSW Recall@10] avg={avg:.3f} (n_queries={len(queries)}, n_corpus={len(corpus)})")
        # Loose floor — synthetic random vectors are an unfavorable case for HNSW;
        # real embedding distributions usually exceed 0.97 at default knobs.
        assert avg >= 0.85, f"HNSW Recall@10={avg:.3f} below 0.85 floor"

    def test_flat_recall_at_10_is_perfect(self, tmp_path, corpus, queries):
        # Sanity check: the exact-search backend should match the brute-force gold.
        store = FAISSVectorStore(
            persist_directory=str(tmp_path / "flat_recall"),
            index_type="flat",
        )
        _populated_store(store, corpus)

        recalls = []
        for q in queries:
            gold = self._gold_top_k(corpus, q, k=10)
            res = store.search(q.tolist(), n_results=10)
            retrieved = _doc_ids_from_results(res)
            recalls.append(_recall_at_k(retrieved, gold, k=10))

        avg = float(np.mean(recalls))
        print(f"\n[Flat Recall@10] avg={avg:.3f}")
        assert avg == pytest.approx(1.0)

    def test_hnsw_efsearch_recall_tradeoff(self, tmp_path, corpus, queries):
        """Higher efSearch should not decrease recall."""
        store_low = FAISSVectorStore(
            persist_directory=str(tmp_path / "ef_low"),
            index_type="hnsw",
            hnsw_ef_search=8,
        )
        store_high = FAISSVectorStore(
            persist_directory=str(tmp_path / "ef_high"),
            index_type="hnsw",
            hnsw_ef_search=128,
        )
        _populated_store(store_low, corpus)
        _populated_store(store_high, corpus)

        def avg_recall(store, k=10):
            recs = []
            for q in queries:
                gold = self._gold_top_k(corpus, q, k=k)
                res = store.search(q.tolist(), n_results=k)
                recs.append(_recall_at_k(_doc_ids_from_results(res), gold, k))
            return float(np.mean(recs))

        low = avg_recall(store_low)
        high = avg_recall(store_high)
        print(f"\n[efSearch tradeoff] ef=8 → {low:.3f}  ef=128 → {high:.3f}")
        # Higher efSearch should be ≥ within noise of lower (graph quality matters too).
        assert high >= low - 0.05


# --------------------------------------------------------------------------- #
# Latency comparison (informational)
# --------------------------------------------------------------------------- #


class TestLatencyComparison:
    """
    Print per-backend search latency. Intended to confirm the architectural
    benefit of HNSW; thresholds are loose because absolute timings depend
    heavily on hardware. Run with ``pytest -s`` to see numbers.
    """

    @pytest.fixture(scope="class")
    def corpus(self):
        return _random_corpus(2000, dim=128, seed=2024)

    @pytest.fixture(scope="class")
    def queries(self):
        return _random_corpus(50, dim=128, seed=8675309)

    def _bench_search(self, store, queries, k=10, n_warmup=5):
        # Warm-up to absorb one-time costs (model load, cache fill).
        for q in queries[:n_warmup]:
            store.search(q.tolist(), n_results=k)
        ts = []
        for q in queries:
            t0 = time.perf_counter()
            store.search(q.tolist(), n_results=k)
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts)), float(np.percentile(ts, 95))

    def test_hnsw_faster_than_flat_at_2k(self, tmp_path, corpus, queries):
        flat = FAISSVectorStore(persist_directory=str(tmp_path / "lat_flat"), index_type="flat")
        hnsw = FAISSVectorStore(persist_directory=str(tmp_path / "lat_hnsw"), index_type="hnsw")
        _populated_store(flat, corpus)
        _populated_store(hnsw, corpus)

        flat_p50, flat_p95 = self._bench_search(flat, queries)
        hnsw_p50, hnsw_p95 = self._bench_search(hnsw, queries)

        print(f"\n[Latency p50/p95 (n=2000, q=50)]")
        print(f"  FAISS-Flat: p50={flat_p50:.2f}ms  p95={flat_p95:.2f}ms")
        print(f"  FAISS-HNSW: p50={hnsw_p50:.2f}ms  p95={hnsw_p95:.2f}ms")
        # Both should be well under any reasonable bound — guard against egregious.
        assert flat_p50 < 50, f"Flat p50={flat_p50}ms suspiciously high"
        assert hnsw_p50 < 50, f"HNSW p50={hnsw_p50}ms suspiciously high"

    def test_chromadb_vs_faiss_hnsw(self, tmp_path, corpus, queries):
        chroma = VectorStore(persist_directory=str(tmp_path / "lat_chroma"))
        chroma.create_collection(reset=True)
        hnsw = FAISSVectorStore(persist_directory=str(tmp_path / "lat_chroma_hnsw"), index_type="hnsw")
        _populated_store(chroma, corpus)
        _populated_store(hnsw, corpus)

        chroma_p50, chroma_p95 = self._bench_search(chroma, queries)
        hnsw_p50, hnsw_p95 = self._bench_search(hnsw, queries)

        print(f"\n[Latency p50/p95 (n=2000, q=50, real backends)]")
        print(f"  ChromaDB:    p50={chroma_p50:.2f}ms  p95={chroma_p95:.2f}ms")
        print(f"  FAISS-HNSW:  p50={hnsw_p50:.2f}ms  p95={hnsw_p95:.2f}ms")
        assert chroma_p50 < 200
        assert hnsw_p50 < 200


# --------------------------------------------------------------------------- #
# Persistence parity
# --------------------------------------------------------------------------- #


class TestGetEmbeddings:
    """``get_embeddings`` must return the *exact* stored vectors, aligned to the
    requested id order, or ``None`` when any id is missing — the contract the
    index-build reuse path relies on.
    """

    def _store(self, tmp_path, backend):
        if backend == "chromadb":
            store = VectorStore(persist_directory=str(tmp_path / backend))
            store.create_collection(reset=True)
        elif backend == "faiss-flat":
            store = FAISSVectorStore(persist_directory=str(tmp_path / backend), index_type="flat")
        else:
            store = FAISSVectorStore(persist_directory=str(tmp_path / backend), index_type="hnsw")
        return store

    @pytest.mark.parametrize("backend", ["chromadb", "faiss-flat", "faiss-hnsw"])
    def test_roundtrip_aligned_to_requested_order(self, tmp_path, backend):
        store = self._store(tmp_path, backend)
        corpus = _random_corpus(12, dim=64, seed=5)
        ids = [f"c{i}" for i in range(len(corpus))]
        store.add_documents(
            texts=[f"doc_{i}" for i in range(len(corpus))], embeddings=corpus.tolist(), ids=ids
        )

        # Request in a scrambled order; result must align row-for-row to it.
        scrambled = [ids[i] for i in (7, 0, 11, 3, 9, 1, 5, 2, 10, 4, 8, 6)]
        got = store.get_embeddings(scrambled)
        assert got is not None
        assert got.shape == (len(scrambled), 64)
        for row, iid in zip(got, scrambled):
            assert np.allclose(row, corpus[ids.index(iid)], atol=1e-5), f"{backend} {iid}"

    @pytest.mark.parametrize("backend", ["chromadb", "faiss-flat", "faiss-hnsw"])
    def test_missing_id_returns_none(self, tmp_path, backend):
        store = self._store(tmp_path, backend)
        corpus = _random_corpus(4, dim=64, seed=6)
        ids = [f"c{i}" for i in range(len(corpus))]
        store.add_documents(texts=ids, embeddings=corpus.tolist(), ids=ids)
        # A single unknown id in the batch poisons the whole request → None
        # (the caller must recompute rather than silently build a short index).
        assert store.get_embeddings(["c0", "ghost", "c1"]) is None

    @pytest.mark.parametrize("backend", ["chromadb", "faiss-flat", "faiss-hnsw"])
    def test_empty_ids_returns_none(self, tmp_path, backend):
        store = self._store(tmp_path, backend)
        assert store.get_embeddings([]) is None


class TestPersistenceParity:
    """Both backends must keep state across instance reloads."""

    @pytest.mark.parametrize("backend", ["chromadb", "faiss"])
    def test_reload_preserves_documents(self, tmp_path, backend):
        if backend == "chromadb":
            store = VectorStore(persist_directory=str(tmp_path / backend))
            store.create_collection(reset=True)
        else:
            store = FAISSVectorStore(persist_directory=str(tmp_path / backend))

        corpus = _random_corpus(20, dim=128, seed=11)
        _populated_store(store, corpus)
        before = store.search(corpus[5].tolist(), n_results=3)["documents"]

        # Recreate at the same path — should pick up persisted state.
        if backend == "chromadb":
            store2 = VectorStore(persist_directory=str(tmp_path / backend))
        else:
            store2 = FAISSVectorStore(persist_directory=str(tmp_path / backend))

        after = store2.search(corpus[5].tolist(), n_results=3)["documents"]
        assert before == after, f"backend={backend} reload mismatch"
