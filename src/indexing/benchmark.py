# src/indexing/benchmark.py

"""
Index benchmarking (Phase 12i).

Measures and compares vector-index recipes on a corpus *without needing labeled
relevance judgments*: an exact **Flat** index over the same vectors provides the
ground truth, and each approximate recipe is scored by how well it reproduces
those exact results.

Metrics per index:

* **Recall@K** — fraction of the exact top-K neighbors the index returns.
* **NDCG@K** — rank quality vs. the exact ordering (graded; 1.0 = identical).
* **MRR** — mean reciprocal rank of the true nearest neighbor.
* **search latency** — mean / p50 / p95 (ms) and queries/sec.
* **build/ingest speed** — chunks/sec during construction.
* **index size** — on-disk bytes (when persisted) + estimated memory.

Results are JSON-serializable and persisted (schema-versioned) under
``experiments/artifacts/`` so runs are reproducible and comparable over time.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.indexing.recipes import build_factory_string, estimate, resolve_params
from src.logging_setup import get_logger

logger = get_logger(__name__)

BENCHMARK_SCHEMA_VERSION = 2  # v2 adds ndcg_at_k per recipe


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def exact_neighbors(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Return the exact top-``k`` neighbor indices per query (inner product).

    Vectors are assumed L2-normalized upstream (the embedder normalizes), so
    inner product ranks identically to cosine similarity.
    """
    sims = queries @ corpus.T            # (q, n)
    k = min(k, corpus.shape[0])
    # argpartition for the top-k, then sort those k by score desc.
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(queries.shape[0])[:, None]
    ordered = part[rows, np.argsort(-sims[rows, part], axis=1)]
    return ordered  # (q, k) indices into corpus


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkResult:
    index_name: str
    recipe: str
    factory_string: str
    k: int
    num_vectors: int
    dimension: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    queries_per_sec: float
    build_seconds: float
    ingest_vectors_per_sec: float
    index_size_bytes: int
    estimated_memory_bytes: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Evaluate a store
# --------------------------------------------------------------------------- #


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), pct))


def _ndcg_at_k(got: Sequence[str], truth: Sequence[str]) -> float:
    """Graded NDCG@k using exact search as the ideal ranking.

    Relevance of a doc = its gain in the exact top-k (nearest neighbour gets
    ``len(truth)``, …, the k-th gets ``1``). DCG over the approximate result
    order is normalized by the ideal DCG (the exact order), so an index that
    reproduces exact ranking scores 1.0 and one that mis-orders or misses the
    true neighbours scores lower. This rewards *rank quality*, not just recall.
    """
    if not truth:
        return 0.0
    rel = {tid: (len(truth) - r) for r, tid in enumerate(truth)}
    dcg = sum(rel.get(g, 0) / np.log2(j + 2) for j, g in enumerate(got))
    idcg = sum((len(truth) - r) / np.log2(r + 2) for r in range(len(truth)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_store(
    store,
    queries: np.ndarray,
    truth_ids: Sequence[Sequence[str]],
    k: int,
) -> Dict[str, float]:
    """Score a store's search results against per-query ground-truth ids.

    ``truth_ids[i]`` is the exact top-``k`` chunk-id list for query ``i``.
    """
    recalls: List[float] = []
    rr: List[float] = []
    ndcgs: List[float] = []
    latencies: List[float] = []

    for i in range(queries.shape[0]):
        t0 = time.perf_counter()
        res = store.search(queries[i], n_results=k)
        latencies.append((time.perf_counter() - t0) * 1000)

        got = list(res.get("ids", []))
        truth = list(truth_ids[i])
        truth_set = set(truth)
        if truth_set:
            hit = sum(1 for g in got if g in truth_set)
            recalls.append(hit / min(k, len(truth)))
        # Reciprocal rank of the true nearest neighbor (truth[0]).
        if truth:
            target = truth[0]
            rank = next((j + 1 for j, g in enumerate(got) if g == target), None)
            rr.append(1.0 / rank if rank else 0.0)
            ndcgs.append(_ndcg_at_k(got, truth))

    total_s = sum(latencies) / 1000.0
    qps = (queries.shape[0] / total_s) if total_s > 0 else 0.0
    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(rr)) if rr else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "latency_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "queries_per_sec": qps,
    }


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- #
# Recipe comparison (builds each recipe in a temp dir)
# --------------------------------------------------------------------------- #


def benchmark_recipes(
    corpus: np.ndarray,
    recipe_ids: Sequence[str],
    *,
    workdir: Path,
    queries: Optional[np.ndarray] = None,
    ids: Optional[Sequence[str]] = None,
    k: int = 10,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    persist_path: Optional[Path] = None,
    skipped: Optional[List[Dict[str, str]]] = None,
    progress=None,
) -> List[BenchmarkResult]:
    """Build each recipe over ``corpus`` and benchmark it against exact search.

    ``progress(pct, msg)`` is an optional callback (e.g. a JobContext.set_progress)
    so this can run as a background job. Returns one BenchmarkResult per recipe
    that built successfully; optionally persists the full run to ``persist_path``.

    A recipe that can't be built on this corpus (e.g. an IVF/PQ recipe whose
    training needs more vectors than are ingested) is **skipped, not fatal** —
    the others still run. If ``skipped`` (a list) is passed, one
    ``{"recipe", "reason"}`` dict is appended to it per skipped recipe.
    """
    from src.faiss_store import FAISSVectorStore

    corpus = np.asarray(corpus, dtype=np.float32)
    n, dim = corpus.shape
    if queries is None:
        # Use a sample of the corpus itself as queries (self-retrieval).
        sample = min(50, n)
        queries = corpus[:sample]
    if ids is None:
        ids = [f"v{i}" for i in range(n)]
    ids = list(ids)

    # Ground truth from exact search.
    gt_idx = exact_neighbors(corpus, queries, k)
    truth_ids = [[ids[j] for j in row] for row in gt_idx]

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    texts = [f"doc {i}" for i in range(n)]

    results: List[BenchmarkResult] = []
    total = len(recipe_ids)
    for r_i, recipe_id in enumerate(recipe_ids):
        if progress:
            progress(100.0 * r_i / total, f"Benchmarking {recipe_id}")
        rparams = resolve_params(recipe_id, (params or {}).get(recipe_id))
        factory = build_factory_string(recipe_id, rparams, dim)
        idx_dir = workdir / f"bench_{recipe_id}"

        try:
            store = FAISSVectorStore(
                persist_directory=str(idx_dir),
                collection_name="bench",
                index_type=recipe_id,
                factory_string=factory,
                nprobe=rparams.get("nprobe"),
                hnsw_ef_search=rparams.get("efSearch", 64),
            )

            t_build = time.perf_counter()
            store.add_documents(texts=texts, embeddings=corpus.tolist(), ids=ids)
            build_s = time.perf_counter() - t_build

            scores = evaluate_store(store, queries, truth_ids, k)
        except Exception as exc:
            # A recipe that can't be built on this corpus (most often: too few
            # vectors to train an IVF/PQ index) shouldn't sink the whole sweep.
            # Record why and move on so the buildable recipes still get scored.
            reason = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            logger.warning("Skipping recipe %r in benchmark: %s", recipe_id, reason)
            if skipped is not None:
                skipped.append({"recipe": recipe_id, "reason": reason})
            continue

        results.append(BenchmarkResult(
            index_name=recipe_id,
            recipe=recipe_id,
            factory_string=factory,
            k=k,
            num_vectors=n,
            dimension=dim,
            recall_at_k=scores["recall_at_k"],
            mrr=scores["mrr"],
            ndcg_at_k=scores["ndcg_at_k"],
            latency_ms_mean=scores["latency_ms_mean"],
            latency_ms_p50=scores["latency_ms_p50"],
            latency_ms_p95=scores["latency_ms_p95"],
            queries_per_sec=scores["queries_per_sec"],
            build_seconds=build_s,
            ingest_vectors_per_sec=(n / build_s) if build_s > 0 else 0.0,
            index_size_bytes=_dir_size_bytes(idx_dir),
            estimated_memory_bytes=estimate(recipe_id, rparams, dim, n).memory_bytes,
        ))

    if progress:
        progress(100.0, "Benchmark complete")

    if persist_path is not None:
        persist_benchmark(results, persist_path, dim=dim, n_vectors=n, k=k)

    return results


def persist_benchmark(
    results: List[BenchmarkResult],
    path: Path,
    *,
    dim: int,
    n_vectors: int,
    k: int,
) -> Path:
    """Write a schema-versioned benchmark artifact to ``path`` (JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": time.time(),
        "dim": dim,
        "n_vectors": n_vectors,
        "k": k,
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Persisted benchmark with %d recipes to %s", len(results), path)
    return path
