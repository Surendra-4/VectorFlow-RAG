# src/jobs/index_jobs.py

"""
Index-build/train background jobs (Phase 12h).

Wraps :class:`IndexManager` operations as cancellable, progress-reporting job
functions so a FAISS build/train never blocks an HTTP worker. The embedding of
the corpus is done in cancellable batches (the slow part for large corpora);
the vector-store build/train is a single reported step.

A job function has the signature ``fn(ctx, **kwargs)`` and is invoked by the
:class:`JobRegistry`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.indexing.profile import IndexProfile
from src.jobs.base import JobContext
from src.logging_setup import get_logger

logger = get_logger(__name__)


def build_index_job(
    ctx: JobContext,
    *,
    manager,
    profile: IndexProfile,
    texts: Sequence[str],
    embedder=None,
    embeddings: Optional[Sequence[Sequence[float]]] = None,
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ids: Optional[Sequence[str]] = None,
    make_active: bool = False,
    overwrite: bool = False,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Build a named index from a corpus, reporting progress + honoring cancel.

    Either ``embeddings`` (precomputed) or ``embedder`` (to compute them) must
    be provided. Embedding is batched so cancellation is responsive on large
    corpora.
    """
    ctx.check_cancel()
    n = len(texts)
    if n == 0:
        raise ValueError("Cannot build an index from an empty corpus")

    # --- Phase 1: embeddings (the expensive, cancellable part) ------------- #
    if embeddings is not None:
        emb = np.asarray(embeddings, dtype=np.float32)
        ctx.set_progress(60.0, f"Using {len(emb)} precomputed embeddings")
    elif embedder is not None:
        vectors: List[np.ndarray] = []
        done = 0
        for start in range(0, n, batch_size):
            ctx.check_cancel()
            batch = list(texts[start:start + batch_size])
            vecs = embedder.encode(batch, show_progress=False, input_type="passage")
            vectors.append(np.asarray(vecs, dtype=np.float32))
            done += len(batch)
            # Embedding is ~0..60% of the job.
            ctx.set_progress(60.0 * done / n, f"Embedded {done}/{n} chunks")
        emb = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
    else:
        raise ValueError("build_index_job requires either `embeddings` or `embedder`")

    # --- Phase 2: build + populate (training happens here for IVF/PQ) ------ #
    ctx.check_cancel()
    ctx.set_progress(70.0, "Building vector index")
    profile.vector_dimension = int(emb.shape[1]) if emb.size else profile.vector_dimension
    manager.create_index(
        profile, list(texts), emb, metadatas, ids,
        make_active=make_active, overwrite=overwrite,
    )
    ctx.set_progress(100.0, f"Index '{profile.name}' ready ({n} chunks)")

    return {
        "index_name": profile.name,
        "backend": profile.backend,
        "index_type": profile.index_type,
        "num_vectors": n,
        "vector_dimension": profile.vector_dimension,
        "made_active": make_active,
    }


def benchmark_recipes_job(
    ctx: JobContext,
    *,
    texts: Sequence[str],
    recipe_ids: Sequence[str],
    workdir,
    embedder=None,
    embeddings: Optional[Sequence[Sequence[float]]] = None,
    ids: Optional[Sequence[str]] = None,
    k: int = 10,
    params: Optional[Dict[str, Any]] = None,
    persist_path=None,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Embed the corpus (if needed) and benchmark several recipes against an
    exact Flat reference, reporting progress and honoring cancellation."""
    from src.indexing.benchmark import benchmark_recipes

    ctx.check_cancel()
    n = len(texts)
    if n == 0:
        raise ValueError("Cannot benchmark on an empty corpus")

    if embeddings is not None:
        emb = np.asarray(embeddings, dtype=np.float32)
        ctx.set_progress(30.0, f"Using {len(emb)} precomputed embeddings")
    elif embedder is not None:
        vectors: List[np.ndarray] = []
        done = 0
        for start in range(0, n, batch_size):
            ctx.check_cancel()
            batch = list(texts[start:start + batch_size])
            vecs = embedder.encode(batch, show_progress=False, input_type="passage")
            vectors.append(np.asarray(vecs, dtype=np.float32))
            done += len(batch)
            ctx.set_progress(30.0 * done / n, f"Embedded {done}/{n}")
        emb = np.vstack(vectors)
    else:
        raise ValueError("benchmark_recipes_job requires `embeddings` or `embedder`")

    def _progress(pct, msg):
        ctx.check_cancel()
        # Embedding was 0–30%; benchmarking spans 30–100%.
        ctx.set_progress(30.0 + 0.7 * pct, msg)

    skipped: List[Dict[str, str]] = []
    results = benchmark_recipes(
        emb, recipe_ids, workdir=workdir, ids=ids, k=k, params=params,
        persist_path=persist_path, skipped=skipped, progress=_progress,
    )
    if not results:
        # Every recipe failed to build (e.g. all need more training vectors than
        # the corpus has). Fail the job with a clear, actionable reason rather
        # than "succeeding" with an empty table.
        reasons = "; ".join(f"{s['recipe']}: {s['reason']}" for s in skipped) or "unknown error"
        raise ValueError(f"No recipe could be benchmarked on this corpus — {reasons}")
    return {
        "k": k,
        "num_vectors": n,
        "dimension": int(emb.shape[1]),
        "results": [r.to_dict() for r in results],
        "skipped": skipped,
        "artifact": str(persist_path) if persist_path else None,
    }
