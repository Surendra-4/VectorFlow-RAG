# src/api/routes/indexes.py

"""
Index management API (Phase 12e–h).

* ``GET  /indexes/recipes``                 — FAISS recipe catalog (+?mode=)
* ``POST /indexes/recipes/validate``        — validate recipe+params for a dim
* ``GET  /indexes``                         — list named indexes (+active)
* ``GET  /indexes/{name}``                  — one index profile
* ``POST /indexes``                         — build a new index (background job)
* ``POST /indexes/{name}/switch``           — set the active index
* ``DELETE /indexes/{name}``                — delete an index
* ``GET  /indexes/{name}/compatibility``    — check vs the staged config
* ``POST /indexes/{name}/export``           — archive to a server path

Index *building* is a background job (see /jobs) so a large FAISS build never
blocks the HTTP worker. The compatibility endpoint powers the "create a new
index?" safety UX.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from src.api.dependencies import (
    get_index_manager,
    get_ingest_lock,
    get_job_registry,
    get_pipeline,
    get_request_id,
)
from src.api.schemas import (
    BenchmarkRequest,
    CompatibilityResponse,
    CreateIndexRequest,
    IndexActionResponse,
    IndexDetailResponse,
    IndexListResponse,
    JobAcceptedResponse,
    RecipeListResponse,
    RecipeValidateRequest,
    RecipeValidateResponse,
)
from src.indexing import (
    IndexProfile,
    IndexRegistryError,
    IndexTargetConfig,
    check_compatibility,
    list_recipes,
    validate_recipe,
)
from src.indexing.recipes import RecipeError
from src.jobs import benchmark_recipes_job, build_index_job
from src.logging_setup import get_logger
from src.observability import get_metrics

logger = get_logger(__name__)

router = APIRouter(tags=["indexes"], prefix="/indexes")


def _live_switch_target(profile, pipeline) -> IndexTargetConfig:
    """Build the compatibility target for 'can this index serve live retrieval'.

    Uses the index's own structural fields (backend / topology / chunking are
    non-issues — switching topology is the whole point) against the LIVE
    embedder identity + corpus fingerprint. So compatible ⇔ same embedding
    model + dimension AND same corpus, which is exactly when a switch is safe.
    """
    live_dim = getattr(getattr(pipeline, "embedder", None), "dimension", 0) or 0
    live_model = getattr(getattr(pipeline, "embedder", None), "model_name", "")
    return IndexTargetConfig(
        embedding_model=live_model,
        backend=profile.backend,
        index_type=profile.index_type,
        chunk_size=profile.chunk_size,
        chunk_overlap=profile.chunk_overlap,
        vector_dimension=live_dim,
        corpus_fingerprint=getattr(pipeline, "corpus_fingerprint", None),
    )


# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #


@router.get("/recipes", response_model=RecipeListResponse)
def get_recipes(
    mode: str | None = None,
    request_id: str = Depends(get_request_id),
) -> RecipeListResponse:
    try:
        specs = list_recipes(mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RecipeListResponse(recipes=[s.to_dict() for s in specs], request_id=request_id)


@router.post("/recipes/validate", response_model=RecipeValidateResponse)
def post_validate_recipe(
    req: RecipeValidateRequest,
    request_id: str = Depends(get_request_id),
) -> RecipeValidateResponse:
    try:
        result = validate_recipe(req.recipe, req.params, dim=req.dim, n_vectors=req.n_vectors)
    except RecipeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RecipeValidateResponse(validation=result.to_dict(), request_id=request_id)


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #


@router.get("", response_model=IndexListResponse)
def list_indexes(
    manager=Depends(get_index_manager),
    request_id: str = Depends(get_request_id),
) -> IndexListResponse:
    profiles = manager.registry.list_profiles()
    return IndexListResponse(
        indexes=[p.to_dict() for p in profiles],
        active=manager.registry.active_name,
        request_id=request_id,
    )


@router.get("/{name}", response_model=IndexDetailResponse)
def get_index(
    name: str,
    manager=Depends(get_index_manager),
    request_id: str = Depends(get_request_id),
) -> IndexDetailResponse:
    try:
        profile = manager.registry.get(name)
    except IndexRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return IndexDetailResponse(index=profile.to_dict(), request_id=request_id)


# --------------------------------------------------------------------------- #
# Build (background job)
# --------------------------------------------------------------------------- #


@router.post("", response_model=JobAcceptedResponse, status_code=202)
def create_index(
    req: CreateIndexRequest,
    manager=Depends(get_index_manager),
    registry=Depends(get_job_registry),
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
) -> JobAcceptedResponse:
    """Build a new named index from the currently-ingested corpus.

    Returns 202 with a job id immediately; poll /jobs/{id} or stream
    /jobs/{id}/stream for progress. The corpus is re-embedded with the
    pipeline's active embedder so the new index is self-consistent.
    """
    # Build from the pipeline's actual chunk records so the named index carries
    # identical chunk_ids + provenance — required for a correct live switch
    # (RRF joins on chunk_id; citations read metadata).
    texts, ids, metas = pipeline.iter_chunk_records()
    if not texts:
        raise HTTPException(status_code=400, detail="No documents ingested to build an index from.")
    if manager.registry.exists(req.name) and not req.overwrite:
        raise HTTPException(status_code=409, detail=f"Index {req.name!r} already exists.")

    # Validate a recipe choice up front (fast fail before a long job).
    from src.indexing.recipes import RECIPES

    if req.index_type in RECIPES:
        v = validate_recipe(req.index_type, {**(req.build_params or {}), **(req.search_params or {})},
                            dim=pipeline.embedder.dimension, n_vectors=len(texts))
        if not v.ok:
            raise HTTPException(status_code=400, detail={"recipe_errors": v.errors})

    profile = IndexProfile(
        name=req.name,
        backend=req.backend,
        index_type=req.index_type,
        embedding_model=pipeline.embedder.model_name,
        embedding_provider="sentence_transformers",
        vector_dimension=pipeline.embedder.dimension,
        build_params=req.build_params or {},
        search_params=req.search_params or {},
        chunk_size=pipeline.settings.chunker.chunk_size,
        chunk_overlap=pipeline.settings.chunker.overlap,
        corpus_fingerprint=getattr(pipeline, "corpus_fingerprint", None),
        description=req.description,
    )

    job = registry.submit(
        "index_build", build_index_job, label=f"Build index {req.name}",
        manager=manager, profile=profile, texts=texts, ids=ids, metadatas=metas,
        embedder=pipeline.embedder, make_active=req.make_active, overwrite=req.overwrite,
    )
    logger.info("Enqueued index build job %s for index %r", job.id, req.name)
    return JobAcceptedResponse(
        job_id=job.id, type=job.type, status=job.status.value, request_id=request_id
    )


# --------------------------------------------------------------------------- #
# Switch / delete / export
# --------------------------------------------------------------------------- #


@router.post("/benchmark", response_model=JobAcceptedResponse, status_code=202)
def benchmark_indexes(
    req: BenchmarkRequest,
    registry=Depends(get_job_registry),
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
) -> JobAcceptedResponse:
    """Benchmark a set of FAISS recipes over the current corpus as a background
    job. Each recipe is scored (Recall@K / MRR / latency / size) against an
    exact Flat reference. Poll /jobs/{id} for results."""
    from src.config import get_settings
    from src.indexing.recipes import RECIPES

    corpus = list(getattr(pipeline, "corpus", []) or [])
    if not corpus:
        raise HTTPException(status_code=400, detail="No documents ingested to benchmark.")

    unknown = [r for r in req.recipes if r not in RECIPES]
    if unknown:
        raise HTTPException(status_code=400, detail={"unknown_recipes": unknown})

    root = Path(get_settings().app.project_root)
    workdir = root / "var" / "benchmarks" / request_id
    persist_path = None
    if req.persist:
        ts = int(__import__("time").time())
        persist_path = root / "experiments" / "artifacts" / f"index_benchmark_{ts}.json"

    job = registry.submit(
        "index_benchmark", benchmark_recipes_job, label="Benchmark recipes",
        texts=corpus, recipe_ids=req.recipes, workdir=workdir,
        embedder=pipeline.embedder, k=req.k, params=req.params, persist_path=persist_path,
    )
    try:
        get_metrics().benchmark_runs_total.inc()
    except Exception:  # pragma: no cover
        pass
    return JobAcceptedResponse(
        job_id=job.id, type=job.type, status=job.status.value, request_id=request_id
    )


@router.post("/{name}/switch", response_model=IndexActionResponse)
def switch_index(
    name: str,
    manager=Depends(get_index_manager),
    pipeline=Depends(get_pipeline),
    lock=Depends(get_ingest_lock),
    request_id: str = Depends(get_request_id),
) -> IndexActionResponse:
    """Make a named index serve live retrieval — compatibility-gated.

    The index must be queryable by the *current* embedder (same model +
    dimension) and built over the *current* corpus (same fingerprint), else the
    vector/BM25 join breaks and provenance is wrong. On a mismatch we return 409
    with the compatibility report instead of switching — the "create a new
    index?" safety contract. Only the vector half of hybrid retrieval changes.
    """
    try:
        profile = manager.registry.get(name)
    except IndexRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    report = check_compatibility(profile, _live_switch_target(profile, pipeline))
    if not report.compatible:
        # Return a structured ErrorResponse-shaped 409 so the frontend gets the
        # full report in `details` (an HTTPException would stringify it).
        return JSONResponse(
            status_code=409,
            content={
                "code": "index_incompatible",
                "message": report.message,
                "details": {"compatibility": report.to_dict()},
                "request_id": request_id,
            },
        )

    try:
        store = manager.load_index(name)
        with lock:
            pipeline.activate_named_index(name, store)
            manager.registry.set_active(name)
    except IndexRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        get_metrics().index_switch_total.inc()
    except Exception:  # pragma: no cover
        pass
    return IndexActionResponse(
        index_name=name, action="switched",
        active=manager.registry.active_name, request_id=request_id,
    )


@router.post("/activate-default", response_model=IndexActionResponse)
def activate_default(
    pipeline=Depends(get_pipeline),
    lock=Depends(get_ingest_lock),
    request_id: str = Depends(get_request_id),
) -> IndexActionResponse:
    """Revert live retrieval to the default store built at ingestion time."""
    try:
        with lock:
            pipeline.activate_default_index()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        get_metrics().index_switch_total.inc()
    except Exception:  # pragma: no cover
        pass
    return IndexActionResponse(
        index_name="(default)", action="activated_default",
        active=None, request_id=request_id,
    )


@router.delete("/{name}", response_model=IndexActionResponse)
def delete_index(
    name: str,
    manager=Depends(get_index_manager),
    request_id: str = Depends(get_request_id),
) -> IndexActionResponse:
    try:
        manager.delete_index(name)
    except IndexRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return IndexActionResponse(
        index_name=name, action="deleted",
        active=manager.registry.active_name, request_id=request_id,
    )


@router.get("/{name}/compatibility", response_model=CompatibilityResponse)
def index_compatibility(
    name: str,
    manager=Depends(get_index_manager),
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
) -> CompatibilityResponse:
    """Can this index serve live retrieval *right now*?

    Checks against the LIVE embedder identity + corpus fingerprint — the exact
    same gate `POST /{name}/switch` enforces — so the frontend "Check" and
    "Use" buttons never contradict each other. compatible ⇔ switch will succeed.
    """
    try:
        profile = manager.registry.get(name)
    except IndexRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    report = check_compatibility(profile, _live_switch_target(profile, pipeline))
    return CompatibilityResponse(report=report.to_dict(), request_id=request_id)


@router.post("/{name}/export", response_model=IndexActionResponse)
def export_index(
    name: str,
    manager=Depends(get_index_manager),
    request_id: str = Depends(get_request_id),
) -> IndexActionResponse:
    """Archive an index under ``<project_root>/var/exports/<name>.zip``."""
    from src.config import get_settings

    dest_dir = Path(get_settings().app.project_root) / "var" / "exports"
    try:
        archive = manager.export_index(name, dest_dir)
    except IndexRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return IndexActionResponse(
        index_name=name, action=f"exported:{archive}",
        active=manager.registry.active_name, request_id=request_id,
    )
