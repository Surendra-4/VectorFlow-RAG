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

from src.api.dependencies import (
    get_index_manager,
    get_job_registry,
    get_pipeline,
    get_request_id,
    get_runtime_config,
)
from src.api.schemas import (
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
    check_compatibility,
    list_recipes,
    target_from_index_settings,
    validate_recipe,
)
from src.indexing.recipes import RecipeError
from src.jobs import build_index_job
from src.logging_setup import get_logger
from src.observability import get_metrics

logger = get_logger(__name__)

router = APIRouter(tags=["indexes"], prefix="/indexes")


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
    corpus = list(getattr(pipeline, "corpus", []) or [])
    if not corpus:
        raise HTTPException(status_code=400, detail="No documents ingested to build an index from.")
    if manager.registry.exists(req.name) and not req.overwrite:
        raise HTTPException(status_code=409, detail=f"Index {req.name!r} already exists.")

    # Validate a recipe choice up front (fast fail before a long job).
    from src.indexing.recipes import RECIPES

    if req.index_type in RECIPES:
        v = validate_recipe(req.index_type, {**(req.build_params or {}), **(req.search_params or {})},
                            dim=pipeline.embedder.dimension, n_vectors=len(corpus))
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
        manager=manager, profile=profile, texts=corpus,
        embedder=pipeline.embedder, make_active=req.make_active, overwrite=req.overwrite,
    )
    logger.info("Enqueued index build job %s for index %r", job.id, req.name)
    return JobAcceptedResponse(
        job_id=job.id, type=job.type, status=job.status.value, request_id=request_id
    )


# --------------------------------------------------------------------------- #
# Switch / delete / export
# --------------------------------------------------------------------------- #


@router.post("/{name}/switch", response_model=IndexActionResponse)
def switch_index(
    name: str,
    manager=Depends(get_index_manager),
    request_id: str = Depends(get_request_id),
) -> IndexActionResponse:
    try:
        manager.registry.set_active(name)
    except IndexRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        get_metrics().index_switch_total.inc()
    except Exception:  # pragma: no cover
        pass
    return IndexActionResponse(
        index_name=name, action="switched",
        active=manager.registry.active_name, request_id=request_id,
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
    runtime=Depends(get_runtime_config),
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
) -> CompatibilityResponse:
    """Check a named index against the *staged* index configuration.

    Powers the "incompatible — create a new index?" UX. Uses the running
    embedder's dimension when available for a precise dimension check."""
    try:
        profile = manager.registry.get(name)
    except IndexRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    dim = getattr(getattr(pipeline, "embedder", None), "dimension", 0) or 0
    target = target_from_index_settings(
        runtime.staged_index, vector_dimension=dim,
        corpus_fingerprint=getattr(pipeline, "corpus_fingerprint", None),
    )
    report = check_compatibility(profile, target)
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
