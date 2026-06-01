# src/api/routes/models.py

"""
Model management API (Phase 12d).

Endpoints (all under ``/api/v1``):

* ``GET    /models/providers``                  — provider capabilities + key status
* ``GET    /models/offline/installed``          — installed Ollama models
* ``GET    /models/offline/catalog``            — curated downloadable models
* ``POST   /models/offline/install``            — pull a model (SSE progress)
* ``DELETE /models/offline/{name}``             — delete an installed model
* ``GET    /models/online/{provider}``          — list an online provider's models
* ``PUT    /models/online/{provider}/key``      — store an API key (backend only)
* ``DELETE /models/online/{provider}/key``      — delete a stored API key
* ``GET    /models/online/{provider}/key``      — key status (configured? hint)
* ``POST   /models/online/{provider}/validate`` — validate the connection
* ``GET    /models/active``                     — the active chat model
* ``POST   /models/select``                     — switch the active chat model

Security: API keys are accepted in request bodies, stored only in the backend
``SecretStore``, and NEVER returned. The frontend only ever sees
``{configured, hint}``.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, Path
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    get_ingest_lock,
    get_pipeline,
    get_request_id,
    get_runtime_config,
)
from src.api.schemas import (
    ActiveModelResponse,
    ApiKeyStatusResponse,
    ConnectionValidationResponse,
    DeleteModelResponse,
    InstallRequest,
    ModelListResponse,
    ProviderCapabilitiesSchema,
    ProviderModelSchema,
    ProvidersResponse,
    SelectModelRequest,
    SetApiKeyRequest,
)
from src.logging_setup import get_logger
from src.observability import get_metrics
from src.providers import (
    ChatModelConfig,
    ProviderError,
    get_provider_capabilities,
    get_secret_store,
    list_provider_capabilities,
    make_chat_provider,
)
from src.providers.registry import get_provider_class

logger = get_logger(__name__)

router = APIRouter(tags=["models"], prefix="/models")


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #


def _model_to_schema(m) -> ProviderModelSchema:
    return ProviderModelSchema(**m.to_dict())


def _provider_for(name: str, *, model: str = "placeholder"):
    """Build a provider instance (key injected) just for listing/validation."""
    cfg = ChatModelConfig(provider=name, model=model)
    return make_chat_provider(cfg, secret_store=get_secret_store())


def _record_provider_metric(provider: str, op: str) -> None:
    try:
        get_metrics().provider_ops_total.inc(provider, op)
    except Exception:  # pragma: no cover - metrics must never break a response
        pass


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


@router.get("/providers", response_model=ProvidersResponse)
def list_providers(request_id: str = Depends(get_request_id)) -> ProvidersResponse:
    store = get_secret_store()
    out = []
    for caps in list_provider_capabilities():
        desc = store.describe().get(caps.name, {})
        out.append(ProviderCapabilitiesSchema(
            **caps.to_dict(),
            key_configured=bool(desc.get("configured", False)),
            key_hint=desc.get("hint"),
        ))
    return ProvidersResponse(providers=out, request_id=request_id)


# --------------------------------------------------------------------------- #
# Offline (Ollama) models
# --------------------------------------------------------------------------- #


@router.get("/offline/installed", response_model=ModelListResponse)
def offline_installed(
    provider: str = "ollama",
    request_id: str = Depends(get_request_id),
) -> ModelListResponse:
    prov = _provider_for(provider)
    models = prov.list_models()  # raises ProviderError → mapped by handler
    _record_provider_metric(provider, "list_installed")
    return ModelListResponse(
        provider=provider,
        models=[_model_to_schema(m) for m in models],
        request_id=request_id,
    )


@router.get("/offline/catalog", response_model=ModelListResponse)
def offline_catalog(
    provider: str = "ollama",
    request_id: str = Depends(get_request_id),
) -> ModelListResponse:
    prov = _provider_for(provider)
    models = prov.list_catalog()
    return ModelListResponse(
        provider=provider,
        models=[_model_to_schema(m) for m in models],
        request_id=request_id,
    )


@router.post("/offline/install")
def offline_install(
    req: InstallRequest,
    request_id: str = Depends(get_request_id),
):
    """Pull a model, streaming progress as SSE.

    Event types:
      * ``progress`` — ``{status, total, completed, percent, digest}``
      * ``done``     — ``{name, request_id}``
      * ``error``    — ``{message, request_id}``
    """
    prov = _provider_for(req.provider)

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

    def generate():
        try:
            m = get_metrics()
        except Exception:  # pragma: no cover
            m = None
        try:
            for progress in prov.install_model(req.name):
                yield _sse("progress", {**progress, "request_id": request_id})
            if m is not None:
                try:
                    m.model_installs_total.inc(req.provider, "success")
                except Exception:  # pragma: no cover
                    pass
            yield _sse("done", {"name": req.name, "request_id": request_id})
        except ProviderError as exc:
            if m is not None:
                try:
                    m.model_installs_total.inc(req.provider, "failure")
                except Exception:  # pragma: no cover
                    pass
            logger.warning("install failed for %s: %s", req.name, exc)
            yield _sse("error", {"message": exc.message, "request_id": request_id})
        except Exception as exc:  # pragma: no cover - defensive
            yield _sse("error", {"message": str(exc), "request_id": request_id})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
    )


@router.delete("/offline/{name:path}", response_model=DeleteModelResponse)
def offline_delete(
    name: str = Path(..., description="Model id to delete."),
    provider: str = "ollama",
    request_id: str = Depends(get_request_id),
) -> DeleteModelResponse:
    prov = _provider_for(provider)
    prov.delete_model(name)
    _record_provider_metric(provider, "delete")
    return DeleteModelResponse(
        provider=provider, name=name, deleted=True, request_id=request_id
    )


# --------------------------------------------------------------------------- #
# Online providers
# --------------------------------------------------------------------------- #


@router.get("/online/{provider}", response_model=ModelListResponse)
def online_models(
    provider: str = Path(..., description="Online provider name."),
    request_id: str = Depends(get_request_id),
) -> ModelListResponse:
    get_provider_capabilities(provider)  # 404-maps unknown provider via handler
    prov = _provider_for(provider)
    models = prov.list_models()
    _record_provider_metric(provider, "list_online")
    return ModelListResponse(
        provider=provider,
        models=[_model_to_schema(m) for m in models],
        request_id=request_id,
    )


@router.put("/online/{provider}/key", response_model=ApiKeyStatusResponse)
def set_api_key(
    req: SetApiKeyRequest,
    provider: str = Path(...),
    request_id: str = Depends(get_request_id),
) -> ApiKeyStatusResponse:
    caps = get_provider_capabilities(provider)
    if not caps.requires_api_key:
        # Storing a key for an offline provider is meaningless; reject clearly.
        raise ProviderError(
            f"{caps.label} does not use an API key.", provider=provider
        )
    store = get_secret_store()
    store.set_secret(provider, req.api_key)
    _record_provider_metric(provider, "set_key")
    desc = store.describe().get(provider, {})
    return ApiKeyStatusResponse(
        provider=provider, configured=True, hint=desc.get("hint"), request_id=request_id
    )


@router.delete("/online/{provider}/key", response_model=ApiKeyStatusResponse)
def delete_api_key(
    provider: str = Path(...),
    request_id: str = Depends(get_request_id),
) -> ApiKeyStatusResponse:
    store = get_secret_store()
    store.delete_secret(provider)
    return ApiKeyStatusResponse(
        provider=provider, configured=False, hint=None, request_id=request_id
    )


@router.get("/online/{provider}/key", response_model=ApiKeyStatusResponse)
def get_api_key_status(
    provider: str = Path(...),
    request_id: str = Depends(get_request_id),
) -> ApiKeyStatusResponse:
    store = get_secret_store()
    desc = store.describe().get(provider, {})
    return ApiKeyStatusResponse(
        provider=provider,
        configured=bool(desc.get("configured", False)),
        hint=desc.get("hint"),
        request_id=request_id,
    )


@router.post("/online/{provider}/validate", response_model=ConnectionValidationResponse)
def validate_connection(
    provider: str = Path(...),
    request_id: str = Depends(get_request_id),
) -> ConnectionValidationResponse:
    get_provider_capabilities(provider)
    prov = _provider_for(provider)
    status = prov.validate_connection()  # never raises
    _record_provider_metric(provider, "validate")
    return ConnectionValidationResponse(
        provider=provider,
        ok=status.ok,
        message=status.message,
        models_available=status.models_available,
        latency_ms=status.latency_ms,
        request_id=request_id,
    )


# --------------------------------------------------------------------------- #
# Active model selection
# --------------------------------------------------------------------------- #


@router.get("/active", response_model=ActiveModelResponse)
def active_model(
    pipeline=Depends(get_pipeline),
    runtime=Depends(get_runtime_config),
    request_id: str = Depends(get_request_id),
) -> ActiveModelResponse:
    live = runtime.live
    location = None
    try:
        location = get_provider_capabilities(live.chat.provider).location.value
    except ProviderError:
        pass
    return ActiveModelResponse(
        provider=live.chat.provider,
        model=getattr(pipeline.llm, "model", live.chat.model),
        base_url=live.chat.base_url,
        location=location,
        request_id=request_id,
    )


@router.post("/select", response_model=ActiveModelResponse)
def select_model(
    req: SelectModelRequest,
    pipeline=Depends(get_pipeline),
    runtime=Depends(get_runtime_config),
    lock=Depends(get_ingest_lock),
    request_id: str = Depends(get_request_id),
) -> ActiveModelResponse:
    """Switch the active chat model at runtime.

    Persists the choice to the runtime config and hot-swaps the live provider
    on the pipeline — no restart. Serialized by the ingest lock so a model
    switch can't interleave with an in-flight ingestion's pipeline mutation.
    """
    # Start from current live chat config; overlay only provided fields.
    current = runtime.live.chat
    patch = {
        "provider": req.provider,
        "model": req.model,
        "base_url": req.base_url if req.base_url is not None else current.base_url,
        "max_tokens": req.max_tokens if req.max_tokens is not None else current.max_tokens,
        "temperature": req.temperature if req.temperature is not None else current.temperature,
        "request_timeout_s": (
            req.request_timeout_s if req.request_timeout_s is not None
            else current.request_timeout_s
        ),
    }
    # update_live validates (provider registered, ranges) and persists.
    new_live = runtime.update_live({"chat": patch})

    with lock:
        pipeline.set_chat_provider(new_live.chat, secret_store=get_secret_store())

    try:
        get_metrics().model_switch_total.inc(req.provider, "chat")
    except Exception:  # pragma: no cover
        pass

    location = None
    try:
        location = get_provider_capabilities(req.provider).location.value
    except ProviderError:
        pass

    logger.info("Active chat model selected: %s/%s", req.provider, req.model)
    return ActiveModelResponse(
        provider=req.provider, model=req.model, base_url=new_live.chat.base_url,
        location=location, request_id=request_id,
    )
