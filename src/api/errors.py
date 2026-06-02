# src/api/errors.py

"""
Structured-error handlers wired into the FastAPI app.

Every error from anywhere in the stack ends up as an :class:`ErrorResponse`
JSON payload with a stable ``code``, ``message``, optional ``details``, and
the ``request_id``. Consumers can switch on ``code`` for branching logic.
"""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.schemas import ErrorResponse
from src.loaders.base import LoaderError
from src.logging_setup import get_logger

logger = get_logger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details=None,
) -> JSONResponse:
    payload = ErrorResponse(
        code=code,
        message=message,
        details=details,
        request_id=_request_id(request),
    )
    # by_alias=False / mode='json' so None fields render as null, not missing.
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def register_error_handlers(app) -> None:
    """Attach the three custom handlers to a FastAPI app."""

    _STATUS_CODE_TO_ERROR = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        413: "payload_too_large",
        422: "validation_error",
        429: "too_many_requests",
        500: "internal_error",
        503: "service_unavailable",
    }

    async def _http_exception_handler(request: Request, exc):
        # Handles both FastAPI HTTPException and Starlette HTTPException
        # (the latter is what's raised for unknown routes / 404s before
        # the FastAPI router gets to dispatch).
        code = _STATUS_CODE_TO_ERROR.get(exc.status_code, "http_error")
        return _error(
            request,
            status_code=exc.status_code,
            code=code,
            message=str(exc.detail),
        )

    # Register the same handler for both exception types so the structured
    # error contract is honored uniformly.
    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        return _error(
            request,
            status_code=422,
            code="validation_error",
            message="Request validation failed",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(LoaderError)
    async def _loader_handler(request: Request, exc: LoaderError):
        return _error(
            request,
            status_code=422,
            code="loader_error",
            message=str(exc),
        )

    @app.exception_handler(FileNotFoundError)
    async def _missing_file_handler(request: Request, exc: FileNotFoundError):
        return _error(
            request,
            status_code=404,
            code="file_not_found",
            message=str(exc),
        )

    @app.exception_handler(ValueError)
    async def _value_handler(request: Request, exc: ValueError):
        # RuntimeConfigError subclasses ValueError and carries a `field`.
        details = None
        field = getattr(exc, "field", None)
        if field:
            details = {"field": field}
        return _error(
            request,
            status_code=400,
            code="bad_request",
            message=str(exc),
            details=details,
        )

    # Provider-layer failures (Phase 12) → clean structured errors instead of
    # a generic 500. Auth is the user's fault (401); unavailability is
    # transient (503); unknown model is 404; anything else is an upstream
    # gateway failure (502).
    from src.providers.base import (
        ModelNotFoundError,
        ProviderAuthError,
        ProviderError,
        ProviderUnavailableError,
        UnknownProviderError,
    )

    @app.exception_handler(ProviderError)
    async def _provider_handler(request: Request, exc: ProviderError):
        if isinstance(exc, ProviderAuthError):
            status_code, code = 401, "provider_auth_error"
        elif isinstance(exc, UnknownProviderError):
            status_code, code = 404, "provider_not_found"
        elif isinstance(exc, ModelNotFoundError):
            status_code, code = 404, "model_not_found"
        elif isinstance(exc, ProviderUnavailableError) or exc.retriable:
            # Retriable base errors (e.g. Ollama list failure) are transient.
            status_code, code = 503, "provider_unavailable"
        else:
            status_code, code = 502, "provider_error"
        # Count provider failures (bounded: provider enum + exception class).
        try:
            from src.observability import get_metrics

            get_metrics().provider_errors_total.inc(
                exc.provider or "unknown", type(exc).__name__
            )
        except Exception:  # pragma: no cover - metrics must never break errors
            pass
        return _error(
            request,
            status_code=status_code,
            code=code,
            message=exc.message,
            details={"provider": exc.provider, "retriable": exc.retriable},
        )

    @app.exception_handler(Exception)
    async def _catch_all(request: Request, exc: Exception):
        logger.exception(
            "unhandled exception path=%s request_id=%s",
            request.url.path, _request_id(request),
        )
        return _error(
            request,
            status_code=500,
            code="internal_error",
            message=f"{type(exc).__name__}: {exc}",
        )
