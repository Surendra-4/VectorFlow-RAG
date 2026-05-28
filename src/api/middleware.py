# src/api/middleware.py

"""HTTP middleware: correlation IDs + per-request timing."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.logging_setup import get_logger
from src.observability import get_metrics

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"


def _route_template(request: Request) -> str:
    """
    Resolve the route template (e.g. ``/api/v1/search``) rather than the
    raw URL so per-endpoint labels stay bounded in cardinality.

    Falls back to the raw path for routes that didn't match (404s) so
    unmatched URLs are still observable, grouped under a single label
    when they share a prefix.
    """
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    # 404s: prefix-group so we don't blow up cardinality on attacker scans.
    path = request.url.path
    if not path.startswith("/api/"):
        return path
    return path


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Per-request correlation ID.

    * Echoes ``X-Request-ID`` from inbound if present (downstream callers
      that already correlate requests can keep their IDs intact).
    * Otherwise generates a UUID4.
    * Stores on ``request.state.request_id`` for the entire request.
    * Also returned in the response's ``X-Request-ID`` header.
    """

    async def dispatch(self, request: Request, call_next):
        inbound = request.headers.get(REQUEST_ID_HEADER)
        request_id = inbound if inbound else str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Records per-request wall-clock, emits structured logs, and feeds the
    observability registry (request counter + per-endpoint latency).
    """

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        request_id = getattr(request.state, "request_id", "unknown")
        metrics = get_metrics()
        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Record a 500 in metrics so unhandled exceptions still show up
            # in the dashboard — without this, a crashing endpoint would
            # be invisible to the operator.
            template = _route_template(request)
            try:
                metrics.requests_total.inc(template, "500")
                metrics.request_latency_ms.observe(template, value=elapsed_ms)
                metrics.recent_errors.append(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": 500,
                        "elapsed_ms": elapsed_ms,
                    }
                )
            except Exception:  # pragma: no cover — metric path must never break the response
                pass
            logger.error(
                "request failed method=%s path=%s elapsed_ms=%.2f request_id=%s",
                request.method, request.url.path, elapsed_ms, request_id,
            )
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.2f}"

        template = _route_template(request)
        try:
            metrics.requests_total.inc(template, str(response.status_code))
            metrics.request_latency_ms.observe(template, value=elapsed_ms)
        except Exception:  # pragma: no cover
            pass

        logger.info(
            "request method=%s path=%s status=%d elapsed_ms=%.2f request_id=%s",
            request.method, request.url.path, response.status_code, elapsed_ms, request_id,
        )
        return response
