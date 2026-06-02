# src/api/routes/ask.py

"""
POST /api/v1/ask — retrieval + LLM answer.

Two response modes:

1. ``stream=false`` (default): standard JSON :class:`AskResponse`.
2. ``stream=true``: ``text/event-stream`` SSE with event types:
   - ``sources`` — JSON-encoded retrieval results (sent once at start)
   - ``token``   — one event per LLM token
   - ``done``    — final event with answer + metrics

The streaming variant requires LLM availability (Ollama). When the LLM
fails the stream still emits ``done`` with an error in metrics.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_pipeline, get_request_id
from src.api.routes.search import _to_result
from src.api.schemas import AskMetrics, AskRequest, AskResponse

router = APIRouter(tags=["retrieval"])


@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    pipeline=Depends(get_pipeline),
    request_id: str = Depends(get_request_id),
):
    if req.stream:
        return _stream_ask_response(req, pipeline, request_id)

    if pipeline.hybrid_retriever is None:
        # No documents to ground on — answer says so, sources empty.
        return AskResponse(
            question=req.query,
            answer="No documents have been ingested yet.",
            sources=[] if req.return_sources else None,
            metrics=AskMetrics(
                retrieval_time_ms=0.0, generation_time_ms=0.0,
                total_time_ms=0.0, num_context_docs=0,
            ),
            request_id=request_id,
        )

    response = pipeline.ask(
        question=req.query,
        k_docs=req.k_docs,
        return_sources=req.return_sources,
        verbose=False,
    )
    _record_provider_chat(pipeline, "success")
    sources_payload = None
    if req.return_sources and "sources" in response:
        sources_payload = [_to_result(s) for s in response["sources"]]
    return AskResponse(
        question=response["question"],
        answer=response["answer"],
        sources=sources_payload,
        metrics=AskMetrics(**response["metrics"]),
        request_id=request_id,
    )


# --------------------------------------------------------------------------- #
# Streaming (SSE) implementation
# --------------------------------------------------------------------------- #


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    """Format a Server-Sent Event chunk."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _record_provider_chat(pipeline, status: str) -> None:
    """Count a chat generation by the active provider + outcome (bounded)."""
    try:
        from src.observability import get_metrics

        provider = getattr(pipeline.llm, "name", "ollama")
        get_metrics().provider_chat_total.inc(provider, status)
    except Exception:  # pragma: no cover - metrics must never break a response
        pass


def _stream_ask_response(req: AskRequest, pipeline, request_id: str) -> StreamingResponse:
    # Lazy import to keep observability optional at the route level —
    # this whole route still works if the registry isn't initialized.
    from src.observability import get_metrics

    def generate():
        t_total = time.perf_counter()
        # Mark this session as open in the gauge so concurrent SSE
        # connections are visible to the dashboard. We always decrement
        # in the `finally` below — even if the client disconnects.
        try:
            m = get_metrics()
            m.active_streams.inc()
            m.stream_sessions_total.inc()
        except Exception:  # pragma: no cover
            m = None

        try:
            if pipeline.hybrid_retriever is None:
                yield _sse_event("error", {"message": "No documents ingested", "request_id": request_id})
                yield _sse_event("done", {"request_id": request_id})
                return

            # Retrieval phase
            t_ret = time.perf_counter()
            retrieval_results = pipeline.search(req.query, k=req.k_docs, return_trace=False)
            retrieval_ms = (time.perf_counter() - t_ret) * 1000

            if req.return_sources:
                yield _sse_event(
                    "sources",
                    {
                        "request_id": request_id,
                        "sources": [_to_result(r).model_dump(mode="json") for r in retrieval_results],
                    },
                )

            # Generation phase — stream tokens via the LLM client.
            context_texts = [r["text"] for r in retrieval_results]
            full_text = ""
            t_gen = time.perf_counter()
            try:
                for token in pipeline.llm.stream_generate(
                    prompt=req.query,
                    context=context_texts,
                ):
                    if token:
                        full_text += token
                        yield _sse_event("token", {"token": token})
            except Exception as exc:
                yield _sse_event("error", {"message": str(exc), "request_id": request_id})
            gen_ms = (time.perf_counter() - t_gen) * 1000

            total_ms = (time.perf_counter() - t_total) * 1000
            yield _sse_event(
                "done",
                {
                    "request_id": request_id,
                    "answer": full_text,
                    "metrics": {
                        "retrieval_time_ms": round(retrieval_ms, 2),
                        "generation_time_ms": round(gen_ms, 2),
                        "total_time_ms": round(total_ms, 2),
                        "num_context_docs": len(retrieval_results),
                    },
                },
            )
        finally:
            if m is not None:
                try:
                    m.active_streams.dec()
                    m.stream_duration_ms.observe((time.perf_counter() - t_total) * 1000)
                except Exception:  # pragma: no cover
                    pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Request-ID": request_id,
        },
    )
