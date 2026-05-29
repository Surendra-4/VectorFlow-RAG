// src/lib/api/ask.ts

import { apiFetch, resolveBaseUrl } from "./client";
import { ApiError } from "./errors";
import { parseSseStream } from "./sse";
import type { AskRequest, AskResponse, AskSseEvent } from "./types";

/** Blocking ask — returns the full AskResponse. */
export function ask(req: AskRequest, signal?: AbortSignal): Promise<AskResponse> {
  return apiFetch("/api/v1/ask", {
    method: "POST",
    body: { ...req, stream: false },
    signal,
  });
}

/**
 * Streaming ask — yields one event at a time.
 *
 * Connection lifecycle:
 *   sources → token* → done   (success)
 *   sources? → error → done?  (failure)
 *
 * The caller decides what to do with each event (typically: render sources
 * up front, append tokens as they arrive, finalize on done).
 */
export async function* streamAsk(
  req: AskRequest,
  signal?: AbortSignal
): AsyncGenerator<AskSseEvent> {
  const url = `${resolveBaseUrl()}/api/v1/ask`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ ...req, stream: true }),
    signal,
  });

  if (!res.ok || !res.body) {
    const requestId = res.headers.get("X-Request-ID");
    throw new ApiError({
      status: res.status,
      code: `http_${res.status}`,
      message: `ask stream failed: ${res.status} ${res.statusText}`,
      requestId,
    });
  }

  for await (const evt of parseSseStream(res.body)) {
    if (!evt.event && !evt.data) continue;
    try {
      const parsed = JSON.parse(evt.data) as Record<string, unknown>;
      // Coerce into the discriminated union. The event name from the SSE
      // header is authoritative; the payload provides the rest.
      switch (evt.event) {
        case "sources":
          yield { type: "sources", ...(parsed as object) } as AskSseEvent;
          break;
        case "token":
          yield { type: "token", token: (parsed.token as string) ?? "" };
          break;
        case "done":
          yield { type: "done", ...(parsed as object) } as AskSseEvent;
          break;
        case "error":
          yield { type: "error", ...(parsed as object) } as AskSseEvent;
          break;
        default:
          // Unknown event type — skip silently.
          break;
      }
    } catch {
      // Malformed data line — skip rather than tear down the stream.
      continue;
    }
  }
}
