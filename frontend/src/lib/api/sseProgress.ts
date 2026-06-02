// src/lib/api/sseProgress.ts

import { apiFetch, resolveBaseUrl } from "./client";
import { parseSseStream } from "./sse";

/**
 * Drive an SSE endpoint and invoke `onEvent(eventName, data)` for each parsed
 * event until the stream closes. Shared by model-install progress and job
 * progress (both emit `progress` / `done` / `error` events with JSON data).
 *
 * Returns when the stream ends. Aborting via `signal` stops consumption.
 */
export async function consumeSse(
  path: string,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  opts: { method?: string; body?: unknown; signal?: AbortSignal } = {}
): Promise<void> {
  const res = (await apiFetch(path, {
    method: opts.method ?? "GET",
    body: opts.body,
    rawResponse: true,
    signal: opts.signal,
  })) as unknown as Response;

  if (!res.body) return;

  for await (const evt of parseSseStream(res.body)) {
    let data: Record<string, unknown> = {};
    if (evt.data) {
      try {
        data = JSON.parse(evt.data) as Record<string, unknown>;
      } catch {
        data = { raw: evt.data };
      }
    }
    onEvent(evt.event || "message", data);
  }
}

/** Absolute URL for an SSE endpoint (handy for EventSource-free fetch use). */
export function sseUrl(path: string): string {
  return `${resolveBaseUrl()}${path.startsWith("/") ? "" : "/"}${path}`;
}
