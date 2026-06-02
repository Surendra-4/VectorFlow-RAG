// src/lib/api/jobs.ts

import { apiFetch } from "./client";
import { consumeSse } from "./sseProgress";
import type { JobState } from "./types";

export function listJobs(
  opts: { type?: string; limit?: number } = {},
  signal?: AbortSignal
): Promise<{ jobs: JobState[]; request_id: string }> {
  const params = new URLSearchParams();
  if (opts.type) params.set("type", opts.type);
  if (opts.limit) params.set("limit", String(opts.limit));
  const q = params.toString();
  return apiFetch(`/api/v1/jobs${q ? `?${q}` : ""}`, { signal });
}

export function getJob(
  jobId: string,
  signal?: AbortSignal
): Promise<{ job: JobState; request_id: string }> {
  return apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`, { signal });
}

/** Stream a job's progress (`progress` / `done` events) until it ends. */
export function streamJob(
  jobId: string,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  signal?: AbortSignal
): Promise<void> {
  return consumeSse(`/api/v1/jobs/${encodeURIComponent(jobId)}/stream`, onEvent, { signal });
}

export function cancelJob(
  jobId: string,
  signal?: AbortSignal
): Promise<{ job: JobState; request_id: string }> {
  return apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
    signal,
  });
}
