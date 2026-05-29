// src/lib/api/metrics.ts

import { apiFetch } from "./client";
import type { MetricsSnapshot, RecentTracesResponse } from "./types";

export function getMetricsSnapshot(signal?: AbortSignal): Promise<MetricsSnapshot> {
  return apiFetch("/api/v1/metrics/snapshot", { signal });
}

export function getRecentTraces(
  limit = 20,
  signal?: AbortSignal
): Promise<RecentTracesResponse> {
  return apiFetch(`/api/v1/traces/recent?limit=${limit}`, { signal });
}

export function getPrometheusText(signal?: AbortSignal): Promise<Response> {
  return apiFetch("/api/v1/metrics/prometheus", { rawResponse: true, signal });
}
