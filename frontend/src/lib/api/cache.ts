// src/lib/api/cache.ts

import { apiFetch } from "./client";
import type { CacheStatsResponse } from "./types";

export function getCacheStats(signal?: AbortSignal): Promise<CacheStatsResponse> {
  return apiFetch("/api/v1/cache/stats", { signal });
}

export function clearCache(signal?: AbortSignal): Promise<{ cleared: boolean; request_id: string }> {
  return apiFetch("/api/v1/cache/clear", { method: "POST", signal });
}
