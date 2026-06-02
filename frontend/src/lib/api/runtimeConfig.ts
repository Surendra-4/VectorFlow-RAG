// src/lib/api/runtimeConfig.ts

import { apiFetch } from "./client";
import type { RuntimeConfigResponse, StagedIndexResponse } from "./types";

export function getRuntimeConfig(signal?: AbortSignal): Promise<RuntimeConfigResponse> {
  return apiFetch("/api/v1/config/runtime", { signal });
}

/** Patch live-query settings (applied to the running pipeline immediately). */
export function patchLive(
  patch: Record<string, unknown>,
  signal?: AbortSignal
): Promise<{ live: Record<string, unknown>; request_id: string }> {
  return apiFetch("/api/v1/config/runtime/live", { method: "PATCH", body: patch, signal });
}

export function getStagedIndex(signal?: AbortSignal): Promise<StagedIndexResponse> {
  return apiFetch("/api/v1/config/runtime/index", { signal });
}

/** Stage index-construction settings — never triggers a rebuild. */
export function patchStagedIndex(
  patch: Record<string, unknown>,
  signal?: AbortSignal
): Promise<StagedIndexResponse> {
  return apiFetch("/api/v1/config/runtime/index", { method: "PATCH", body: patch, signal });
}

export function resetStagedIndex(signal?: AbortSignal): Promise<StagedIndexResponse> {
  return apiFetch("/api/v1/config/runtime/index/reset", { method: "POST", signal });
}
