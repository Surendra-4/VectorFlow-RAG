// src/lib/api/status.ts

import { apiFetch } from "./client";
import type { StatusResponse } from "./types";

export function getHealth(signal?: AbortSignal): Promise<{ status: string; uptime_s: number; version: string }> {
  return apiFetch("/health", { signal });
}

export function getStatus(signal?: AbortSignal): Promise<StatusResponse> {
  return apiFetch("/api/v1/status", { signal });
}
