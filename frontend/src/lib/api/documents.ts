// src/lib/api/documents.ts

import { apiFetch } from "./client";
import type { DocumentsResponse } from "./types";

export function listDocuments(signal?: AbortSignal): Promise<DocumentsResponse> {
  return apiFetch("/api/v1/documents", { signal });
}

export function resetIndex(
  confirm: boolean,
  signal?: AbortSignal
): Promise<{ cleared: boolean; request_id: string }> {
  const qs = confirm ? "?confirm=true" : "";
  return apiFetch(`/api/v1/index${qs}`, { method: "DELETE", signal });
}
