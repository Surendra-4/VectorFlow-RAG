// src/lib/api/search.ts

import { apiFetch } from "./client";
import type { SearchRequest, SearchResponse } from "./types";

export function searchDocuments(
  req: SearchRequest,
  signal?: AbortSignal
): Promise<SearchResponse> {
  return apiFetch("/api/v1/search", {
    method: "POST",
    body: req,
    signal,
  });
}
