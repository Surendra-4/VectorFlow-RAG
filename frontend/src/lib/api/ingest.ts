// src/lib/api/ingest.ts

import { apiFetch } from "./client";
import type {
  IngestionResponse,
  IngestPathsRequest,
  IngestTextRequest,
} from "./types";

export function ingestText(
  body: IngestTextRequest,
  signal?: AbortSignal
): Promise<IngestionResponse> {
  return apiFetch("/api/v1/ingest/text", { method: "POST", body, signal });
}

export function ingestPaths(
  body: IngestPathsRequest,
  signal?: AbortSignal
): Promise<IngestionResponse> {
  return apiFetch("/api/v1/ingest/paths", { method: "POST", body, signal });
}

export function ingestFiles(
  files: File[],
  reset: boolean,
  signal?: AbortSignal
): Promise<IngestionResponse> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  fd.append("reset", String(reset));
  return apiFetch("/api/v1/ingest/files", {
    method: "POST",
    body: fd,
    formData: true,
    signal,
  });
}
