// src/lib/api/index.ts

export * from "./types";
export { ApiError } from "./errors";
export { apiFetch, resolveBaseUrl } from "./client";
export { parseSseStream, parseEventBlock } from "./sse";

export * as statusApi from "./status";
export * as searchApi from "./search";
export * as ingestApi from "./ingest";
export * as askApi from "./ask";
export * as cacheApi from "./cache";
export * as documentsApi from "./documents";
export * as metricsApi from "./metrics";
