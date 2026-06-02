// src/lib/api/models.ts

import { apiFetch } from "./client";
import { consumeSse } from "./sseProgress";
import type {
  ActiveModelResponse,
  ApiKeyStatusResponse,
  ConnectionValidationResponse,
  ModelListResponse,
  ProvidersResponse,
} from "./types";

export function listProviders(signal?: AbortSignal): Promise<ProvidersResponse> {
  return apiFetch("/api/v1/models/providers", { signal });
}

export function listInstalled(
  provider = "ollama",
  signal?: AbortSignal
): Promise<ModelListResponse> {
  return apiFetch(`/api/v1/models/offline/installed?provider=${encodeURIComponent(provider)}`, {
    signal,
  });
}

export function listCatalog(
  provider = "ollama",
  signal?: AbortSignal
): Promise<ModelListResponse> {
  return apiFetch(`/api/v1/models/offline/catalog?provider=${encodeURIComponent(provider)}`, {
    signal,
  });
}

/** Pull a model, streaming progress events (`progress` / `done` / `error`). */
export function installModel(
  name: string,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  opts: { provider?: string; signal?: AbortSignal } = {}
): Promise<void> {
  return consumeSse("/api/v1/models/offline/install", onEvent, {
    method: "POST",
    body: { name, provider: opts.provider ?? "ollama" },
    signal: opts.signal,
  });
}

export function deleteModel(
  name: string,
  provider = "ollama",
  signal?: AbortSignal
): Promise<{ deleted: boolean; name: string; request_id: string }> {
  return apiFetch(
    `/api/v1/models/offline/${encodeURIComponent(name)}?provider=${encodeURIComponent(provider)}`,
    { method: "DELETE", signal }
  );
}

export function listOnlineModels(
  provider: string,
  signal?: AbortSignal
): Promise<ModelListResponse> {
  return apiFetch(`/api/v1/models/online/${encodeURIComponent(provider)}`, { signal });
}

export function setApiKey(
  provider: string,
  apiKey: string,
  signal?: AbortSignal
): Promise<ApiKeyStatusResponse> {
  return apiFetch(`/api/v1/models/online/${encodeURIComponent(provider)}/key`, {
    method: "PUT",
    body: { api_key: apiKey },
    signal,
  });
}

export function deleteApiKey(
  provider: string,
  signal?: AbortSignal
): Promise<ApiKeyStatusResponse> {
  return apiFetch(`/api/v1/models/online/${encodeURIComponent(provider)}/key`, {
    method: "DELETE",
    signal,
  });
}

export function getApiKeyStatus(
  provider: string,
  signal?: AbortSignal
): Promise<ApiKeyStatusResponse> {
  return apiFetch(`/api/v1/models/online/${encodeURIComponent(provider)}/key`, { signal });
}

export function validateConnection(
  provider: string,
  signal?: AbortSignal
): Promise<ConnectionValidationResponse> {
  return apiFetch(`/api/v1/models/online/${encodeURIComponent(provider)}/validate`, {
    method: "POST",
    signal,
  });
}

export function getActiveModel(signal?: AbortSignal): Promise<ActiveModelResponse> {
  return apiFetch("/api/v1/models/active", { signal });
}

export interface SelectModelBody {
  provider: string;
  model: string;
  base_url?: string | null;
  max_tokens?: number;
  temperature?: number;
  request_timeout_s?: number;
}

export function selectModel(
  body: SelectModelBody,
  signal?: AbortSignal
): Promise<ActiveModelResponse> {
  return apiFetch("/api/v1/models/select", { method: "POST", body, signal });
}
