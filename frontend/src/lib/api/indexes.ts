// src/lib/api/indexes.ts

import { apiFetch } from "./client";
import type {
  CompatibilityReport,
  IndexListResponse,
  IndexProfile,
  JobAcceptedResponse,
  RecipeSpec,
  RecipeValidation,
} from "./types";

export function listRecipes(
  mode?: "basic" | "advanced",
  signal?: AbortSignal
): Promise<{ recipes: RecipeSpec[]; request_id: string }> {
  const q = mode ? `?mode=${mode}` : "";
  return apiFetch(`/api/v1/indexes/recipes${q}`, { signal });
}

export function validateRecipe(
  body: { recipe: string; params?: Record<string, number>; dim: number; n_vectors?: number },
  signal?: AbortSignal
): Promise<{ validation: RecipeValidation; request_id: string }> {
  return apiFetch("/api/v1/indexes/recipes/validate", { method: "POST", body, signal });
}

export function listIndexes(signal?: AbortSignal): Promise<IndexListResponse> {
  return apiFetch("/api/v1/indexes", { signal });
}

export function getIndex(
  name: string,
  signal?: AbortSignal
): Promise<{ index: IndexProfile; request_id: string }> {
  return apiFetch(`/api/v1/indexes/${encodeURIComponent(name)}`, { signal });
}

export interface CreateIndexBody {
  name: string;
  backend?: string;
  index_type?: string;
  build_params?: Record<string, number>;
  search_params?: Record<string, number>;
  make_active?: boolean;
  overwrite?: boolean;
  description?: string;
}

export function createIndex(
  body: CreateIndexBody,
  signal?: AbortSignal
): Promise<JobAcceptedResponse> {
  return apiFetch("/api/v1/indexes", { method: "POST", body, signal });
}

export function benchmarkRecipes(
  body: { recipes: string[]; k?: number; params?: Record<string, Record<string, number>>; persist?: boolean },
  signal?: AbortSignal
): Promise<JobAcceptedResponse> {
  return apiFetch("/api/v1/indexes/benchmark", { method: "POST", body, signal });
}

export function switchIndex(
  name: string,
  signal?: AbortSignal
): Promise<{ index_name: string; action: string; active?: string | null; request_id: string }> {
  return apiFetch(`/api/v1/indexes/${encodeURIComponent(name)}/switch`, {
    method: "POST",
    signal,
  });
}

export function deleteIndex(
  name: string,
  signal?: AbortSignal
): Promise<{ index_name: string; action: string; request_id: string }> {
  return apiFetch(`/api/v1/indexes/${encodeURIComponent(name)}`, { method: "DELETE", signal });
}

export function getCompatibility(
  name: string,
  signal?: AbortSignal
): Promise<{ report: CompatibilityReport; request_id: string }> {
  return apiFetch(`/api/v1/indexes/${encodeURIComponent(name)}/compatibility`, { signal });
}
