// src/lib/api/client.ts

import { ApiError } from "./errors";
import { clearToken, getToken } from "@/lib/auth/token";

const DEFAULT_BASE_URL = "http://localhost:8000";

/**
 * Resolve the API base URL.
 *
 * Order of precedence:
 * 1. Runtime override stored in localStorage (lets the Vercel/hosted
 *    deployment let users point at their own backend without rebuild).
 * 2. Build-time `NEXT_PUBLIC_API_BASE_URL`.
 * 3. Default `http://localhost:8000`.
 */
export function resolveBaseUrl(): string {
  if (typeof window !== "undefined") {
    try {
      const override = window.localStorage.getItem("vfr_api_base_url");
      if (override) return stripTrailingSlash(override);
    } catch {
      // localStorage can throw in private-browsing/test contexts — ignore.
    }
  }
  const envUrl =
    typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_BASE_URL;
  return stripTrailingSlash(envUrl || DEFAULT_BASE_URL);
}

function stripTrailingSlash(url: string): string {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

/**
 * Headers added to every backend request, shared by `apiFetch` and the
 * streaming `fetch` paths so they behave identically:
 *  - the bearer token, when the user is signed in;
 *  - `ngrok-skip-browser-warning` when the backend is an ngrok free tunnel, so
 *    ngrok serves the real JSON/SSE response instead of its HTML interstitial.
 *    The header is harmless against any other backend (it's simply ignored).
 */
export function defaultHeaders(baseUrl: string): Record<string, string> {
  const headers: Record<string, string> = {};
  if (typeof window !== "undefined") {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }
  if (baseUrl.includes("ngrok")) headers["ngrok-skip-browser-warning"] = "true";
  return headers;
}

export interface ApiRequestInit {
  method?: string;
  body?: unknown; // JSON-serializable
  headers?: Record<string, string>;
  signal?: AbortSignal;
  // When true, returns the raw Response so callers can stream the body
  // themselves (used by SSE). When false (default), parses JSON.
  rawResponse?: boolean;
  // When true, expects multipart form-data; `body` is sent as-is and the
  // browser sets the boundary automatically.
  formData?: boolean;
  // Per-request base URL override (useful for tests).
  baseUrl?: string;
}

/**
 * Thin fetch wrapper that:
 * - prepends the base URL
 * - JSON-encodes the body (unless `formData` is set)
 * - normalizes errors into `ApiError`
 * - propagates `X-Request-ID` when the caller provides one
 */
export async function apiFetch<T = unknown>(
  path: string,
  init: ApiRequestInit = {}
): Promise<T> {
  const base = init.baseUrl ?? resolveBaseUrl();
  const url = `${base}${path.startsWith("/") ? "" : "/"}${path}`;

  // Base headers (bearer token + ngrok skip) first, so an explicit caller
  // header always wins.
  const headers: Record<string, string> = { ...defaultHeaders(base), ...init.headers };

  let body: BodyInit | undefined;

  if (init.body !== undefined) {
    if (init.formData) {
      body = init.body as BodyInit;
    } else {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(init.body);
    }
  }

  let res: Response;
  try {
    res = await fetch(url, {
      method: init.method ?? "GET",
      headers,
      body,
      signal: init.signal,
    });
  } catch (cause) {
    throw new ApiError({
      status: 0,
      code: "network_error",
      message:
        cause instanceof Error
          ? `Network error: ${cause.message}`
          : "Network error",
    });
  }

  const requestId = res.headers.get("X-Request-ID");

  if (init.rawResponse) {
    if (!res.ok) {
      await throwFromError(res, requestId);
    }
    return res as unknown as T;
  }

  // 204 — no body to parse.
  if (res.status === 204) {
    return undefined as unknown as T;
  }

  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    parsed = undefined;
  }

  if (!res.ok) {
    // An expired/invalid token → drop it so the app falls back to the login gate.
    if (res.status === 401 && typeof window !== "undefined") clearToken();
    const errorBody = (parsed as Record<string, unknown> | undefined) ?? {};
    throw new ApiError({
      status: res.status,
      code: typeof errorBody.code === "string" ? (errorBody.code as string) : `http_${res.status}`,
      message:
        typeof errorBody.message === "string"
          ? (errorBody.message as string)
          : res.statusText || `HTTP ${res.status}`,
      requestId,
      details: (errorBody.details as Record<string, unknown> | undefined) ?? null,
    });
  }

  return parsed as T;
}

async function throwFromError(res: Response, requestId: string | null): Promise<never> {
  let body: Record<string, unknown> = {};
  try {
    body = (await res.json()) as Record<string, unknown>;
  } catch {
    // ignore
  }
  throw new ApiError({
    status: res.status,
    code: typeof body.code === "string" ? (body.code as string) : `http_${res.status}`,
    message:
      typeof body.message === "string" ? (body.message as string) : res.statusText,
    requestId,
    details: (body.details as Record<string, unknown> | undefined) ?? null,
  });
}
