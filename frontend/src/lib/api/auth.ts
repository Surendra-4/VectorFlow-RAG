// src/lib/api/auth.ts

import { apiFetch, resolveBaseUrl } from "./client";
import type { AuthProvidersInfo, AuthUser, TokenResponse, UserStats } from "./types";

interface MeResponseShape {
  user: AuthUser;
  request_id: string;
}

export function getProviders(signal?: AbortSignal): Promise<AuthProvidersInfo> {
  return apiFetch("/api/v1/auth/providers", { signal });
}

export function signup(
  body: { email: string; password: string; name?: string },
  signal?: AbortSignal
): Promise<TokenResponse> {
  return apiFetch("/api/v1/auth/signup", { method: "POST", body, signal });
}

export function login(
  body: { email: string; password: string },
  signal?: AbortSignal
): Promise<TokenResponse> {
  return apiFetch("/api/v1/auth/login", { method: "POST", body, signal });
}

export function me(signal?: AbortSignal): Promise<MeResponseShape> {
  return apiFetch("/api/v1/auth/me", { signal });
}

export function logout(signal?: AbortSignal): Promise<{ message: string }> {
  return apiFetch("/api/v1/auth/logout", { method: "POST", signal });
}

export function resetRequest(
  email: string,
  signal?: AbortSignal
): Promise<{ message: string; reset_link?: string | null }> {
  return apiFetch("/api/v1/auth/reset/request", { method: "POST", body: { email }, signal });
}

export function resetConfirm(
  token: string,
  password: string,
  signal?: AbortSignal
): Promise<TokenResponse> {
  return apiFetch("/api/v1/auth/reset/confirm", { method: "POST", body: { token, password }, signal });
}

export function getMyStats(signal?: AbortSignal): Promise<{ stats: UserStats; request_id: string }> {
  return apiFetch("/api/v1/auth/me/stats", { signal });
}

export function resetMyStats(signal?: AbortSignal): Promise<{ stats: UserStats; request_id: string }> {
  return apiFetch("/api/v1/auth/me/stats/reset", { method: "POST", signal });
}

/** Full URL to kick off an OAuth provider (used as an <a href> — a real
 * top-level navigation, so the provider redirect + state cookie work). */
export function oauthStartUrl(provider: "google" | "github"): string {
  return `${resolveBaseUrl()}/api/v1/auth/${provider}`;
}
