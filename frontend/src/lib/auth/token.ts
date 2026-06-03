// src/lib/auth/token.ts

/**
 * JWT access-token storage. Persisted in localStorage so a refresh keeps the
 * session; a tiny in-memory mirror avoids a localStorage read on every request.
 * A subscriber list lets the API client clear the token on a 401.
 */

const KEY = "vfr_access_token";

let cached: string | null | undefined; // undefined = not yet read
const listeners = new Set<(t: string | null) => void>();

export function getToken(): string | null {
  if (cached !== undefined) return cached;
  try {
    cached = window.localStorage.getItem(KEY);
  } catch {
    cached = null;
  }
  return cached;
}

export function setToken(token: string | null): void {
  cached = token;
  try {
    if (token) window.localStorage.setItem(KEY, token);
    else window.localStorage.removeItem(KEY);
  } catch {
    // private mode / SSR — in-memory mirror still works for the session.
  }
  listeners.forEach((fn) => fn(token));
}

export function clearToken(): void {
  setToken(null);
}

export function onTokenChange(fn: (t: string | null) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
