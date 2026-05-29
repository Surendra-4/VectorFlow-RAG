// src/lib/hooks/usePolling.ts

"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";

export interface PollingResult<T> {
  data: T | null;
  error: ApiError | null;
  lastFetchedAt: number | null;
}

/**
 * Periodically re-runs `fetcher` at `intervalMs`.
 *
 * Behavior:
 *   - immediate fetch on mount
 *   - subsequent fetches every `intervalMs` ms
 *   - skips while `document.visibilityState === "hidden"` (saves backend load)
 *   - aborts in-flight on unmount or interval restart
 *   - swallows AbortError; surfaces ApiError
 */
export function usePolling<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  intervalMs: number,
  enabled = true
): PollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [lastFetchedAt, setLastFetchedAt] = useState<number | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let ctrl = new AbortController();

    const runOnce = async () => {
      if (typeof document !== "undefined" && document.visibilityState === "hidden") {
        return;
      }
      ctrl = new AbortController();
      try {
        const result = await fetcherRef.current(ctrl.signal);
        if (!cancelled) {
          setData(result);
          setError(null);
          setLastFetchedAt(Date.now());
        }
      } catch (e) {
        if (cancelled || (e as Error)?.name === "AbortError") return;
        if (e instanceof ApiError) setError(e);
        else setError(new ApiError({ status: 0, code: "unknown", message: String(e) }));
      }
    };

    runOnce();
    const timer = setInterval(runOnce, intervalMs);
    return () => {
      cancelled = true;
      ctrl.abort();
      clearInterval(timer);
    };
  }, [intervalMs, enabled]);

  return { data, error, lastFetchedAt };
}
