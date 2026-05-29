// src/lib/hooks/useApi.ts

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";

export type ApiState = "idle" | "loading" | "success" | "error";

export interface ApiResult<T> {
  state: ApiState;
  data: T | null;
  error: ApiError | null;
  refresh: () => Promise<void>;
}

/**
 * Generic loader hook.
 *
 * - cancels in-flight requests on dep change / unmount via AbortController
 * - exposes a stable `refresh()` for manual reloads
 * - normalizes errors into ApiError (Errors thrown by callers should already be ApiError)
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: ReadonlyArray<unknown> = []
): ApiResult<T> {
  const [state, setState] = useState<ApiState>("idle");
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const run = useCallback(async (): Promise<void> => {
    const ctrl = new AbortController();
    setState("loading");
    setError(null);
    try {
      const result = await fetcherRef.current(ctrl.signal);
      setData(result);
      setState("success");
    } catch (e) {
      if ((e as Error)?.name === "AbortError") return;
      const apiErr = e instanceof ApiError
        ? e
        : new ApiError({ status: 0, code: "unknown", message: String(e) });
      setError(apiErr);
      setState("error");
    }
  }, []);

  useEffect(() => {
    const ctrl = new AbortController();
    let cancelled = false;
    setState("loading");
    setError(null);
    fetcherRef
      .current(ctrl.signal)
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setState("success");
        }
      })
      .catch((e) => {
        if (cancelled || (e as Error)?.name === "AbortError") return;
        const apiErr =
          e instanceof ApiError
            ? e
            : new ApiError({ status: 0, code: "unknown", message: String(e) });
        setError(apiErr);
        setState("error");
      });
    return () => {
      cancelled = true;
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { state, data, error, refresh: run };
}
