// src/lib/hooks/useStreamingAsk.ts

"use client";

import { useCallback, useRef, useState } from "react";
import { askApi, type AskMetrics, type AskRequest, type RetrievalResult } from "@/lib/api";

export type StreamState = "idle" | "streaming" | "done" | "error";

export interface UseStreamingAskResult {
  state: StreamState;
  answer: string;
  sources: RetrievalResult[];
  metrics: AskMetrics | null;
  errorMessage: string | null;
  start: (req: AskRequest) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

/**
 * Hook around `streamAsk` that aggregates SSE events into local state.
 *
 * Reset semantics: each `start()` call resets state before opening a new
 * connection. `cancel()` aborts the in-flight stream without altering
 * the displayed answer (lets the user keep what was generated so far).
 */
export function useStreamingAsk(): UseStreamingAskResult {
  const [state, setState] = useState<StreamState>("idle");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<RetrievalResult[]>([]);
  const [metrics, setMetrics] = useState<AskMetrics | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    ctrlRef.current?.abort();
    ctrlRef.current = null;
    setState((s) => (s === "streaming" ? "done" : s));
  }, []);

  const reset = useCallback(() => {
    cancel();
    setAnswer("");
    setSources([]);
    setMetrics(null);
    setErrorMessage(null);
    setState("idle");
  }, [cancel]);

  const start = useCallback(
    async (req: AskRequest) => {
      reset();
      const ctrl = new AbortController();
      ctrlRef.current = ctrl;
      setState("streaming");

      try {
        let accumulated = "";
        for await (const evt of askApi.streamAsk({ ...req, stream: true }, ctrl.signal)) {
          if (ctrl.signal.aborted) break;
          switch (evt.type) {
            case "sources":
              setSources(evt.sources);
              break;
            case "token":
              accumulated += evt.token;
              // Functional updater so React batches without dropping tokens.
              setAnswer((prev) => prev + evt.token);
              break;
            case "done":
              setMetrics(evt.metrics);
              // The final answer in `done` is authoritative — but only
              // update if it materially differs (it always should match
              // accumulated tokens; this is a safety net).
              if (evt.answer && evt.answer !== accumulated) {
                setAnswer(evt.answer);
              }
              setState("done");
              break;
            case "error":
              setErrorMessage(evt.message);
              setState("error");
              break;
          }
        }
        // If the stream ended without an explicit done/error, mark done.
        setState((s) => (s === "streaming" ? "done" : s));
      } catch (e) {
        if ((e as Error)?.name === "AbortError") {
          setState("done");
        } else {
          setErrorMessage((e as Error)?.message ?? String(e));
          setState("error");
        }
      } finally {
        ctrlRef.current = null;
      }
    },
    [reset]
  );

  return {
    state,
    answer,
    sources,
    metrics,
    errorMessage,
    start,
    cancel,
    reset,
  };
}
