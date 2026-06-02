"use client";

import * as React from "react";
import { jobsApi } from "@/lib/api";

export interface JobProgress {
  status: string;
  progress: number;
  message: string;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

/**
 * Track a background job's progress via SSE.
 *
 * `start(jobId)` opens a stream; events update `progress`. The stream closes
 * automatically when the job reaches a terminal state. Aborts on unmount.
 */
export function useJobProgress() {
  const [jobId, setJobId] = React.useState<string | null>(null);
  const [progress, setProgress] = React.useState<JobProgress | null>(null);
  const [streaming, setStreaming] = React.useState(false);
  const ctrlRef = React.useRef<AbortController | null>(null);

  const stop = React.useCallback(() => {
    ctrlRef.current?.abort();
    ctrlRef.current = null;
    setStreaming(false);
  }, []);

  const start = React.useCallback((id: string) => {
    stop();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setJobId(id);
    setStreaming(true);
    setProgress({ status: "pending", progress: 0, message: "Starting…" });

    void jobsApi
      .streamJob(
        id,
        (event, data) => {
          if (event === "progress" || event === "done") {
            setProgress({
              status: String(data.status ?? "running"),
              progress: Number(data.progress ?? 0),
              message: String(data.message ?? ""),
              result: (data.result as Record<string, unknown>) ?? null,
              error: (data.error as string) ?? null,
            });
          }
        },
        ctrl.signal
      )
      .catch((e) => {
        if ((e as Error)?.name !== "AbortError") {
          setProgress((p) => ({
            status: "failed",
            progress: p?.progress ?? 0,
            message: "Stream error",
            error: String(e),
          }));
        }
      })
      .finally(() => setStreaming(false));
  }, [stop]);

  React.useEffect(() => () => stop(), [stop]);

  const done = progress ? TERMINAL.has(progress.status) : false;
  return { jobId, progress, streaming, done, start, stop };
}
