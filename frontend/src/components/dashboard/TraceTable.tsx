"use client";

import * as React from "react";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { metricsApi } from "@/lib/api";
import { usePolling } from "@/lib/hooks/usePolling";
import { formatLatencyMs, truncate } from "@/lib/utils/format";

interface TraceRow {
  original_query?: string;
  total_latency_ms?: number;
  expansion_latency_ms?: number;
  rerank_latency_ms?: number;
  reranker_used?: boolean;
  from_cache?: boolean;
  final_result_count?: number;
  fused_pool_size?: number;
  strategies_used?: string[];
  [key: string]: unknown;
}

export function TraceTable() {
  const [limit, setLimit] = React.useState(20);
  const { data, error } = usePolling(
    (s) => metricsApi.getRecentTraces(limit, s),
    5000
  );
  const [open, setOpen] = React.useState<number | null>(null);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <p className="text-sm text-fg-muted">Loading traces…</p>;

  const traces = (data.traces as TraceRow[]) ?? [];

  return (
    <Card>
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-fg-muted">
          Recent retrievals
        </h2>
        <label className="text-xs text-fg-muted">
          Limit:{" "}
          <select
            className="ml-1 rounded border border-border bg-surface px-2 py-1 text-xs"
            value={limit}
            onChange={(e) => setLimit(parseInt(e.target.value, 10))}
          >
            {[10, 20, 50, 100].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>

      {traces.length === 0 ? (
        <p className="text-sm text-fg-muted">No recent retrievals yet.</p>
      ) : (
        <table className="w-full text-left text-xs">
          <thead className="text-fg-muted">
            <tr>
              <th className="py-1 font-normal">Query</th>
              <th className="py-1 font-normal">Total</th>
              <th className="py-1 font-normal">Cache</th>
              <th className="py-1 font-normal">Pool</th>
              <th className="py-1 font-normal">Returned</th>
              <th className="py-1 font-normal">Strategies</th>
              <th className="py-1 font-normal" />
            </tr>
          </thead>
          <tbody>
            {traces.map((t, i) => {
              const isOpen = open === i;
              return (
                <React.Fragment key={i}>
                  <tr className="border-t border-border align-top">
                    <td className="py-1 font-mono">
                      {truncate(String(t.original_query ?? ""), 60)}
                    </td>
                    <td className="py-1">{formatLatencyMs(t.total_latency_ms ?? null)}</td>
                    <td className="py-1">
                      {t.from_cache ? <Badge tone="success">hit</Badge> : <Badge tone="neutral">miss</Badge>}
                    </td>
                    <td className="py-1">{t.fused_pool_size ?? "—"}</td>
                    <td className="py-1">{t.final_result_count ?? "—"}</td>
                    <td className="py-1">
                      {(t.strategies_used ?? []).length === 0
                        ? "—"
                        : (t.strategies_used ?? []).join(", ")}
                    </td>
                    <td className="py-1">
                      <button
                        className="text-xs text-accent hover:underline"
                        onClick={() => setOpen(isOpen ? null : i)}
                      >
                        {isOpen ? "hide" : "details"}
                      </button>
                    </td>
                  </tr>
                  {isOpen && (
                    <tr className="border-t border-border bg-surface-raised">
                      <td colSpan={7} className="py-2">
                        <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded bg-surface p-3 text-[11px] text-fg-muted">
                          {JSON.stringify(t, null, 2)}
                        </pre>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      )}
    </Card>
  );
}
