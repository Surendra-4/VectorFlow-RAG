"use client";

import * as React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { indexesApi } from "@/lib/api";
import type { BenchmarkResultRow, RecipeSpec } from "@/lib/api/types";
import { useJobProgress } from "@/lib/hooks/useJobProgress";
import { formatBytes, formatLatencyMs } from "@/lib/utils/format";

/**
 * Benchmark several FAISS recipes over the current corpus and compare them
 * side-by-side (Recall@K / MRR / latency / QPS / size). Runs as a background
 * job so a large sweep never blocks the UI.
 */
export function BenchmarkPanel() {
  const [recipes, setRecipes] = React.useState<RecipeSpec[]>([]);
  const [selected, setSelected] = React.useState<Set<string>>(new Set(["flat", "hnsw", "ivf_pq"]));
  const [error, setError] = React.useState<Error | null>(null);
  const job = useJobProgress();

  React.useEffect(() => {
    void indexesApi
      .listRecipes()
      .then((r) => setRecipes(r.recipes))
      .catch((e) => setError(e instanceof Error ? e : new Error(String(e))));
  }, []);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const run = async () => {
    setError(null);
    try {
      const res = await indexesApi.benchmarkRecipes({
        recipes: [...selected],
        k: 10,
        persist: true,
      });
      job.start(res.job_id);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  };

  const rows = (job.progress?.result?.results as BenchmarkResultRow[] | undefined) ?? [];
  const running = job.streaming && !job.done;

  return (
    <Card>
      <CardTitle>Benchmark recipes</CardTitle>
      <ErrorBox error={error} />

      <div className="mb-3 flex flex-wrap gap-2">
        {recipes.map((r) => {
          const on = selected.has(r.id);
          return (
            <button
              key={r.id}
              type="button"
              aria-pressed={on}
              onClick={() => toggle(r.id)}
              className={`rounded border px-2 py-1 text-xs transition-colors ${
                on
                  ? "border-accent bg-accent/10 text-fg"
                  : "border-border text-fg-muted hover:text-fg"
              }`}
            >
              {r.label}
            </button>
          );
        })}
      </div>

      <Button size="sm" disabled={running || selected.size === 0} loading={running} onClick={run}>
        Run benchmark
      </Button>

      {job.progress && running && (
        <div className="mt-3">
          <ProgressBar value={job.progress.progress} label={job.progress.message} />
        </div>
      )}

      {rows.length > 0 && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-fg-muted">
              <tr className="border-b border-border">
                <th className="py-1 pr-3">Recipe</th>
                <th className="py-1 pr-3">Recall@K</th>
                <th className="py-1 pr-3">MRR</th>
                <th className="py-1 pr-3">p50</th>
                <th className="py-1 pr-3">p95</th>
                <th className="py-1 pr-3">QPS</th>
                <th className="py-1 pr-3">Size</th>
              </tr>
            </thead>
            <tbody>
              {rows
                .slice()
                .sort((a, b) => b.recall_at_k - a.recall_at_k)
                .map((r) => (
                  <tr key={r.recipe} className="border-b border-border/50">
                    <td className="py-1 pr-3">
                      <Badge tone="accent">{r.recipe}</Badge>
                    </td>
                    <td className="py-1 pr-3 font-mono">{r.recall_at_k.toFixed(3)}</td>
                    <td className="py-1 pr-3 font-mono">{r.mrr.toFixed(3)}</td>
                    <td className="py-1 pr-3">{formatLatencyMs(r.latency_ms_p50)}</td>
                    <td className="py-1 pr-3">{formatLatencyMs(r.latency_ms_p95)}</td>
                    <td className="py-1 pr-3 font-mono">{r.queries_per_sec.toFixed(0)}</td>
                    <td className="py-1 pr-3">{formatBytes(r.index_size_bytes)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          {typeof job.progress?.result?.artifact === "string" && (
            <p className="mt-2 text-xs text-fg-muted">
              Artifact saved: {String(job.progress.result.artifact)}
            </p>
          )}
        </div>
      )}
    </Card>
  );
}
