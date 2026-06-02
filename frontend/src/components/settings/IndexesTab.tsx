"use client";

import * as React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { Spinner } from "@/components/ui/Spinner";
import { indexesApi, statusApi } from "@/lib/api";
import type { CompatibilityReport, IndexProfile } from "@/lib/api/types";
import { useJobProgress } from "@/lib/hooks/useJobProgress";
import { formatNumber } from "@/lib/utils/format";
import { IndexBuilder } from "./IndexBuilder";
import { BenchmarkPanel } from "./BenchmarkPanel";

export function IndexesTab() {
  const [indexes, setIndexes] = React.useState<IndexProfile[]>([]);
  const [active, setActive] = React.useState<string | null>(null);
  const [dim, setDim] = React.useState(0);
  const [nVectors, setNVectors] = React.useState(0);
  const [error, setError] = React.useState<Error | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [busy, setBusy] = React.useState(false);
  const [compat, setCompat] = React.useState<Record<string, CompatibilityReport>>({});

  const build = useJobProgress();

  const load = React.useCallback(async () => {
    setError(null);
    try {
      const [list, status] = await Promise.all([
        indexesApi.listIndexes(),
        statusApi.getStatus().catch(() => null),
      ]);
      setIndexes(list.indexes);
      setActive(list.active ?? null);
      if (status) {
        setDim(status.embedder_dimension ?? 0);
        setNVectors(status.chunks_indexed ?? 0);
      }
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  // When a build job finishes, refresh the list.
  React.useEffect(() => {
    if (build.done && build.progress?.status === "succeeded") {
      void load();
    }
  }, [build.done, build.progress?.status, load]);

  const switchTo = async (name: string) => {
    setBusy(true);
    try {
      await indexesApi.switchIndex(name);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (name: string) => {
    setBusy(true);
    try {
      await indexesApi.deleteIndex(name);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  const checkCompat = async (name: string) => {
    try {
      const res = await indexesApi.getCompatibility(name);
      setCompat((prev) => ({ ...prev, [name]: res.report }));
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  };

  if (loading) return <Spinner label="Loading indexes…" />;

  return (
    <div className="space-y-4">
      <ErrorBox error={error} onRetry={load} />

      <Card>
        <div className="mb-2 flex items-center justify-between">
          <CardTitle>Named indexes</CardTitle>
          <Button size="sm" variant="ghost" onClick={load} disabled={busy}>
            Refresh
          </Button>
        </div>
        {indexes.length === 0 ? (
          <p className="text-sm text-fg-muted">
            No named indexes yet. Build one below to compare backends and FAISS recipes.
          </p>
        ) : (
          <ul className="divide-y divide-border">
            {indexes.map((idx) => {
              const report = compat[idx.name];
              const recall = idx.metrics?.recall_at_k as number | undefined;
              return (
                <li key={idx.name} className="py-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{idx.name}</span>
                        {idx.name === active && <Badge tone="success">active</Badge>}
                        <Badge tone="neutral">{idx.backend}</Badge>
                        <Badge tone="accent">{idx.index_type}</Badge>
                      </div>
                      <div className="text-xs text-fg-muted">
                        {formatNumber(idx.num_vectors)} vecs · dim {idx.vector_dimension} ·{" "}
                        {idx.embedding_model}
                        {recall != null && <span> · recall@k {recall.toFixed(3)}</span>}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={busy || idx.name === active}
                        onClick={() => switchTo(idx.name)}
                      >
                        Use
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => checkCompat(idx.name)}>
                        Check
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={busy}
                        onClick={() => remove(idx.name)}
                      >
                        Delete
                      </Button>
                    </div>
                  </div>
                  {report && <CompatibilityNote report={report} />}
                </li>
              );
            })}
          </ul>
        )}
      </Card>

      {/* Build-job progress */}
      {build.progress && !build.done && (
        <Card>
          <CardTitle>Building index…</CardTitle>
          <ProgressBar value={build.progress.progress} label={build.progress.message} />
        </Card>
      )}
      {build.done && build.progress?.status === "failed" && (
        <Card>
          <p className="text-sm text-danger">Build failed: {build.progress.error}</p>
        </Card>
      )}

      <IndexBuilder dim={dim} nVectors={nVectors} onCreate={(jobId) => build.start(jobId)} />

      <BenchmarkPanel />
    </div>
  );
}

function CompatibilityNote({ report }: { report: CompatibilityReport }) {
  const tone =
    report.action === "reuse"
      ? "border-success/40 bg-success/10"
      : report.action === "rebuild"
        ? "border-warning/40 bg-warning/10"
        : "border-danger/40 bg-danger/10";
  return (
    <div className={`mt-2 rounded border p-2 text-xs ${tone}`}>
      <p className="font-medium">{report.message}</p>
      {report.issues.length > 0 && (
        <ul className="mt-1 list-inside list-disc text-fg-muted">
          {report.issues.map((iss, i) => (
            <li key={i}>
              <span className="uppercase">[{iss.severity}]</span> {iss.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
