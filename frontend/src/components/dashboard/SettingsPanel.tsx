"use client";

import * as React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { Input } from "@/components/ui/Input";
import { Spinner } from "@/components/ui/Spinner";
import { cacheApi, statusApi, ApiError, type CacheStatsResponse, type StatusResponse } from "@/lib/api";
import { resolveBaseUrl } from "@/lib/api/client";
import { formatNumber, formatPercent, formatUptime } from "@/lib/utils/format";

/**
 * Read-only pipeline status + runtime backend URL override.
 *
 * Phase 10 doesn't switch models or backends from the UI — those still
 * live in the backend's env. The override here is purely client-side
 * (localStorage) so a Vercel-hosted frontend can point at any backend.
 */
export function SettingsPanel() {
  const [status, setStatus] = React.useState<StatusResponse | null>(null);
  const [cache, setCache] = React.useState<CacheStatsResponse | null>(null);
  const [error, setError] = React.useState<ApiError | Error | null>(null);
  const [loading, setLoading] = React.useState(true);

  const [baseUrlInput, setBaseUrlInput] = React.useState("");
  const [savedHint, setSavedHint] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, c] = await Promise.all([
        statusApi.getStatus(),
        cacheApi.getCacheStats(),
      ]);
      setStatus(s);
      setCache(c);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    setBaseUrlInput(resolveBaseUrl());
    void load();
  }, [load]);

  const saveBaseUrl = () => {
    try {
      if (baseUrlInput) {
        window.localStorage.setItem("vfr_api_base_url", baseUrlInput);
      } else {
        window.localStorage.removeItem("vfr_api_base_url");
      }
      setSavedHint("Saved — will take effect on next request. Reload for full effect.");
    } catch (e) {
      setSavedHint(`Could not save: ${(e as Error).message}`);
    }
  };

  const clearCache = async () => {
    try {
      await cacheApi.clearCache();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  };

  if (loading && !status) return <Spinner label="Loading status…" />;

  return (
    <div className="space-y-4">
      <ErrorBox error={error} onRetry={load} />

      <Card>
        <CardTitle>Backend URL (client-side override)</CardTitle>
        <p className="mb-2 text-xs text-fg-muted">
          Persisted in this browser only. Leave blank to fall back to
          <code className="mx-1">NEXT_PUBLIC_API_BASE_URL</code>.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Input
            value={baseUrlInput}
            onChange={(e) => setBaseUrlInput(e.target.value)}
            placeholder="http://localhost:8000"
            className="max-w-md"
          />
          <Button size="sm" variant="secondary" onClick={saveBaseUrl}>
            Save
          </Button>
        </div>
        {savedHint && <p className="mt-2 text-xs text-fg-muted">{savedHint}</p>}
      </Card>

      {status && (
        <Card>
          <CardTitle>Pipeline status</CardTitle>
          <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
            <Row label="App" value={`${status.app_name} ${status.app_version}`} />
            <Row label="Uptime" value={formatUptime(status.uptime_s)} />
            <Row label="Vector backend" value={status.vector_store_backend} />
            <Row label="Embedder model" value={status.embedder_model} />
            <Row
              label="Reranker"
              value={
                status.reranker_enabled
                  ? `on · ${status.reranker_model ?? ""}`
                  : "off"
              }
            />
            <Row label="Expansion" value={status.expansion_enabled ? "on" : "off"} />
            <Row label="Cache backend" value={status.cache_backend} />
            <Row label="Documents" value={formatNumber(status.documents_ingested)} />
            <Row label="Chunks" value={formatNumber(status.chunks_indexed)} />
            <Row
              label="Corpus fingerprint"
              value={
                <code className="break-all font-mono text-xs">
                  {status.corpus_fingerprint ?? "—"}
                </code>
              }
            />
          </dl>
        </Card>
      )}

      {cache && (
        <Card>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <CardTitle>Cache</CardTitle>
            <Button
              size="sm"
              variant="ghost"
              onClick={clearCache}
              disabled={cache.backend === "null"}
            >
              Clear cache
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge tone="accent">{cache.backend}</Badge>
            <Badge tone="neutral">hits {formatNumber(cache.hits)}</Badge>
            <Badge tone="neutral">misses {formatNumber(cache.misses)}</Badge>
            <Badge tone="neutral">sets {formatNumber(cache.sets)}</Badge>
            <Badge tone={cache.errors > 0 ? "warning" : "neutral"}>
              errors {formatNumber(cache.errors)}
            </Badge>
            <Badge tone="success">hit ratio {formatPercent(cache.hit_ratio)}</Badge>
          </div>
        </Card>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <>
      <dt className="text-xs uppercase tracking-wide text-fg-muted">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </>
  );
}
