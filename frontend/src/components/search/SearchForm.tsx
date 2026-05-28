"use client";

import * as React from "react";
import { searchApi, ApiError, type SearchResponse } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { Spinner } from "@/components/ui/Spinner";
import { SourcePanel } from "@/components/citations/SourcePanel";

export function SearchForm() {
  const [query, setQuery] = React.useState("");
  const [k, setK] = React.useState(5);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<ApiError | Error | null>(null);
  const [resp, setResp] = React.useState<SearchResponse | null>(null);

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const q = query.trim();
    if (!q) return;
    setBusy(true);
    setError(null);
    try {
      const r = await searchApi.searchDocuments({ query: q, k });
      setResp(r);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <form onSubmit={submit} className="space-y-3">
          <label htmlFor="search-q" className="sr-only">
            Search query
          </label>
          <Input
            id="search-q"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search…"
            disabled={busy}
          />
          <div className="flex items-center justify-between gap-2">
            <label className="text-xs text-fg-muted">
              Results:{" "}
              <select
                className="ml-1 rounded border border-border bg-surface px-2 py-1 text-xs"
                value={k}
                onChange={(e) => setK(parseInt(e.target.value, 10))}
                disabled={busy}
              >
                {[3, 5, 10, 20].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <Button type="submit" size="sm" loading={busy} disabled={!query.trim()}>
              {busy ? "Searching…" : "Search"}
            </Button>
          </div>
        </form>
      </Card>

      <ErrorBox error={error} onRetry={() => submit()} />

      {busy && !resp && <Spinner label="Retrieving…" />}

      {resp && (
        <Card>
          <CardTitle>
            {resp.results.length} result{resp.results.length === 1 ? "" : "s"}
          </CardTitle>
          {resp.results.length === 0 ? (
            <p className="text-sm text-fg-muted">
              Nothing matched. Try a different query or ingest more documents.
            </p>
          ) : (
            <SourcePanel sources={resp.results} />
          )}
        </Card>
      )}
    </div>
  );
}
