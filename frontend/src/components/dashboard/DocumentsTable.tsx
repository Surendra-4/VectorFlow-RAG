"use client";

import * as React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { Spinner } from "@/components/ui/Spinner";
import { documentsApi, ApiError, type DocumentsResponse } from "@/lib/api";

export function DocumentsTable() {
  const [data, setData] = React.useState<DocumentsResponse | null>(null);
  const [error, setError] = React.useState<ApiError | Error | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [busyReset, setBusyReset] = React.useState(false);

  const load = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await documentsApi.listDocuments();
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  const onReset = async () => {
    const ok = window.confirm(
      "This will delete every chunk in the index. Are you sure?"
    );
    if (!ok) return;
    setBusyReset(true);
    try {
      await documentsApi.resetIndex(true);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusyReset(false);
    }
  };

  if (loading && !data) return <Spinner label="Loading documents…" />;
  if (error) return <ErrorBox error={error} onRetry={load} />;
  if (!data) return null;

  return (
    <Card>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <CardTitle>
          {data.total_documents} documents · {data.total_chunks} chunks
        </CardTitle>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={load}>
            Refresh
          </Button>
          <Button
            size="sm"
            variant="danger"
            onClick={onReset}
            loading={busyReset}
            disabled={data.total_documents === 0}
          >
            Reset index
          </Button>
        </div>
      </div>

      {data.documents.length === 0 ? (
        <p className="text-sm text-fg-muted">
          No documents indexed yet. Head to the Ingest page to add some.
        </p>
      ) : (
        <table className="w-full text-left text-xs">
          <thead className="text-fg-muted">
            <tr>
              <th className="py-1 font-normal">Document</th>
              <th className="py-1 font-normal">Source</th>
              <th className="py-1 text-right font-normal">Chunks</th>
              <th className="py-1 font-normal">doc_id</th>
            </tr>
          </thead>
          <tbody>
            {data.documents.map((d) => (
              <tr key={d.doc_id} className="border-t border-border align-top">
                <td dir="auto" className="py-1">
                  {d.document_name ?? <span className="text-fg-muted">(unnamed)</span>}
                </td>
                <td className="py-1 break-all font-mono text-[10px] text-fg-muted">
                  {d.source_path ?? "—"}
                </td>
                <td className="py-1 text-right">
                  <Badge tone="neutral">{d.chunk_count}</Badge>
                </td>
                <td className="py-1 break-all font-mono text-[10px] text-fg-muted">
                  {d.doc_id}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}
