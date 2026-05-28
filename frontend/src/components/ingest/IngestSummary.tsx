"use client";

import { Badge } from "@/components/ui/Badge";
import { Card, CardTitle } from "@/components/ui/Card";
import type { IngestionResponse } from "@/lib/api";

export function IngestSummary({ result }: { result: IngestionResponse }) {
  const ok = result.failures.length === 0;
  return (
    <Card>
      <CardTitle>Ingestion result</CardTitle>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <Badge tone={ok ? "success" : "warning"}>
          {ok ? "All succeeded" : `${result.failures.length} failed`}
        </Badge>
        <Badge tone="neutral">{result.successes.length} succeeded</Badge>
        <Badge tone="neutral">{result.chunks} chunks indexed</Badge>
        <Badge tone="neutral">{result.documents_ingested} docs total</Badge>
      </div>

      {result.failures.length > 0 && (
        <details className="rounded border border-warning/40 bg-warning/10 p-3 text-sm">
          <summary className="cursor-pointer font-medium text-warning">
            Show failures
          </summary>
          <ul className="mt-2 space-y-1 text-fg-muted">
            {result.failures.map((f, i) => (
              <li key={`${i}-${f.path}`} className="break-all">
                <code className="text-xs">{f.path}</code> — {f.reason}
              </li>
            ))}
          </ul>
        </details>
      )}

      {result.corpus_fingerprint && (
        <p className="mt-3 break-all font-mono text-[10px] text-fg-muted">
          fingerprint: {result.corpus_fingerprint} · request_id: {result.request_id}
        </p>
      )}
    </Card>
  );
}
