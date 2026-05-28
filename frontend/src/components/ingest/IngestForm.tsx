"use client";

import * as React from "react";
import { ApiError, type IngestionResponse, ingestApi } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { Textarea } from "@/components/ui/Input";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { DropZone } from "./DropZone";
import { IngestSummary } from "./IngestSummary";

type Mode = "files" | "text";

/**
 * Combined ingest UI:
 *   - File-drop tab: drag-and-drop or browse → multipart upload
 *   - Text-paste tab: blank lines separate documents
 *
 * Reset toggle controls whether the index is wiped before this batch.
 */
export function IngestForm() {
  const [mode, setMode] = React.useState<Mode>("files");
  const [reset, setReset] = React.useState(true);
  const [files, setFiles] = React.useState<File[]>([]);
  const [text, setText] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [result, setResult] = React.useState<IngestionResponse | null>(null);
  const [error, setError] = React.useState<ApiError | Error | null>(null);

  const onFiles = (incoming: File[]) => {
    setFiles((prev) => [...prev, ...incoming]);
  };

  const removeFile = (idx: number) =>
    setFiles((prev) => prev.filter((_, i) => i !== idx));

  const submit = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      if (mode === "files") {
        if (files.length === 0) {
          setError(new Error("Select at least one file."));
          return;
        }
        const res = await ingestApi.ingestFiles(files, reset);
        setResult(res);
        setFiles([]);
      } else {
        const docs = text
          .split(/\n\s*\n/)
          .map((s) => s.trim())
          .filter(Boolean);
        if (docs.length === 0) {
          setError(new Error("Enter at least one document (blank line separates docs)."));
          return;
        }
        const res = await ingestApi.ingestText({ documents: docs, reset });
        setResult(res);
        setText("");
      }
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div role="tablist" aria-label="Ingestion mode" className="flex gap-1">
            {(["files", "text"] as Mode[]).map((m) => (
              <button
                key={m}
                role="tab"
                aria-selected={mode === m}
                onClick={() => setMode(m)}
                className={
                  mode === m
                    ? "rounded bg-accent px-3 py-1 text-xs font-medium text-accent-fg"
                    : "rounded px-3 py-1 text-xs text-fg-muted hover:bg-surface-raised"
                }
              >
                {m === "files" ? "Upload files" : "Paste text"}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-2 text-xs text-fg-muted">
            <input
              type="checkbox"
              checked={reset}
              onChange={(e) => setReset(e.target.checked)}
              disabled={busy}
            />
            Reset index before ingesting
          </label>
        </div>

        {mode === "files" && (
          <div className="space-y-3">
            <DropZone onFiles={onFiles} disabled={busy} />
            {files.length > 0 && (
              <ul className="divide-y divide-border rounded border border-border">
                {files.map((f, i) => (
                  <li
                    key={`${i}-${f.name}`}
                    className="flex items-center justify-between gap-2 px-3 py-2 text-sm"
                  >
                    <span className="truncate text-fg" title={f.name}>
                      {f.name}
                    </span>
                    <span className="shrink-0 text-xs text-fg-muted">
                      {(f.size / 1024).toFixed(1)} KB
                    </span>
                    <button
                      onClick={() => removeFile(i)}
                      disabled={busy}
                      className="ml-2 text-xs text-fg-muted hover:text-danger"
                      aria-label={`Remove ${f.name}`}
                    >
                      remove
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {mode === "text" && (
          <div>
            <CardTitle>Paste documents</CardTitle>
            <p className="mb-2 text-xs text-fg-muted">
              Separate documents with a blank line.
            </p>
            <Textarea
              rows={10}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="First document goes here.&#10;&#10;Second document starts after a blank line."
              disabled={busy}
            />
          </div>
        )}

        <div className="mt-4 flex justify-end">
          <Button onClick={submit} loading={busy}>
            {busy ? "Ingesting…" : "Ingest"}
          </Button>
        </div>
      </Card>

      <ErrorBox error={error} />

      {result && <IngestSummary result={result} />}
    </div>
  );
}
