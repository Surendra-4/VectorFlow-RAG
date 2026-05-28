"use client";

import * as React from "react";
import { Button } from "@/components/ui/Button";
import { Textarea } from "@/components/ui/Input";
import { Card, CardTitle } from "@/components/ui/Card";
import { Spinner } from "@/components/ui/Spinner";
import { MessageBubble } from "./MessageBubble";
import { LatencyChip } from "./LatencyChip";
import { SourcePanel } from "@/components/citations/SourcePanel";
import { useStreamingAsk } from "@/lib/hooks/useStreamingAsk";

/**
 * Single-turn chat UI:
 *
 *   - user types a query
 *   - on submit, streamingAsk runs:
 *       sources appear in the right column once retrieval completes
 *       tokens stream into the answer bubble as they arrive
 *       done event finalizes metrics
 *
 * Phase 10 keeps this single-turn (no conversation history). Conversation
 * persistence is reserved for a later phase.
 */
export function ChatInterface() {
  const [query, setQuery] = React.useState("");
  const [submittedQuery, setSubmittedQuery] = React.useState<string | null>(null);
  const [kDocs, setKDocs] = React.useState(3);
  const stream = useStreamingAsk();

  const submit = React.useCallback(() => {
    const q = query.trim();
    if (!q || stream.state === "streaming") return;
    setSubmittedQuery(q);
    void stream.start({ query: q, k_docs: kDocs, return_sources: true });
  }, [query, kDocs, stream]);

  const onSubmit: React.FormEventHandler = (e) => {
    e.preventDefault();
    submit();
  };

  // Ctrl/Cmd+Enter submits without leaving the textarea.
  const onKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      {/* Left: conversation column */}
      <div className="space-y-4 lg:col-span-2">
        <Card>
          <form onSubmit={onSubmit} className="space-y-3">
            <label htmlFor="ask-query" className="sr-only">
              Question
            </label>
            <Textarea
              id="ask-query"
              rows={3}
              placeholder="Ask a question about your documents…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={stream.state === "streaming"}
              aria-label="Question"
            />
            <div className="flex flex-wrap items-center justify-between gap-2">
              <label className="text-xs text-fg-muted">
                Context docs:{" "}
                <select
                  className="ml-1 rounded border border-border bg-surface px-2 py-1 text-xs text-fg"
                  value={kDocs}
                  onChange={(e) => setKDocs(parseInt(e.target.value, 10))}
                  disabled={stream.state === "streaming"}
                >
                  {[1, 2, 3, 5, 8].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              <div className="flex items-center gap-2">
                {stream.state === "streaming" && (
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={stream.cancel}
                  >
                    Cancel
                  </Button>
                )}
                <Button
                  type="submit"
                  loading={stream.state === "streaming"}
                  disabled={!query.trim()}
                  size="sm"
                >
                  {stream.state === "streaming" ? "Streaming…" : "Ask"}
                </Button>
              </div>
            </div>
            <p className="text-xs text-fg-muted">
              Tip: <kbd className="rounded border border-border bg-surface-raised px-1">⌘</kbd>{" "}
              + <kbd className="rounded border border-border bg-surface-raised px-1">Enter</kbd>{" "}
              to send.
            </p>
          </form>
        </Card>

        {submittedQuery && (
          <div className="space-y-3">
            <MessageBubble role="user">{submittedQuery}</MessageBubble>

            <MessageBubble role="assistant">
              {stream.state === "streaming" && !stream.answer && (
                <Spinner label="Retrieving and generating…" />
              )}
              {stream.answer && stream.answer}
              {stream.state === "streaming" && stream.answer && (
                <span className="ml-0.5 inline-block h-3 w-2 animate-pulse bg-fg-muted align-middle" aria-hidden="true" />
              )}
              {stream.state === "error" && (
                <span className="text-danger">Error: {stream.errorMessage}</span>
              )}
            </MessageBubble>

            <div className="flex items-center justify-between">
              <LatencyChip metrics={stream.metrics} />
              {stream.state === "done" && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    stream.reset();
                    setSubmittedQuery(null);
                  }}
                >
                  Clear
                </Button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Right: source / citation column */}
      <aside aria-label="Sources" className="space-y-3">
        <Card>
          <CardTitle>Sources</CardTitle>
          <SourcePanel sources={stream.sources} />
        </Card>
      </aside>
    </div>
  );
}
