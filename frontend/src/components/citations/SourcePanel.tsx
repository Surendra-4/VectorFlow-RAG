"use client";

import * as React from "react";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import type { RetrievalResult } from "@/lib/api";

interface SourcePanelProps {
  sources: RetrievalResult[];
  /** When set, that source is highlighted (e.g. when user hovers a citation). */
  highlightedIndex?: number | null;
}

/**
 * Renders the per-chunk source list with full provenance.
 *
 * Each card shows:
 *   - chunk preview (with text wrap)
 *   - document / page metadata
 *   - hybrid_score + per-modality ranks (RRF transparency)
 *   - rerank_score when present
 *   - chunk_id / doc_id (mono, small) for debugging / future deep-link
 */
export function SourcePanel({ sources, highlightedIndex = null }: SourcePanelProps) {
  if (sources.length === 0) {
    return (
      <p className="text-sm text-fg-muted">
        No sources retrieved yet.
      </p>
    );
  }
  return (
    <ol className="space-y-3">
      {sources.map((src, i) => {
        const highlighted = highlightedIndex === i;
        return (
          <li key={src.chunk_id ?? `${i}-${src.text.slice(0, 16)}`}>
            <Card
              className={
                highlighted
                  ? "border-accent"
                  : "border-border"
              }
            >
              <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                <Badge tone="accent">[{i + 1}]</Badge>
                {src.document_name && (
                  <span dir="auto" className="font-medium text-fg">{src.document_name}</span>
                )}
                {src.page_number != null && (
                  <Badge tone="neutral">p.{src.page_number}</Badge>
                )}
                {src.chunk_index != null && (
                  <span className="text-fg-muted">chunk {src.chunk_index}</span>
                )}
              </div>

              <p dir="auto" className="whitespace-pre-wrap text-sm leading-relaxed text-fg">
                {src.text}
              </p>

              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-fg-muted">
                <Badge tone="neutral" title="Reciprocal Rank Fusion score">
                  RRF {src.hybrid_score.toFixed(4)}
                </Badge>
                {src.vector_rank != null && (
                  <Badge tone="neutral" title="Rank in the dense (vector) modality">
                    vec #{src.vector_rank}
                  </Badge>
                )}
                {src.bm25_rank != null && (
                  <Badge tone="neutral" title="Rank in the BM25 (sparse) modality">
                    bm25 #{src.bm25_rank}
                  </Badge>
                )}
                {src.rerank_score != null && (
                  <Badge tone="accent" title="Cross-encoder rerank score">
                    rerank {src.rerank_score.toFixed(2)}
                  </Badge>
                )}
              </div>

              {src.chunk_id && (
                <p className="mt-2 break-all font-mono text-[10px] text-fg-muted">
                  chunk_id: {src.chunk_id}
                </p>
              )}
            </Card>
          </li>
        );
      })}
    </ol>
  );
}
