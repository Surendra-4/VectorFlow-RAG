import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageBubble } from "@/components/chat/MessageBubble";
import { SourcePanel } from "@/components/citations/SourcePanel";
import type { RetrievalResult } from "@/lib/api";

/**
 * Phase 11: verify user-content containers carry dir="auto" so RTL scripts
 * (Arabic/Hebrew) render with correct direction. We don't assert visual
 * direction (jsdom/happy-dom don't lay out bidi) — only that the hint is
 * present, which is what delegates direction resolution to the browser.
 */

function result(over: Partial<RetrievalResult> = {}): RetrievalResult {
  return {
    text: "مرحبا بالعالم — Arabic sample text",
    chunk_id: "ar:0:abcd1234",
    doc_id: "ar",
    document_name: "وثيقة.pdf",
    source_path: "/tmp/doc.pdf",
    page_number: 1,
    chunk_index: 0,
    hybrid_score: 0.05,
    rrf_score: 0.05,
    vector_rank: 1,
    bm25_rank: null,
    rerank_score: null,
    metadata: {},
    ...over,
  };
}

describe("Bidirectional rendering", () => {
  it("MessageBubble sets dir=auto on its content container", () => {
    const { container } = render(
      <MessageBubble role="assistant">مرحبا</MessageBubble>
    );
    const bubble = container.querySelector('[data-role="assistant"]');
    expect(bubble).toHaveAttribute("dir", "auto");
  });

  it("SourcePanel chunk text carries dir=auto", () => {
    const { container } = render(<SourcePanel sources={[result()]} />);
    // The chunk-text paragraph and the document-name span both opt into auto.
    const autoEls = container.querySelectorAll('[dir="auto"]');
    expect(autoEls.length).toBeGreaterThanOrEqual(2);
  });

  it("renders RTL document names without error", () => {
    render(<SourcePanel sources={[result()]} />);
    expect(screen.getByText("وثيقة.pdf")).toBeInTheDocument();
  });

  it("renders mixed-script chunk text", () => {
    render(<SourcePanel sources={[result()]} />);
    expect(screen.getByText(/Arabic sample text/)).toBeInTheDocument();
  });
});
