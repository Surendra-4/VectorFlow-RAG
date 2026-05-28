import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SourcePanel } from "@/components/citations/SourcePanel";
import type { RetrievalResult } from "@/lib/api";

function mockResult(over: Partial<RetrievalResult> = {}): RetrievalResult {
  return {
    text: "Photosynthesis converts light energy into chemical energy.",
    chunk_id: "doc_a:0:abcd1234",
    doc_id: "doc_a",
    document_name: "biology.pdf",
    source_path: "/tmp/biology.pdf",
    page_number: 3,
    chunk_index: 0,
    hybrid_score: 0.0421,
    rrf_score: 0.0421,
    vector_rank: 1,
    bm25_rank: 2,
    rerank_score: null,
    metadata: {},
    ...over,
  };
}

describe("SourcePanel", () => {
  it("renders an empty-state message when no sources", () => {
    render(<SourcePanel sources={[]} />);
    expect(screen.getByText(/No sources retrieved/i)).toBeInTheDocument();
  });

  it("renders document name, page, and chunk text", () => {
    render(<SourcePanel sources={[mockResult()]} />);
    expect(screen.getByText("biology.pdf")).toBeInTheDocument();
    expect(screen.getByText("p.3")).toBeInTheDocument();
    expect(screen.getByText(/Photosynthesis converts/i)).toBeInTheDocument();
  });

  it("renders RRF score and per-modality ranks", () => {
    render(<SourcePanel sources={[mockResult()]} />);
    expect(screen.getByText(/RRF\s+0\.0421/)).toBeInTheDocument();
    expect(screen.getByText(/vec\s+#1/)).toBeInTheDocument();
    expect(screen.getByText(/bm25\s+#2/)).toBeInTheDocument();
  });

  it("renders rerank score when present", () => {
    render(<SourcePanel sources={[mockResult({ rerank_score: 7.42 })]} />);
    expect(screen.getByText(/rerank\s+7\.42/)).toBeInTheDocument();
  });

  it("renders citation index for each source", () => {
    render(
      <SourcePanel
        sources={[
          mockResult(),
          mockResult({ chunk_id: "doc_b:0:beef0001", document_name: "x.pdf" }),
        ]}
      />
    );
    expect(screen.getByText("[1]")).toBeInTheDocument();
    expect(screen.getByText("[2]")).toBeInTheDocument();
  });

  it("highlights the indexed source when highlightedIndex matches", () => {
    const { container } = render(
      <SourcePanel sources={[mockResult(), mockResult()]} highlightedIndex={1} />
    );
    const cards = container.querySelectorAll("li > div");
    expect(cards.length).toBe(2);
    expect(cards[1]?.className).toMatch(/border-accent/);
    expect(cards[0]?.className).not.toMatch(/border-accent/);
  });

  it("displays chunk_id at the bottom for debugging", () => {
    render(<SourcePanel sources={[mockResult()]} />);
    expect(screen.getByText(/doc_a:0:abcd1234/)).toBeInTheDocument();
  });
});
