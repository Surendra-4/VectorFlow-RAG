import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MetricsPanel } from "@/components/dashboard/MetricsPanel";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  vi.restoreAllMocks();
});

function snapshotResponse(payload: object) {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("MetricsPanel", () => {
  it("renders loading state initially", () => {
    fetchMock.mockReturnValue(new Promise(() => {}));
    render(<MetricsPanel />);
    expect(screen.getByText(/Loading metrics/i)).toBeInTheDocument();
  });

  it("renders counters and histograms when data arrives", async () => {
    fetchMock.mockResolvedValue(
      snapshotResponse({
        uptime_s: 12.5,
        counters: {
          retrievals_total: 42,
          reranker_used_total: 5,
          cache_hits_total: 10,
          chunks_ingested_total: 100,
          ingest_failures_total: 0,
          stream_sessions_total: 3,
        },
        gauges: { active_streams: 1 },
        labeled_counters: {
          requests_total: [
            { endpoint: "/api/v1/search", status_code: "200", value: 42 },
            { endpoint: "/api/v1/ask", status_code: "200", value: 3 },
          ],
          expansion_strategy_usage_total: [],
          cache_ops_total: [],
          ingestions_total: [],
        },
        histograms: {
          retrieval_latency_ms: { count: 42, p50: 12.3, p95: 45.6, p99: 88, min: 2, max: 99, mean: 20 },
          ingest_latency_ms: { count: 1, p50: 1500, p95: 1500, p99: 1500, min: 1500, max: 1500, mean: 1500 },
          stream_duration_ms: { count: 3, p50: 2000, p95: 4000, p99: 4000, min: 1000, max: 4000, mean: 2333 },
        },
        labeled_histograms: { request_latency_ms: [], retrieval_stage_latency_ms: [] },
        ring_buffer_sizes: { recent_traces: 42, recent_errors: 0 },
        request_id: "test-req",
      })
    );

    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/Throughput/i)).toBeInTheDocument();
    });

    // Latency card values are uniquely formatted.
    expect(screen.getByText("12 ms")).toBeInTheDocument(); // p50
    expect(screen.getByText("46 ms")).toBeInTheDocument(); // p95 (rounded)
    // Endpoint table renders a row for /api/v1/search.
    expect(screen.getByText("/api/v1/search")).toBeInTheDocument();
    // The retrievals_total counter (42) appears at least once.
    expect(screen.getAllByText(/\b42\b/).length).toBeGreaterThan(0);
  });

  it("renders an error box on failed fetch", async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({ code: "server_error", message: "boom" }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      )
    );
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
  });
});
