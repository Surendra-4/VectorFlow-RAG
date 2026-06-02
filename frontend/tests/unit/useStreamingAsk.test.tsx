import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

// `vi.hoisted` ensures this is created BEFORE the vi.mock factory runs
// (mocks are hoisted to the top of the file before any other imports).
const mocks = vi.hoisted(() => ({
  streamAsk: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const real = await vi.importActual<object>("@/lib/api");
  return {
    ...real,
    askApi: { streamAsk: mocks.streamAsk },
  };
});

import { useStreamingAsk } from "@/lib/hooks/useStreamingAsk";

// Helper: produce an async generator yielding a scripted event list.
async function* scriptedEvents(events: unknown[]) {
  for (const evt of events) {
    yield evt;
  }
}

beforeEach(() => {
  mocks.streamAsk.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useStreamingAsk", () => {
  it("starts in idle and exposes empty defaults", () => {
    const { result } = renderHook(() => useStreamingAsk());
    expect(result.current.state).toBe("idle");
    expect(result.current.answer).toBe("");
    expect(result.current.sources).toEqual([]);
    expect(result.current.metrics).toBeNull();
  });

  it("accumulates tokens and finalizes on done", async () => {
    mocks.streamAsk.mockReturnValue(
      scriptedEvents([
        { type: "sources", request_id: "x", sources: [{ text: "src1" }] },
        { type: "token", token: "Hel" },
        { type: "token", token: "lo " },
        { type: "token", token: "world" },
        {
          type: "done",
          request_id: "x",
          answer: "Hello world",
          metrics: {
            retrieval_time_ms: 10,
            generation_time_ms: 50,
            total_time_ms: 60,
            num_context_docs: 1,
          },
        },
      ])
    );

    const { result } = renderHook(() => useStreamingAsk());

    await act(async () => {
      await result.current.start({ query: "hi" });
    });

    await waitFor(() => {
      expect(result.current.state).toBe("done");
    });
    expect(result.current.answer).toBe("Hello world");
    expect(result.current.sources).toHaveLength(1);
    expect(result.current.metrics?.total_time_ms).toBe(60);
  });

  it("surfaces error events", async () => {
    mocks.streamAsk.mockReturnValue(
      scriptedEvents([
        { type: "sources", request_id: "x", sources: [] },
        { type: "error", message: "LLM exploded" },
      ])
    );

    const { result } = renderHook(() => useStreamingAsk());

    await act(async () => {
      await result.current.start({ query: "boom" });
    });

    await waitFor(() => {
      expect(result.current.state).toBe("error");
    });
    expect(result.current.errorMessage).toBe("LLM exploded");
  });

  it("keeps error state when a done event follows an error (LLM down)", async () => {
    // The server emits sources → error → done when retrieval succeeds but the
    // LLM is unreachable. `done` must not clobber the error, or the failure is
    // silently swallowed (empty answer, no explanation).
    mocks.streamAsk.mockReturnValue(
      scriptedEvents([
        { type: "sources", request_id: "x", sources: [{ text: "ctx", chunk_id: "c0" }] },
        { type: "error", message: "[Error communicating with Ollama: port 11434 refused]" },
        {
          type: "done",
          request_id: "x",
          answer: "",
          metrics: {
            retrieval_time_ms: 12, generation_time_ms: 0, total_time_ms: 12,
            num_context_docs: 1,
          },
        },
      ])
    );

    const { result } = renderHook(() => useStreamingAsk());
    await act(async () => {
      await result.current.start({ query: "q" });
    });

    await waitFor(() => {
      expect(result.current.state).toBe("error");
    });
    // Error preserved AND sources still available (retrieval succeeded).
    expect(result.current.errorMessage).toContain("11434");
    expect(result.current.sources).toHaveLength(1);
  });

  it("reset clears all state back to idle defaults", async () => {
    mocks.streamAsk.mockReturnValue(
      scriptedEvents([
        { type: "token", token: "x" },
        {
          type: "done",
          answer: "x",
          metrics: {
            retrieval_time_ms: 1,
            generation_time_ms: 1,
            total_time_ms: 2,
            num_context_docs: 0,
          },
          request_id: "1",
        },
      ])
    );
    const { result } = renderHook(() => useStreamingAsk());
    await act(async () => {
      await result.current.start({ query: "q" });
    });
    expect(result.current.answer).toBe("x");

    act(() => {
      result.current.reset();
    });

    expect(result.current.state).toBe("idle");
    expect(result.current.answer).toBe("");
    expect(result.current.metrics).toBeNull();
  });
});
