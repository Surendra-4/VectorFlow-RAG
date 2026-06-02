import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const mocks = vi.hoisted(() => ({
  listIndexes: vi.fn(),
  switchIndex: vi.fn(),
  activateDefault: vi.fn(),
  deleteIndex: vi.fn(),
  getCompatibility: vi.fn(),
  listRecipes: vi.fn(),
  validateRecipe: vi.fn(),
  createIndex: vi.fn(),
  benchmarkRecipes: vi.fn(),
  getStatus: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const real = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...real,
    ApiError: real.ApiError,
    indexesApi: {
      listIndexes: mocks.listIndexes,
      switchIndex: mocks.switchIndex,
      activateDefault: mocks.activateDefault,
      deleteIndex: mocks.deleteIndex,
      getCompatibility: mocks.getCompatibility,
      listRecipes: mocks.listRecipes,
      validateRecipe: mocks.validateRecipe,
      createIndex: mocks.createIndex,
      benchmarkRecipes: mocks.benchmarkRecipes,
    },
    statusApi: { getStatus: mocks.getStatus },
  };
});

import { ApiError } from "@/lib/api";
import { IndexesTab } from "@/components/settings/IndexesTab";

function idx(over: Record<string, unknown> = {}) {
  return {
    name: "faiss_hnsw", backend: "faiss", index_type: "hnsw",
    embedding_model: "all-MiniLM-L6-v2", embedding_provider: "sentence_transformers",
    vector_dimension: 384, build_params: {}, search_params: {},
    corpus_fingerprint: "fp", chunk_size: 500, chunk_overlap: 50, normalize: true,
    num_vectors: 5, metrics: {}, created_at: 0, last_used: 0, description: "",
    compatibility_info: {}, ...over,
  };
}

beforeEach(() => {
  Object.values(mocks).forEach((m) => m.mockReset());
  mocks.listRecipes.mockResolvedValue({ recipes: [], request_id: "x" });
  mocks.validateRecipe.mockResolvedValue({ validation: { ok: false, errors: [] }, request_id: "x" });
  mocks.listIndexes.mockResolvedValue({ indexes: [idx()], active: "faiss_hnsw", request_id: "x" });
  mocks.getStatus.mockResolvedValue({
    embedder_dimension: 384, chunks_indexed: 5, active_index_name: null,
    app_name: "x", app_version: "0", vector_store_backend: "chromadb",
    embedder_model: "all-MiniLM-L6-v2", reranker_enabled: false, expansion_enabled: false,
    cache_backend: "none", documents_ingested: 1, corpus_fingerprint: "fp",
    uptime_s: 1, request_id: "x",
  });
});

afterEach(() => vi.restoreAllMocks());

describe("IndexesTab (Phase 13)", () => {
  it("shows the default store as live when no named index is active", async () => {
    render(<IndexesTab />);
    expect(await screen.findByText("default index")).toBeInTheDocument();
  });

  it("shows the live index + a Use-default button when one is active", async () => {
    mocks.getStatus.mockResolvedValue({
      embedder_dimension: 384, chunks_indexed: 5, active_index_name: "faiss_hnsw",
      app_name: "x", app_version: "0", vector_store_backend: "faiss",
      embedder_model: "all-MiniLM-L6-v2", reranker_enabled: false, expansion_enabled: false,
      cache_backend: "none", documents_ingested: 1, corpus_fingerprint: "fp",
      uptime_s: 1, request_id: "x",
    });
    render(<IndexesTab />);
    expect(await screen.findByRole("button", { name: "Use default retrieval" })).toBeInTheDocument();
  });

  it("surfaces the compatibility report when a switch returns 409", async () => {
    mocks.switchIndex.mockRejectedValue(
      new ApiError({
        status: 409, code: "index_incompatible", message: "incompatible",
        details: {
          compatibility: {
            index_name: "faiss_hnsw", compatible: false, action: "create_new",
            message: "Index 'faiss_hnsw' is incompatible with the selected configuration. Create a new index?",
            issues: [{ field: "embedding_model", severity: "blocking", message: "model changed" }],
          },
        },
      })
    );
    render(<IndexesTab />);
    const useBtn = await screen.findByRole("button", { name: "Use" });
    fireEvent.click(useBtn);
    expect(await screen.findByText(/Create a new index\?/)).toBeInTheDocument();
    expect(screen.getByText(/model changed/)).toBeInTheDocument();
  });

  it("reverts to default via activateDefault", async () => {
    mocks.getStatus.mockResolvedValueOnce({
      embedder_dimension: 384, chunks_indexed: 5, active_index_name: "faiss_hnsw",
      app_name: "x", app_version: "0", vector_store_backend: "faiss",
      embedder_model: "all-MiniLM-L6-v2", reranker_enabled: false, expansion_enabled: false,
      cache_backend: "none", documents_ingested: 1, corpus_fingerprint: "fp",
      uptime_s: 1, request_id: "x",
    });
    mocks.activateDefault.mockResolvedValue({ index_name: "(default)", action: "activated_default", active: null, request_id: "x" });
    render(<IndexesTab />);
    const btn = await screen.findByRole("button", { name: "Use default retrieval" });
    fireEvent.click(btn);
    await waitFor(() => expect(mocks.activateDefault).toHaveBeenCalled());
  });
});
