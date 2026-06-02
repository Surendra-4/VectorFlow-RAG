import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";

const mocks = vi.hoisted(() => ({
  listProviders: vi.fn(),
  getActiveModel: vi.fn(),
  listInstalled: vi.fn(),
  listCatalog: vi.fn(),
  listOnlineModels: vi.fn(),
  selectModel: vi.fn(),
  setApiKey: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const real = await vi.importActual<object>("@/lib/api");
  return {
    ...real,
    modelsApi: {
      listProviders: mocks.listProviders,
      getActiveModel: mocks.getActiveModel,
      listInstalled: mocks.listInstalled,
      listCatalog: mocks.listCatalog,
      listOnlineModels: mocks.listOnlineModels,
      selectModel: mocks.selectModel,
      setApiKey: mocks.setApiKey,
      deleteModel: vi.fn(),
      installModel: vi.fn(),
      deleteApiKey: vi.fn(),
      validateConnection: vi.fn(),
      getApiKeyStatus: vi.fn(),
    },
  };
});

import { ModelsTab } from "@/components/settings/ModelsTab";

const PROVIDERS = {
  providers: [
    {
      name: "ollama", label: "Ollama (local)", location: "offline",
      requires_api_key: false, supports_chat: true, supports_streaming: true,
      supports_model_listing: true, supports_install: true, supports_embeddings: true,
      base_url_configurable: true, default_base_url: "http://localhost:11434",
      docs_url: null, notes: "Runs offline.", key_configured: false, key_hint: null,
    },
    {
      name: "openai", label: "OpenAI", location: "online",
      requires_api_key: true, supports_chat: true, supports_streaming: true,
      supports_model_listing: true, supports_install: false, supports_embeddings: true,
      base_url_configurable: true, default_base_url: "https://api.openai.com/v1",
      docs_url: null, notes: "", key_configured: false, key_hint: null,
    },
  ],
  request_id: "x",
};

beforeEach(() => {
  Object.values(mocks).forEach((m) => m.mockReset());
  mocks.listProviders.mockResolvedValue(PROVIDERS);
  mocks.getActiveModel.mockResolvedValue({
    provider: "ollama", model: "tinyllama", location: "offline", request_id: "x",
  });
  mocks.listInstalled.mockResolvedValue({
    provider: "ollama",
    models: [
      { id: "tinyllama", kind: "chat", supports_streaming: true, supports_tools: false,
        multilingual: false, installed: true, parameter_size: "1.1B", size_bytes: 600000000 },
    ],
    request_id: "x",
  });
  mocks.listCatalog.mockResolvedValue({
    provider: "ollama",
    models: [
      { id: "llama3.2:1b", kind: "chat", supports_streaming: true, supports_tools: false,
        multilingual: true, installed: false, description: "Fast small model",
        ram_estimate_bytes: 2000000000 },
    ],
    request_id: "x",
  });
  mocks.listOnlineModels.mockResolvedValue({
    provider: "openai",
    models: [{ id: "gpt-4o-mini", kind: "chat", supports_streaming: true,
               supports_tools: true, multilingual: true, context_window: 128000 }],
    request_id: "x",
  });
});

afterEach(() => vi.restoreAllMocks());

describe("ModelsTab", () => {
  it("shows the active model and the local provider's installed models", async () => {
    render(<ModelsTab />);
    expect(await screen.findByText("Active chat model")).toBeInTheDocument();
    // installed model visible (appears in both the active card and the list)
    expect((await screen.findAllByText("tinyllama")).length).toBeGreaterThan(0);
    // downloadable model visible
    expect(await screen.findByText("llama3.2:1b")).toBeInTheDocument();
  });

  it("activates a local model via selectModel", async () => {
    mocks.selectModel.mockResolvedValue({ provider: "ollama", model: "tinyllama", request_id: "x" });
    render(<ModelsTab />);
    const useBtn = await screen.findByRole("button", { name: "Use" });
    fireEvent.click(useBtn);
    await waitFor(() =>
      expect(mocks.selectModel).toHaveBeenCalledWith({ provider: "ollama", model: "tinyllama" })
    );
  });

  it("switches to an online provider and shows the API-key field", async () => {
    render(<ModelsTab />);
    const select = await screen.findByLabelText("Provider");
    fireEvent.change(select, { target: { value: "openai" } });
    expect(await screen.findByText("Online models (OpenAI)")).toBeInTheDocument();
    // Key not configured → password input present.
    expect(await screen.findByPlaceholderText("OpenAI API key")).toBeInTheDocument();
  });
});
