import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { apiFetch, resolveBaseUrl } from "@/lib/api/client";
import { ApiError } from "@/lib/api/errors";

describe("resolveBaseUrl", () => {
  beforeEach(() => {
    try {
      window.localStorage.clear();
    } catch {
      /* ignore */
    }
  });

  it("falls back to default when no overrides", () => {
    expect(resolveBaseUrl()).toBe("http://localhost:8000");
  });

  it("uses localStorage override when present", () => {
    window.localStorage.setItem("vfr_api_base_url", "https://my-backend.example.com/");
    expect(resolveBaseUrl()).toBe("https://my-backend.example.com");
  });
});

describe("apiFetch", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function mockResponse(opts: {
    status?: number;
    body?: unknown;
    headers?: Record<string, string>;
  }) {
    const status = opts.status ?? 200;
    const body = opts.body === undefined ? undefined : JSON.stringify(opts.body);
    return new Response(body, {
      status,
      headers: { "Content-Type": "application/json", ...(opts.headers ?? {}) },
    });
  }

  it("returns parsed JSON on success", async () => {
    fetchMock.mockResolvedValue(mockResponse({ body: { answer: "hi" } }));
    const res = await apiFetch<{ answer: string }>("/api/v1/x");
    expect(res).toEqual({ answer: "hi" });
  });

  it("constructs URL from base + path", async () => {
    fetchMock.mockResolvedValue(mockResponse({ body: {} }));
    await apiFetch("/api/v1/x", { baseUrl: "http://example.test" });
    const [calledUrl] = fetchMock.mock.calls[0];
    expect(calledUrl).toBe("http://example.test/api/v1/x");
  });

  it("JSON-encodes body and sets Content-Type", async () => {
    fetchMock.mockResolvedValue(mockResponse({ body: {} }));
    await apiFetch("/api/v1/x", { method: "POST", body: { q: "hello" } });
    const init = fetchMock.mock.calls[0][1];
    expect(init.body).toBe(JSON.stringify({ q: "hello" }));
    expect(init.headers["Content-Type"]).toBe("application/json");
  });

  it("does not JSON-encode FormData bodies", async () => {
    fetchMock.mockResolvedValue(mockResponse({ body: {} }));
    const fd = new FormData();
    fd.append("k", "v");
    await apiFetch("/api/v1/x", { method: "POST", body: fd, formData: true });
    const init = fetchMock.mock.calls[0][1];
    expect(init.body).toBe(fd);
    // Browser sets multipart boundary; we must NOT set Content-Type manually.
    expect(init.headers["Content-Type"]).toBeUndefined();
  });

  it("throws structured ApiError on non-OK response", async () => {
    fetchMock.mockResolvedValue(
      mockResponse({
        status: 422,
        body: { code: "validation_error", message: "bad input", details: { field: "x" } },
        headers: { "X-Request-ID": "req-123" },
      })
    );
    await expect(apiFetch("/api/v1/x")).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      code: "validation_error",
      message: "bad input",
      requestId: "req-123",
    });
  });

  it("falls back to http_<status> when the body has no code", async () => {
    fetchMock.mockResolvedValue(new Response("", { status: 500 }));
    await expect(apiFetch("/api/v1/x")).rejects.toMatchObject({
      status: 500,
      code: "http_500",
    });
  });

  it("translates network failures into ApiError(network_error)", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(apiFetch("/api/v1/x")).rejects.toMatchObject({
      status: 0,
      code: "network_error",
    });
  });

  it("returns the raw Response when rawResponse=true", async () => {
    fetchMock.mockResolvedValue(mockResponse({ body: { ok: true } }));
    const res = await apiFetch<Response>("/api/v1/x", { rawResponse: true });
    expect(res).toBeInstanceOf(Response);
    expect(await res.json()).toEqual({ ok: true });
  });

  it("returns undefined for 204 No Content", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
    const res = await apiFetch<void>("/api/v1/x");
    expect(res).toBeUndefined();
  });
});

describe("ApiError", () => {
  it("is an Error subclass", () => {
    const e = new ApiError({ status: 400, message: "no" });
    expect(e instanceof Error).toBe(true);
    expect(e instanceof ApiError).toBe(true);
  });

  it("defaults code/requestId/details when omitted", () => {
    const e = new ApiError({ status: 500, message: "boom" });
    expect(e.code).toBe("unknown");
    expect(e.requestId).toBeNull();
    expect(e.details).toBeNull();
  });
});
