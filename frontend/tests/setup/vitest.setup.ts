import "@testing-library/jest-dom/vitest";
import { afterEach, beforeAll } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
});

beforeAll(() => {
  // happy-dom provides AbortController but be defensive for older versions.
  if (typeof globalThis.AbortController === "undefined") {
    // @ts-expect-error: polyfill assignment
    globalThis.AbortController = class {
      signal = { aborted: false };
      abort() {
        this.signal.aborted = true;
      }
    };
  }

  // happy-dom's localStorage shim is sometimes a read-only stub; ensure
  // it's a functional in-memory implementation that tests can drive.
  const memory = new Map<string, string>();
  const storage = {
    getItem: (k: string) => (memory.has(k) ? memory.get(k)! : null),
    setItem: (k: string, v: string) => {
      memory.set(k, String(v));
    },
    removeItem: (k: string) => {
      memory.delete(k);
    },
    clear: () => memory.clear(),
    key: (i: number) => Array.from(memory.keys())[i] ?? null,
    get length() {
      return memory.size;
    },
  };
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: storage,
  });
});
