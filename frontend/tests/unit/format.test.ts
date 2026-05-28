import { describe, expect, it } from "vitest";
import {
  formatBytes,
  formatLatencyMs,
  formatNumber,
  formatPercent,
  formatUptime,
  truncate,
} from "@/lib/utils/format";

describe("formatLatencyMs", () => {
  it("dash for null/undefined", () => {
    expect(formatLatencyMs(null)).toBe("—");
    expect(formatLatencyMs(undefined)).toBe("—");
  });
  it("<1 ms shortcut", () => {
    expect(formatLatencyMs(0.5)).toBe("<1 ms");
  });
  it("rounds ms under 1s", () => {
    expect(formatLatencyMs(123.4)).toBe("123 ms");
  });
  it("renders seconds with one decimal", () => {
    expect(formatLatencyMs(1234)).toBe("1.2 s");
  });
  it("renders minute+second over 60s", () => {
    expect(formatLatencyMs(65_000)).toBe("1m 5s");
  });
});

describe("formatNumber", () => {
  it("uses locale formatting", () => {
    expect(formatNumber(1234567)).toBe((1234567).toLocaleString());
  });
  it("returns dash for null", () => {
    expect(formatNumber(null)).toBe("—");
  });
});

describe("formatPercent", () => {
  it("formats fraction as percent", () => {
    expect(formatPercent(0.4567)).toBe("45.7%");
  });
});

describe("truncate", () => {
  it("returns the input when under the limit", () => {
    expect(truncate("hello", 10)).toBe("hello");
  });
  it("truncates with ellipsis", () => {
    expect(truncate("hello world", 8)).toBe("hello w…");
  });
});

describe("formatBytes", () => {
  it("returns bytes for small values", () => {
    expect(formatBytes(500)).toBe("500 B");
  });
  it("returns KB for medium values", () => {
    // Sub-10 values keep one decimal for finer readability.
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(20 * 1024)).toBe("20 KB");
  });
  it("returns MB for large values", () => {
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatBytes(15 * 1024 * 1024)).toBe("15 MB");
  });
});

describe("formatUptime", () => {
  it("under a minute", () => {
    expect(formatUptime(42)).toBe("42s");
  });
  it("includes minutes", () => {
    expect(formatUptime(125)).toBe("2m 5s");
  });
  it("includes hours when applicable", () => {
    expect(formatUptime(3661)).toBe("1h 1m 1s");
  });
});
