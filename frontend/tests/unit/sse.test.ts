import { describe, expect, it } from "vitest";
import { parseEventBlock, parseSseStream } from "@/lib/api/sse";

function bytes(s: string): Uint8Array {
  return new TextEncoder().encode(s);
}

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(bytes(c));
      controller.close();
    },
  });
}

describe("parseEventBlock", () => {
  it("parses event + data", () => {
    expect(parseEventBlock("event: token\ndata: hi")).toEqual({
      event: "token",
      data: "hi",
      id: undefined,
    });
  });

  it("handles multi-line data", () => {
    expect(parseEventBlock("event: msg\ndata: line1\ndata: line2")).toEqual({
      event: "msg",
      data: "line1\nline2",
      id: undefined,
    });
  });

  it("strips a single leading space after the colon", () => {
    expect(parseEventBlock("data: hello")).toEqual({
      event: "",
      data: "hello",
      id: undefined,
    });
    expect(parseEventBlock("data:no-space")).toEqual({
      event: "",
      data: "no-space",
      id: undefined,
    });
  });

  it("ignores comment lines starting with :", () => {
    expect(parseEventBlock(":heartbeat\nevent: ping\ndata: 1")).toEqual({
      event: "ping",
      data: "1",
      id: undefined,
    });
  });

  it("returns null for empty blocks", () => {
    expect(parseEventBlock("")).toBeNull();
    expect(parseEventBlock(":comment-only")).toBeNull();
  });

  it("preserves id field", () => {
    expect(parseEventBlock("id: 42\nevent: x\ndata: y")).toEqual({
      event: "x",
      data: "y",
      id: "42",
    });
  });
});

describe("parseSseStream", () => {
  it("yields events split by \\n\\n", async () => {
    const stream = streamOf([
      "event: sources\ndata: {\"x\":1}\n\nevent: token\ndata: {\"token\":\"hi\"}\n\n",
    ]);
    const out = [];
    for await (const evt of parseSseStream(stream)) out.push(evt);
    expect(out).toEqual([
      { event: "sources", data: '{"x":1}', id: undefined },
      { event: "token", data: '{"token":"hi"}', id: undefined },
    ]);
  });

  it("handles events split across chunks", async () => {
    const stream = streamOf([
      "event: token\n",
      "data: hello",
      "\n\nevent: done\ndata: {}\n\n",
    ]);
    const out = [];
    for await (const evt of parseSseStream(stream)) out.push(evt);
    expect(out).toEqual([
      { event: "token", data: "hello", id: undefined },
      { event: "done", data: "{}", id: undefined },
    ]);
  });

  it("flushes a trailing block without final delimiter", async () => {
    const stream = streamOf(["event: x\ndata: y"]);
    const out = [];
    for await (const evt of parseSseStream(stream)) out.push(evt);
    expect(out).toEqual([{ event: "x", data: "y", id: undefined }]);
  });

  it("ignores stray comment-only blocks", async () => {
    const stream = streamOf([":heartbeat\n\nevent: x\ndata: y\n\n"]);
    const out = [];
    for await (const evt of parseSseStream(stream)) out.push(evt);
    expect(out).toEqual([{ event: "x", data: "y", id: undefined }]);
  });
});
