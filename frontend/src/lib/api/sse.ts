// src/lib/api/sse.ts

/**
 * SSE parser for `fetch`-driven streams.
 *
 * We can't use the browser's `EventSource` because:
 * - it's GET-only (we POST a JSON body)
 * - it doesn't support custom headers cleanly
 *
 * This implementation consumes a `ReadableStream<Uint8Array>` (the response
 * body), splits on the SSE delimiter (`\n\n`), and yields parsed events.
 *
 * Format we accept:
 *   event: token
 *   data: {"token": "hello"}
 *
 * Lines starting with `:` are comments (heartbeats) and are skipped.
 */

export interface ParsedSseEvent {
  event: string;     // SSE event name; "" if not provided
  data: string;      // raw data payload
  id?: string;       // optional event ID
}

const DELIMITER = "\n\n";

export async function* parseSseStream(
  stream: ReadableStream<Uint8Array>
): AsyncGenerator<ParsedSseEvent> {
  const decoder = new TextDecoder("utf-8");
  const reader = stream.getReader();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        // Flush whatever's left.
        if (buffer.trim()) {
          const last = parseEventBlock(buffer);
          if (last) yield last;
        }
        return;
      }
      buffer += decoder.decode(value, { stream: true });

      let delimIdx = buffer.indexOf(DELIMITER);
      while (delimIdx !== -1) {
        const block = buffer.slice(0, delimIdx);
        buffer = buffer.slice(delimIdx + DELIMITER.length);
        const parsed = parseEventBlock(block);
        if (parsed) yield parsed;
        delimIdx = buffer.indexOf(DELIMITER);
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // already released
    }
  }
}

export function parseEventBlock(block: string): ParsedSseEvent | null {
  let event = "";
  let dataLines: string[] = [];
  let id: string | undefined;

  for (const rawLine of block.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (!line || line.startsWith(":")) continue;

    const colonIdx = line.indexOf(":");
    const field = colonIdx === -1 ? line : line.slice(0, colonIdx);
    let value = colonIdx === -1 ? "" : line.slice(colonIdx + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
    else if (field === "id") id = value;
  }

  if (dataLines.length === 0 && !event) return null;
  return { event, data: dataLines.join("\n"), id };
}
