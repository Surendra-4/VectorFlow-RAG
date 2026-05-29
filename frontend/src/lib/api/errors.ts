// src/lib/api/errors.ts

/**
 * Structured ApiError raised by the fetch wrapper.
 *
 * Always carries:
 * - `status`  — HTTP status (or 0 for network failures)
 * - `code`    — backend's structured `code` (e.g. "bad_request") when present
 * - `message` — human-friendly text
 * - `requestId` — propagated from the X-Request-ID header for correlation
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;
  readonly details: Record<string, unknown> | null;

  constructor(args: {
    status: number;
    code?: string;
    message: string;
    requestId?: string | null;
    details?: Record<string, unknown> | null;
  }) {
    super(args.message);
    this.name = "ApiError";
    this.status = args.status;
    this.code = args.code ?? "unknown";
    this.requestId = args.requestId ?? null;
    this.details = args.details ?? null;
  }
}
