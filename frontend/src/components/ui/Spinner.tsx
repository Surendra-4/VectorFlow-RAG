"use client";

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <span
      role="status"
      aria-live="polite"
      className="inline-flex items-center gap-2 text-sm text-fg-muted"
    >
      <span
        aria-hidden="true"
        className="h-3 w-3 animate-spin rounded-full border-2 border-current border-r-transparent"
      />
      {label}
    </span>
  );
}
