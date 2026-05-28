"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/Button";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[global error boundary]", error);
  }, [error]);

  return (
    <div className="mx-auto max-w-xl space-y-4 py-12 text-center">
      <h1 className="text-xl font-semibold">Something went wrong</h1>
      <p className="text-sm text-fg-muted">{error.message}</p>
      <Button onClick={reset} variant="secondary" size="sm">
        Try again
      </Button>
    </div>
  );
}
