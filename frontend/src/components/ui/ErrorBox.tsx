"use client";

import * as React from "react";
import { ApiError } from "@/lib/api";
import { Button } from "./Button";

interface ErrorBoxProps {
  error: ApiError | Error | null;
  onRetry?: () => void;
  className?: string;
}

export function ErrorBox({ error, onRetry, className }: ErrorBoxProps) {
  if (!error) return null;
  const isApi = error instanceof ApiError;
  return (
    <div
      role="alert"
      className={`rounded border border-danger/40 bg-danger/10 p-3 text-sm text-danger ${className ?? ""}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <strong className="font-semibold">
            {isApi ? `Error · ${(error as ApiError).code}` : "Error"}
          </strong>
          <p className="mt-1 text-fg-muted">{error.message}</p>
          {isApi && (error as ApiError).requestId && (
            <p className="mt-2 text-xs text-fg-muted">
              request_id: <code>{(error as ApiError).requestId}</code>
            </p>
          )}
        </div>
        {onRetry && (
          <Button size="sm" variant="secondary" onClick={onRetry}>
            Retry
          </Button>
        )}
      </div>
    </div>
  );
}
