"use client";

import { useEffect, useState } from "react";
import { statusApi, ApiError } from "@/lib/api";
import { cn } from "@/lib/utils/cn";

type Status = "checking" | "online" | "offline";

/**
 * Polls /health every 20s; renders a status pill the operator can glance at.
 */
export function StatusBadge() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        await statusApi.getHealth();
        if (!cancelled) setStatus("online");
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status >= 200 && e.status < 500) {
          // 4xx → backend is alive, just rejected something — still "online".
          setStatus("online");
        } else {
          setStatus("offline");
        }
      }
    };

    check();
    const t = setInterval(check, 20_000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  const meta = {
    checking: { dot: "bg-fg-muted", label: "Checking…" },
    online: { dot: "bg-success", label: "Backend online" },
    offline: { dot: "bg-danger", label: "Backend offline" },
  }[status];

  return (
    <span
      role="status"
      aria-live="polite"
      className="inline-flex items-center gap-2 rounded-full border border-border/70 bg-surface-raised/50 px-2.5 py-1 text-xs font-medium backdrop-blur"
    >
      <span className="relative flex h-2 w-2">
        {status === "online" && (
          <span className={cn("absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping", meta.dot)} />
        )}
        <span className={cn("relative inline-flex h-2 w-2 rounded-full", meta.dot)} />
      </span>
      {meta.label}
    </span>
  );
}
