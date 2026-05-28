"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { statusApi, ApiError } from "@/lib/api";

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

  if (status === "checking") return <Badge tone="neutral">Checking…</Badge>;
  if (status === "online") return <Badge tone="success">Backend online</Badge>;
  return <Badge tone="danger">Backend offline</Badge>;
}
