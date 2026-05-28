"use client";

import { Badge } from "@/components/ui/Badge";
import type { AskMetrics } from "@/lib/api";
import { formatLatencyMs } from "@/lib/utils/format";

export function LatencyChip({ metrics }: { metrics: AskMetrics | null }) {
  if (!metrics) return null;
  return (
    <div className="flex flex-wrap items-center gap-1 text-xs">
      <Badge tone="neutral" title="Total wall-clock from request to last token">
        total {formatLatencyMs(metrics.total_time_ms)}
      </Badge>
      <Badge tone="neutral" title="Retrieval (hybrid + optional expansion + optional rerank)">
        retrieve {formatLatencyMs(metrics.retrieval_time_ms)}
      </Badge>
      <Badge tone="neutral" title="LLM generation only">
        gen {formatLatencyMs(metrics.generation_time_ms)}
      </Badge>
      <Badge tone="neutral">
        {metrics.num_context_docs} ctx
      </Badge>
    </div>
  );
}
