"use client";

import { Card, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { StatCard } from "@/components/ui/StatCard";
import { ActivityIcon, SearchIcon, SparkIcon, UploadIcon } from "@/components/ui/icons";
import { metricsApi } from "@/lib/api";
import { usePolling } from "@/lib/hooks/usePolling";
import { formatLatencyMs, formatNumber, formatPercent, formatUptime } from "@/lib/utils/format";

/** Sum every series of a labeled counter into a single total. */
function sumLabeled(
  data: { labeled_counters?: Record<string, Array<{ value: number }>> },
  name: string
): number {
  return (data.labeled_counters?.[name] ?? []).reduce((acc, r) => acc + (r.value ?? 0), 0);
}

function statRow(label: string, value: React.ReactNode) {
  return (
    <div className="flex items-baseline justify-between border-b border-border py-2 last:border-b-0">
      <span className="text-xs text-fg-muted">{label}</span>
      <span className="text-sm font-medium">{value}</span>
    </div>
  );
}

export function MetricsPanel() {
  const { data, error } = usePolling(metricsApi.getMetricsSnapshot, 5000);

  if (error) return <ErrorBox error={error} />;
  if (!data) {
    return <p className="text-sm text-fg-muted">Loading metrics…</p>;
  }

  const retrievalHist = data.histograms?.retrieval_latency_ms;
  const ingestHist = data.histograms?.ingest_latency_ms;
  const streamHist = data.histograms?.stream_duration_ms;

  const retrievals = data.counters?.retrievals_total ?? 0;
  const cacheHits = data.counters?.cache_hits_total ?? 0;
  const hitRatio = retrievals ? cacheHits / retrievals : 0;

  return (
    <div className="space-y-4">
      {/* Headline metrics — animated count-up tiles. */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Retrievals"
          value={retrievals}
          icon={<SearchIcon />}
          hint="hybrid searches served"
        />
        <StatCard
          label="Chunks ingested"
          value={data.counters?.chunks_ingested_total ?? 0}
          icon={<UploadIcon />}
          tone="success"
        />
        <StatCard
          label="Cache hit ratio"
          value={hitRatio * 100}
          format={(n) => `${n.toFixed(0)}%`}
          icon={<SparkIcon />}
          tone="warning"
          hint={`${formatNumber(cacheHits)} hits`}
        />
        <StatCard
          label="Uptime (s)"
          value={Math.floor(data.uptime_s ?? 0)}
          icon={<ActivityIcon />}
          hint={formatUptime(data.uptime_s)}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardTitle>Throughput</CardTitle>
        {statRow("Retrievals", formatNumber(data.counters?.retrievals_total))}
        {statRow("Reranker used", formatNumber(data.counters?.reranker_used_total))}
        {statRow("Cache hits", formatNumber(data.counters?.cache_hits_total))}
        {statRow("Chunks ingested", formatNumber(data.counters?.chunks_ingested_total))}
        {statRow("Ingest failures", formatNumber(data.counters?.ingest_failures_total))}
      </Card>

      <Card>
        <CardTitle>Retrieval latency</CardTitle>
        {statRow("p50", formatLatencyMs(retrievalHist?.p50))}
        {statRow("p95", formatLatencyMs(retrievalHist?.p95))}
        {statRow("p99", formatLatencyMs(retrievalHist?.p99))}
        {statRow("count", formatNumber(retrievalHist?.count))}
      </Card>

      <Card>
        <CardTitle>Ingest latency</CardTitle>
        {statRow("p50", formatLatencyMs(ingestHist?.p50))}
        {statRow("p95", formatLatencyMs(ingestHist?.p95))}
        {statRow("count", formatNumber(ingestHist?.count))}
      </Card>

      <Card>
        <CardTitle>Streams</CardTitle>
        {statRow("Active", formatNumber(data.gauges?.active_streams))}
        {statRow("Sessions total", formatNumber(data.counters?.stream_sessions_total))}
        {statRow("Avg duration", formatLatencyMs(streamHist?.mean))}
      </Card>

      <Card className="md:col-span-2">
        <CardTitle>Requests by endpoint</CardTitle>
        <table className="w-full text-left text-xs">
          <thead className="text-fg-muted">
            <tr>
              <th className="py-1 font-normal">Endpoint</th>
              <th className="py-1 font-normal">Status</th>
              <th className="py-1 text-right font-normal">Count</th>
            </tr>
          </thead>
          <tbody>
            {(data.labeled_counters?.requests_total ?? []).slice(0, 20).map((row, i) => (
              <tr key={i} className="border-t border-border">
                <td className="py-1 font-mono">{String(row.endpoint)}</td>
                <td className="py-1">
                  <Badge tone={Number(row.status_code) >= 400 ? "danger" : "neutral"}>
                    {String(row.status_code)}
                  </Badge>
                </td>
                <td className="py-1 text-right">{formatNumber(row.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card className="md:col-span-2">
        <CardTitle>Platform activity (models &amp; indexes)</CardTitle>
        {statRow("Model switches", formatNumber(sumLabeled(data, "model_switch_total")))}
        {statRow("Chat generations", formatNumber(sumLabeled(data, "provider_chat_total")))}
        {statRow("Provider errors", formatNumber(sumLabeled(data, "provider_errors_total")))}
        {statRow("Model installs", formatNumber(sumLabeled(data, "model_installs_total")))}
        {statRow("Index builds", formatNumber(sumLabeled(data, "index_builds_total")))}
        {statRow("Index switches", formatNumber(data.counters?.index_switch_total))}
        {statRow("Benchmark runs", formatNumber(data.counters?.benchmark_runs_total))}
      </Card>

      <Card className="md:col-span-2">
        <CardTitle>System</CardTitle>
        {statRow("Uptime", formatUptime(data.uptime_s))}
        {statRow(
          "Hit ratio",
          formatPercent(
            data.counters?.cache_hits_total != null && data.counters?.retrievals_total
              ? data.counters.cache_hits_total / data.counters.retrievals_total
              : null
          )
        )}
        {statRow("Recent traces buffer", formatNumber(data.ring_buffer_sizes?.recent_traces))}
        {statRow("Recent errors buffer", formatNumber(data.ring_buffer_sizes?.recent_errors))}
      </Card>
      </div>
    </div>
  );
}
