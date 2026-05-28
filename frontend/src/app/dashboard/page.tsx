import { MetricsPanel } from "@/components/dashboard/MetricsPanel";

export default function DashboardPage() {
  return (
    <section aria-label="Dashboard" className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-sm text-fg-muted">
          Live process metrics. Polls every 5 seconds; pauses when the tab is hidden.
        </p>
      </header>
      <MetricsPanel />
    </section>
  );
}
