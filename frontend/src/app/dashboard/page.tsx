import { MetricsPanel } from "@/components/dashboard/MetricsPanel";
import { PageHeader } from "@/components/layout/PageHeader";
import { Reveal } from "@/components/ui/motion";
import { ActivityIcon } from "@/components/ui/icons";

export const metadata = { title: "Dashboard" };

export default function DashboardPage() {
  return (
    <section aria-label="Dashboard" className="space-y-8">
      <PageHeader
        eyebrow="Observability"
        title="Live"
        highlight="metrics"
        icon={<ActivityIcon />}
        description="Process metrics in real time — throughput, latency percentiles, cache, streams, and platform activity. Polls every 5 seconds; pauses when the tab is hidden."
      />
      <Reveal>
        <MetricsPanel />
      </Reveal>
    </section>
  );
}
