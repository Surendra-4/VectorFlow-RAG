import { MetricsPanel } from "@/components/dashboard/MetricsPanel";
import { UserStatsPanel } from "@/components/dashboard/UserStatsPanel";
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
        description="Your personal activity plus live process metrics — throughput, latency percentiles, cache, streams, and platform activity."
      />
      <Reveal>
        <UserStatsPanel />
      </Reveal>
      <Reveal delay={80}>
        <MetricsPanel />
      </Reveal>
    </section>
  );
}
