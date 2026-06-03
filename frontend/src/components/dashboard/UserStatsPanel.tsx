"use client";

import * as React from "react";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { StatCard } from "@/components/ui/StatCard";
import { ActivityIcon, ChatIcon, DocsIcon, SearchIcon } from "@/components/ui/icons";
import { authApi, type UserStats } from "@/lib/api";
import { useAuth } from "@/lib/auth/AuthContext";
import { formatNumber } from "@/lib/utils/format";

/**
 * Per-user activity (from the database) with a "reset my statistics" action —
 * distinct from the process-wide metrics below it. Only the signed-in user's
 * own counters; resetting affects nobody else.
 */
export function UserStatsPanel() {
  const { user } = useAuth();
  const [stats, setStats] = React.useState<UserStats | null>(null);
  const [error, setError] = React.useState<Error | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [confirming, setConfirming] = React.useState(false);

  const load = React.useCallback(async () => {
    setError(null);
    try {
      const res = await authApi.getMyStats();
      setStats(res.stats);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  const reset = async () => {
    setBusy(true);
    try {
      const res = await authApi.resetMyStats();
      setStats(res.stats);
      setConfirming(false);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card variant="gradient" className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <CardTitle className="mb-0.5">Your activity</CardTitle>
          <p className="text-xs text-fg-muted">
            {user?.email} · {stats?.reset_at ? `last reset ${new Date(stats.reset_at).toLocaleDateString()}` : "since you joined"}
          </p>
        </div>
        {confirming ? (
          <div className="flex items-center gap-2">
            <span className="text-xs text-fg-muted">Reset all your stats?</span>
            <Button size="sm" variant="danger" loading={busy} onClick={reset}>Confirm</Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirming(false)}>Cancel</Button>
          </div>
        ) : (
          <Button size="sm" variant="secondary" onClick={() => setConfirming(true)}>
            Reset my statistics
          </Button>
        )}
      </div>

      <ErrorBox error={error} onRetry={load} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Searches" value={stats?.searches ?? 0} icon={<SearchIcon />} />
        <StatCard label="Questions" value={stats?.asks ?? 0} icon={<ChatIcon />} tone="success" />
        <StatCard label="Documents" value={stats?.documents_ingested ?? 0} icon={<DocsIcon />} tone="warning" />
        <StatCard label="Chunks" value={stats?.chunks_ingested ?? 0} icon={<ActivityIcon />} hint={`${formatNumber(stats?.retrievals ?? 0)} retrievals`} />
      </div>
    </Card>
  );
}
