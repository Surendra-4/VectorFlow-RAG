"use client";

import * as React from "react";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { Input } from "@/components/ui/Input";
import { Spinner } from "@/components/ui/Spinner";
import { Toggle } from "@/components/ui/Toggle";
import { runtimeConfigApi } from "@/lib/api";

/**
 * Runtime retrieval controls — *live* settings that apply to the running
 * pipeline immediately (no rebuild). Index-construction settings live in the
 * Indexes tab because changing them requires a rebuild.
 */
interface LiveState {
  reranker_enabled: boolean;
  expansion_enabled: boolean;
  retrieval_k_default: number;
  retrieval_rrf_k: number;
  retrieval_candidates_per_modality: number;
  cache_enabled: boolean;
}

const NUMERIC_FIELDS: Array<{ key: keyof LiveState; label: string; min: number; max: number }> = [
  { key: "retrieval_k_default", label: "Top-k (results)", min: 1, max: 100 },
  { key: "retrieval_rrf_k", label: "RRF k", min: 1, max: 1000 },
  { key: "retrieval_candidates_per_modality", label: "Candidates / modality", min: 1, max: 200 },
];

export function RetrievalTab() {
  const [live, setLive] = React.useState<Record<string, unknown> | null>(null);
  const [error, setError] = React.useState<Error | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [saving, setSaving] = React.useState(false);
  const [savedHint, setSavedHint] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const cfg = await runtimeConfigApi.getRuntimeConfig();
      setLive(cfg.live);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  const patch = async (body: Record<string, unknown>) => {
    setSaving(true);
    setError(null);
    setSavedHint(null);
    try {
      const res = await runtimeConfigApi.patchLive(body);
      setLive(res.live);
      setSavedHint("Applied to the running pipeline.");
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setSaving(false);
    }
  };

  if (loading && !live) return <Spinner label="Loading retrieval settings…" />;
  const v = (live ?? {}) as unknown as LiveState;

  return (
    <div className="space-y-4">
      <ErrorBox error={error} onRetry={load} />

      <Card>
        <CardTitle>Live retrieval settings</CardTitle>
        <p className="mb-3 text-xs text-fg-muted">
          These apply immediately — no reindex needed.
        </p>

        <div className="space-y-3">
          <ToggleRow
            label="Cross-encoder reranker"
            checked={!!v.reranker_enabled}
            disabled={saving}
            onChange={(c) => patch({ reranker_enabled: c })}
          />
          <ToggleRow
            label="Query expansion"
            checked={!!v.expansion_enabled}
            disabled={saving}
            onChange={(c) => patch({ expansion_enabled: c })}
          />
          <ToggleRow
            label="Retrieval cache"
            checked={!!v.cache_enabled}
            disabled={saving}
            onChange={(c) => patch({ cache_enabled: c })}
          />
        </div>
      </Card>

      <Card>
        <CardTitle>Fusion & candidates</CardTitle>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {NUMERIC_FIELDS.map((f) => (
            <NumberField
              key={f.key}
              label={f.label}
              value={Number(v[f.key] ?? 0)}
              min={f.min}
              max={f.max}
              disabled={saving}
              onCommit={(val) => patch({ [f.key]: val })}
            />
          ))}
        </div>
        {savedHint && <p className="mt-3 text-xs text-success">{savedHint}</p>}
      </Card>
    </div>
  );
}

function ToggleRow({
  label,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  checked: boolean;
  onChange: (c: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm">{label}</span>
      <Toggle checked={checked} onChange={onChange} disabled={disabled} label={label} />
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  onCommit,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onCommit: (v: number) => void;
  disabled?: boolean;
}) {
  const [local, setLocal] = React.useState(String(value));
  React.useEffect(() => setLocal(String(value)), [value]);

  const commit = () => {
    const n = Number(local);
    if (Number.isFinite(n) && n >= min && n <= max && n !== value) onCommit(n);
    else setLocal(String(value));
  };

  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs uppercase tracking-wide text-fg-muted">{label}</span>
      <Input
        type="number"
        min={min}
        max={max}
        value={local}
        disabled={disabled}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
      />
    </label>
  );
}
