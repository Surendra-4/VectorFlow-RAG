"use client";

import * as React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { Input } from "@/components/ui/Input";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { Select } from "@/components/ui/Select";
import { Spinner } from "@/components/ui/Spinner";
import { modelsApi } from "@/lib/api";
import type {
  ActiveModelResponse,
  ProviderCapabilities,
  ProviderModel,
} from "@/lib/api/types";
import { formatBytes } from "@/lib/utils/format";

/**
 * Model management: pick any local (Ollama) open-source model, or a
 * closed-source model through a provider API key — exactly the choice the
 * brief calls for. The frontend never sees provider internals or raw keys.
 */
export function ModelsTab() {
  const [providers, setProviders] = React.useState<ProviderCapabilities[] | null>(null);
  const [active, setActive] = React.useState<ActiveModelResponse | null>(null);
  const [providerName, setProviderName] = React.useState<string>("ollama");
  const [error, setError] = React.useState<Error | null>(null);
  const [loading, setLoading] = React.useState(true);

  const load = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [p, a] = await Promise.all([
        modelsApi.listProviders(),
        modelsApi.getActiveModel().catch(() => null),
      ]);
      setProviders(p.providers);
      if (a) {
        setActive(a);
        setProviderName(a.provider);
      }
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  if (loading && !providers) return <Spinner label="Loading providers…" />;

  const current = providers?.find((p) => p.name === providerName) ?? null;

  return (
    <div className="space-y-4">
      <ErrorBox error={error} onRetry={load} />

      {active && (
        <Card>
          <CardTitle>Active chat model</CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="accent">{active.provider}</Badge>
            <span className="font-mono text-sm">{active.model}</span>
            {active.location && <Badge tone="neutral">{active.location}</Badge>}
          </div>
        </Card>
      )}

      <Card>
        <CardTitle>Provider</CardTitle>
        <Select
          aria-label="Provider"
          value={providerName}
          onChange={(e) => setProviderName(e.target.value)}
          options={(providers ?? []).map((p) => ({
            value: p.name,
            label: `${p.label}${p.location === "offline" ? " · local" : " · API"}`,
          }))}
          className="max-w-sm"
        />
        {current?.notes && <p className="mt-2 text-xs text-fg-muted">{current.notes}</p>}
      </Card>

      {current?.location === "offline" ? (
        <OfflineModels provider={current} onActivated={load} />
      ) : current ? (
        <OnlineModels provider={current} onActivated={load} onKeyChange={load} />
      ) : null}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Offline (Ollama)
// --------------------------------------------------------------------------- //

function OfflineModels({
  provider,
  onActivated,
}: {
  provider: ProviderCapabilities;
  onActivated: () => void;
}) {
  const [installed, setInstalled] = React.useState<ProviderModel[]>([]);
  const [catalog, setCatalog] = React.useState<ProviderModel[]>([]);
  const [error, setError] = React.useState<Error | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [installName, setInstallName] = React.useState<string | null>(null);
  const [installPct, setInstallPct] = React.useState(0);
  const [installMsg, setInstallMsg] = React.useState("");

  const load = React.useCallback(async () => {
    setError(null);
    try {
      const [inst, cat] = await Promise.all([
        modelsApi.listInstalled(provider.name).catch(() => ({ models: [] as ProviderModel[] })),
        modelsApi.listCatalog(provider.name).catch(() => ({ models: [] as ProviderModel[] })),
      ]);
      setInstalled(inst.models);
      setCatalog(cat.models);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, [provider.name]);

  React.useEffect(() => {
    void load();
  }, [load]);

  const install = async (name: string) => {
    setInstallName(name);
    setInstallPct(0);
    setInstallMsg("Starting…");
    setError(null);
    try {
      await modelsApi.installModel(name, (event, data) => {
        if (event === "progress") {
          if (typeof data.percent === "number") setInstallPct(data.percent);
          if (typeof data.status === "string") setInstallMsg(data.status);
        } else if (event === "error") {
          setError(new Error(String(data.message ?? "install failed")));
        }
      }, { provider: provider.name });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setInstallName(null);
    }
  };

  const remove = async (name: string) => {
    setBusy(true);
    try {
      await modelsApi.deleteModel(name, provider.name);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  const activate = async (name: string) => {
    setBusy(true);
    try {
      await modelsApi.selectModel({ provider: provider.name, model: name });
      onActivated();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  const installedIds = new Set(installed.map((m) => m.id));
  const downloadable = catalog.filter((m) => !installedIds.has(m.id) && !m.installed);

  return (
    <Card>
      <CardTitle>Offline models ({provider.label})</CardTitle>
      <ErrorBox error={error} />

      {installName && (
        <div className="mb-3">
          <ProgressBar value={installPct} label={`Installing ${installName} · ${installMsg}`} />
        </div>
      )}

      <p className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Installed</p>
      {installed.length === 0 ? (
        <p className="mb-3 text-sm text-fg-muted">No models installed yet. Install one below.</p>
      ) : (
        <ul className="mb-4 divide-y divide-border">
          {installed.map((m) => (
            <li key={m.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm">{m.id}</span>
                  {m.kind !== "chat" && <Badge tone="neutral">{m.kind}</Badge>}
                  {m.multilingual && <Badge tone="accent">multilingual</Badge>}
                </div>
                <div className="text-xs text-fg-muted">
                  {m.parameter_size && <span>{m.parameter_size} · </span>}
                  {m.quantization && <span>{m.quantization} · </span>}
                  {m.size_bytes != null && <span>{formatBytes(m.size_bytes)}</span>}
                </div>
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="secondary" disabled={busy} onClick={() => activate(m.id)}>
                  Use
                </Button>
                <Button size="sm" variant="ghost" disabled={busy} onClick={() => remove(m.id)}>
                  Delete
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}

      <p className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">
        Downloadable
      </p>
      <ul className="divide-y divide-border">
        {downloadable.map((m) => (
          <li key={m.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm">{m.id}</span>
                {m.multilingual && <Badge tone="accent">multilingual</Badge>}
              </div>
              <div className="text-xs text-fg-muted">
                {m.description}
                {m.ram_estimate_bytes != null && (
                  <span> · ~{formatBytes(m.ram_estimate_bytes)} RAM</span>
                )}
              </div>
            </div>
            <Button
              size="sm"
              disabled={!!installName}
              loading={installName === m.id}
              onClick={() => install(m.id)}
            >
              Install
            </Button>
          </li>
        ))}
      </ul>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// Online (API providers)
// --------------------------------------------------------------------------- //

function OnlineModels({
  provider,
  onActivated,
  onKeyChange,
}: {
  provider: ProviderCapabilities;
  onActivated: () => void;
  onKeyChange: () => void;
}) {
  const [apiKey, setApiKey] = React.useState("");
  const [models, setModels] = React.useState<ProviderModel[]>([]);
  const [selected, setSelected] = React.useState<string>("");
  const [error, setError] = React.useState<Error | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [validateMsg, setValidateMsg] = React.useState<string | null>(null);

  const loadModels = React.useCallback(async () => {
    try {
      const res = await modelsApi.listOnlineModels(provider.name);
      setModels(res.models);
      if (res.models.length && !selected) setSelected(res.models[0].id);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, [provider.name, selected]);

  React.useEffect(() => {
    void loadModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider.name]);

  const saveKey = async () => {
    if (!apiKey) return;
    setBusy(true);
    setError(null);
    try {
      await modelsApi.setApiKey(provider.name, apiKey);
      setApiKey("");
      onKeyChange();
      await loadModels();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  const removeKey = async () => {
    setBusy(true);
    try {
      await modelsApi.deleteApiKey(provider.name);
      onKeyChange();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  const validate = async () => {
    setBusy(true);
    setValidateMsg(null);
    try {
      const res = await modelsApi.validateConnection(provider.name);
      setValidateMsg(res.ok ? `✓ ${res.message}` : `✗ ${res.message}`);
    } catch (e) {
      setValidateMsg(`✗ ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const activate = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await modelsApi.selectModel({ provider: provider.name, model: selected });
      onActivated();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardTitle>Online models ({provider.label})</CardTitle>
      <ErrorBox error={error} />

      <div className="mb-4">
        <p className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">API key</p>
        {provider.key_configured ? (
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="success">configured · {provider.key_hint}</Badge>
            <Button size="sm" variant="ghost" disabled={busy} onClick={removeKey}>
              Remove key
            </Button>
            <Button size="sm" variant="secondary" disabled={busy} onClick={validate}>
              Test connection
            </Button>
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            <Input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={`${provider.label} API key`}
              className="max-w-md"
              autoComplete="off"
            />
            <Button size="sm" disabled={busy || !apiKey} onClick={saveKey}>
              Save key
            </Button>
          </div>
        )}
        <p className="mt-1 text-xs text-fg-muted">
          Stored server-side only — never in the browser, never logged.
        </p>
        {validateMsg && <p className="mt-1 text-xs">{validateMsg}</p>}
      </div>

      <p className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Model</p>
      <div className="flex flex-wrap items-center gap-2">
        <Select
          aria-label="Online model"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          options={models.map((m) => ({
            value: m.id,
            label: m.label ?? m.id,
          }))}
          className="max-w-sm"
        />
        <Button size="sm" disabled={busy || !selected} onClick={activate}>
          Use this model
        </Button>
      </div>
      {!provider.key_configured && (
        <p className="mt-2 text-xs text-fg-muted">
          Save an API key to enable live model listing and chat.
        </p>
      )}
    </Card>
  );
}
