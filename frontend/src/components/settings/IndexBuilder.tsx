"use client";

import * as React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox } from "@/components/ui/ErrorBox";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Tabs } from "@/components/ui/Tabs";
import { indexesApi } from "@/lib/api";
import type { RecipeSpec, RecipeValidation } from "@/lib/api/types";
import { formatBytes } from "@/lib/utils/format";

/**
 * Advanced FAISS index builder with Basic/Advanced modes. Every parameter set
 * is validated against the backend (which validates statically *and* by
 * constructing the index in FAISS) before a build can be started — the brief's
 * "never expose raw FAISS complexity without validation" rule.
 */
export function IndexBuilder({
  dim,
  nVectors,
  onCreate,
}: {
  dim: number;
  nVectors: number;
  onCreate: (jobId: string) => void;
}) {
  const [mode, setMode] = React.useState<"basic" | "advanced">("basic");
  const [recipes, setRecipes] = React.useState<RecipeSpec[]>([]);
  const [recipeId, setRecipeId] = React.useState<string>("hnsw");
  const [params, setParams] = React.useState<Record<string, number>>({});
  const [validation, setValidation] = React.useState<RecipeValidation | null>(null);
  const [name, setName] = React.useState("");
  const [error, setError] = React.useState<Error | null>(null);
  const [busy, setBusy] = React.useState(false);

  // Load recipe catalog once.
  React.useEffect(() => {
    void indexesApi
      .listRecipes()
      .then((r) => setRecipes(r.recipes))
      .catch((e) => setError(e instanceof Error ? e : new Error(String(e))));
  }, []);

  const visibleRecipes = recipes.filter((r) => r.mode === mode);
  const spec = recipes.find((r) => r.id === recipeId) ?? null;

  // When the recipe changes, reset params to its defaults.
  React.useEffect(() => {
    if (!spec) return;
    const defaults: Record<string, number> = {};
    for (const p of spec.params) defaults[p.name] = p.default;
    setParams(defaults);
  }, [spec]);

  // If the recipe isn't in the current mode's list, pick the first one.
  React.useEffect(() => {
    if (visibleRecipes.length && !visibleRecipes.some((r) => r.id === recipeId)) {
      setRecipeId(visibleRecipes[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, recipes]);

  // Validate (debounced) whenever recipe/params/dim change.
  React.useEffect(() => {
    if (!spec || dim <= 0) return;
    const handle = setTimeout(() => {
      void indexesApi
        .validateRecipe({ recipe: recipeId, params, dim, n_vectors: nVectors })
        .then((r) => setValidation(r.validation))
        .catch(() => setValidation(null));
    }, 250);
    return () => clearTimeout(handle);
  }, [recipeId, params, dim, nVectors, spec]);

  const create = async () => {
    if (!name.trim()) {
      setError(new Error("Give the index a name."));
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await indexesApi.createIndex({
        name: name.trim(),
        backend: "faiss",
        index_type: recipeId,
        build_params: params,
        search_params: params,
        make_active: false,
      });
      onCreate(res.job_id);
      setName("");
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  };

  const est = validation?.estimate ?? null;
  const canBuild = !!validation?.ok && !!name.trim() && !busy;

  return (
    <Card>
      <CardTitle>Build a new index</CardTitle>

      <Tabs
        tabs={[
          { id: "basic", label: "Basic" },
          { id: "advanced", label: "Advanced" },
        ]}
        active={mode}
        onChange={(m) => setMode(m as "basic" | "advanced")}
        className="mb-3"
      />

      <ErrorBox error={error} />

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="mb-1 block text-xs uppercase tracking-wide text-fg-muted">Recipe</span>
          <Select
            aria-label="Recipe"
            value={recipeId}
            onChange={(e) => setRecipeId(e.target.value)}
            options={visibleRecipes.map((r) => ({ value: r.id, label: r.label }))}
          />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block text-xs uppercase tracking-wide text-fg-muted">
            Index name
          </span>
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. faiss-ivfpq-bge"
          />
        </label>
      </div>

      {spec && <p className="mt-2 text-xs text-fg-muted">{spec.description}</p>}

      {spec && spec.params.length > 0 && (
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
          {spec.params.map((p) => (
            <label key={p.name} className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wide text-fg-muted">
                {p.label}
              </span>
              <Input
                type="number"
                min={p.minimum}
                max={p.maximum}
                value={params[p.name] ?? p.default}
                title={p.description}
                onChange={(e) =>
                  setParams((prev) => ({ ...prev, [p.name]: Number(e.target.value) }))
                }
              />
            </label>
          ))}
        </div>
      )}

      {/* Validation + estimate */}
      <div className="mt-3 space-y-2">
        {validation && !validation.ok && (
          <div className="rounded border border-danger/40 bg-danger/10 p-2 text-xs">
            {validation.errors.map((e, i) => (
              <div key={i}>
                <strong>{e.field}:</strong> {e.message}
              </div>
            ))}
          </div>
        )}
        {validation?.warnings?.map((w, i) => (
          <p key={i} className="text-xs text-warning">
            ⚠ {w}
          </p>
        ))}
        {validation?.ok && (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge tone="neutral">factory: {validation.factory_string}</Badge>
            {est && <Badge tone="neutral">~{formatBytes(est.memory_bytes)} / {nVectors} vecs</Badge>}
            {est && <Badge tone="accent">{est.latency_class} search</Badge>}
            {est && <Badge tone="neutral">train: {est.training_cost}</Badge>}
          </div>
        )}
      </div>

      <div className="mt-3">
        <Button disabled={!canBuild} loading={busy} onClick={create}>
          Build index
        </Button>
        {dim <= 0 && (
          <p className="mt-1 text-xs text-fg-muted">
            Ingest documents first so the embedding dimension is known.
          </p>
        )}
      </div>
    </Card>
  );
}
