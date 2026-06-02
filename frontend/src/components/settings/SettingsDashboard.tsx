"use client";

import * as React from "react";
import { Tabs, TabPanel } from "@/components/ui/Tabs";
import { SettingsPanel } from "@/components/dashboard/SettingsPanel";
import { ModelsTab } from "./ModelsTab";
import { RetrievalTab } from "./RetrievalTab";
import { IndexesTab } from "./IndexesTab";

const TABS = [
  { id: "models", label: "Models" },
  { id: "retrieval", label: "Retrieval" },
  { id: "indexes", label: "Indexes" },
  { id: "status", label: "Status & cache" },
];

/**
 * Professional settings dashboard (Phase 12k).
 *
 * - Models: pick a local open-source model (Ollama) or a closed-source model
 *   via a provider API key.
 * - Retrieval: live-tune reranker / expansion / fusion (applied immediately).
 * - Indexes: build advanced FAISS recipes, switch indexes, check
 *   compatibility, and benchmark — all backed by background jobs.
 * - Status & cache: the existing read-only panel + backend-URL override.
 */
export function SettingsDashboard() {
  const [active, setActive] = React.useState("models");

  return (
    <div>
      <Tabs tabs={TABS} active={active} onChange={setActive} />
      <TabPanel id="models" active={active}>
        <ModelsTab />
      </TabPanel>
      <TabPanel id="retrieval" active={active}>
        <RetrievalTab />
      </TabPanel>
      <TabPanel id="indexes" active={active}>
        <IndexesTab />
      </TabPanel>
      <TabPanel id="status" active={active}>
        <SettingsPanel />
      </TabPanel>
    </div>
  );
}
