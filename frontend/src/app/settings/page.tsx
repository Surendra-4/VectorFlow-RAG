import { SettingsDashboard } from "@/components/settings/SettingsDashboard";

export default function SettingsPage() {
  return (
    <section aria-label="Settings" className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-fg-muted">
          Choose models (local or via API), tune retrieval live, and build,
          compare, and switch FAISS indexes — all without restarting the backend.
        </p>
      </header>
      <SettingsDashboard />
    </section>
  );
}
