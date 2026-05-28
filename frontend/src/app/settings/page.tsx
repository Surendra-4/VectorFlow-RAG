import { SettingsPanel } from "@/components/dashboard/SettingsPanel";

export default function SettingsPage() {
  return (
    <section aria-label="Settings" className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-fg-muted">
          Pipeline status, cache controls, and a client-side backend URL override.
        </p>
      </header>
      <SettingsPanel />
    </section>
  );
}
