import { SettingsDashboard } from "@/components/settings/SettingsDashboard";
import { PageHeader } from "@/components/layout/PageHeader";
import { SettingsIcon } from "@/components/ui/icons";

export const metadata = { title: "Settings" };

export default function SettingsPage() {
  return (
    <section aria-label="Settings" className="space-y-8">
      <PageHeader
        eyebrow="Configuration"
        title="Platform"
        highlight="settings"
        icon={<SettingsIcon />}
        description="Choose models (local or via API), tune retrieval live, and build, compare, and switch FAISS indexes — all without restarting the backend."
      />
      <SettingsDashboard />
    </section>
  );
}
