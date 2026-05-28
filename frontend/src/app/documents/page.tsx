import { DocumentsTable } from "@/components/dashboard/DocumentsTable";

export default function DocumentsPage() {
  return (
    <section aria-label="Documents" className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
        <p className="text-sm text-fg-muted">
          What's currently in the index, grouped by stable doc_id.
        </p>
      </header>
      <DocumentsTable />
    </section>
  );
}
