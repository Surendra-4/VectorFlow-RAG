import { IngestForm } from "@/components/ingest/IngestForm";

export default function IngestPage() {
  return (
    <section aria-label="Ingest" className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Ingest</h1>
        <p className="text-sm text-fg-muted">
          Upload documents or paste text. Files are processed by the backend's loader
          registry and indexed with stable chunk identity.
        </p>
      </header>
      <IngestForm />
    </section>
  );
}
