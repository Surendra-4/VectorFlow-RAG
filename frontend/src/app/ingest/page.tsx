import { IngestForm } from "@/components/ingest/IngestForm";
import { PageHeader } from "@/components/layout/PageHeader";
import { Reveal } from "@/components/ui/motion";
import { UploadIcon } from "@/components/ui/icons";

export const metadata = { title: "Ingest" };

export default function IngestPage() {
  return (
    <section aria-label="Ingest" className="space-y-8">
      <PageHeader
        eyebrow="Ingestion"
        title="Add"
        highlight="documents"
        icon={<UploadIcon />}
        description="Upload files or paste text. The backend's loader registry parses 9 formats and indexes them with stable, content-derived chunk identity."
      />
      <Reveal>
        <IngestForm />
      </Reveal>
    </section>
  );
}
