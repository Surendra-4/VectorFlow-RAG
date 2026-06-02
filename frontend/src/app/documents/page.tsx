import { DocumentsTable } from "@/components/dashboard/DocumentsTable";
import { PageHeader } from "@/components/layout/PageHeader";
import { Reveal } from "@/components/ui/motion";
import { DocsIcon } from "@/components/ui/icons";

export const metadata = { title: "Documents" };

export default function DocumentsPage() {
  return (
    <section aria-label="Documents" className="space-y-8">
      <PageHeader
        eyebrow="Corpus"
        title="Indexed"
        highlight="documents"
        icon={<DocsIcon />}
        description="Everything currently in the index, grouped by stable doc_id with chunk counts and provenance."
      />
      <Reveal>
        <DocumentsTable />
      </Reveal>
    </section>
  );
}
