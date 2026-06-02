import { TraceTable } from "@/components/dashboard/TraceTable";
import { PageHeader } from "@/components/layout/PageHeader";
import { Reveal } from "@/components/ui/motion";
import { TraceIcon } from "@/components/ui/icons";

export const metadata = { title: "Traces" };

export default function TracesPage() {
  return (
    <section aria-label="Traces" className="space-y-8">
      <PageHeader
        eyebrow="Inspector"
        title="Recent"
        highlight="traces"
        icon={<TraceIcon />}
        description="Each row is one search request. Expand to inspect the full RetrievalTrace — expansion, fusion, rerank, cache, and provenance."
      />
      <Reveal>
        <TraceTable />
      </Reveal>
    </section>
  );
}
