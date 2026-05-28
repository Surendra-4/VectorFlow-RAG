import { TraceTable } from "@/components/dashboard/TraceTable";

export default function TracesPage() {
  return (
    <section aria-label="Traces" className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Recent traces</h1>
        <p className="text-sm text-fg-muted">
          Each row is one search request. Expand to inspect the full
          RetrievalTrace JSON (expansion, fusion, rerank, cache, provenance).
        </p>
      </header>
      <TraceTable />
    </section>
  );
}
