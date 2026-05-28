import { SearchForm } from "@/components/search/SearchForm";

export default function SearchPage() {
  return (
    <section aria-label="Search" className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Search</h1>
        <p className="text-sm text-fg-muted">
          Retrieval-only mode. Shows the full hybrid+rerank result list with
          per-modality ranks and provenance.
        </p>
      </header>
      <SearchForm />
    </section>
  );
}
