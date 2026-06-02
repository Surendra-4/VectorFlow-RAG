import { SearchForm } from "@/components/search/SearchForm";
import { PageHeader } from "@/components/layout/PageHeader";
import { Reveal } from "@/components/ui/motion";
import { SearchIcon } from "@/components/ui/icons";

export const metadata = { title: "Search" };

export default function SearchPage() {
  return (
    <section aria-label="Search" className="space-y-8">
      <PageHeader
        eyebrow="Retrieval"
        title="Hybrid"
        highlight="search"
        icon={<SearchIcon />}
        description="Retrieval-only mode — the full RRF-fused, optionally reranked result list with per-modality ranks and provenance, no LLM."
      />
      <Reveal>
        <SearchForm />
      </Reveal>
    </section>
  );
}
