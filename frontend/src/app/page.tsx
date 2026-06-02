import { ChatInterface } from "@/components/chat/ChatInterface";
import { Constellation } from "@/components/brand/Constellation";
import { Reveal } from "@/components/ui/motion";
import { SparkIcon } from "@/components/ui/icons";

const FEATURES = [
  "Hybrid retrieval · RRF",
  "Cross-encoder reranking",
  "Provenance-rich citations",
  "100% local-first",
];

export default function ChatPage() {
  return (
    <section aria-label="Chat" className="space-y-8">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-xl2 border border-border/60 glass">
        <Constellation className="opacity-70" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-bg/40" />
        <div className="relative px-6 py-12 sm:px-10 sm:py-16">
          <p className="animate-fade-up mb-3 inline-flex items-center gap-2 rounded-full border border-border/70 bg-surface-raised/50 px-3 py-1 text-xs font-medium text-fg-muted backdrop-blur">
            <SparkIcon className="text-accent" />
            Local-first hybrid RAG platform
          </p>
          <h1
            className="animate-fade-up font-display text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl"
            style={{ animationDelay: "60ms" }}
          >
            Ask your documents.
            <br className="hidden sm:block" />
            <span className="gradient-text">Get grounded answers.</span>
          </h1>
          <p
            className="animate-fade-up mt-4 max-w-xl text-sm leading-relaxed text-fg-muted sm:text-base"
            style={{ animationDelay: "120ms" }}
          >
            Retrieval and generation run entirely on your machine — every answer is traced back to
            the exact source chunks it came from.
          </p>
          <ul
            className="animate-fade-up mt-6 flex flex-wrap gap-2"
            style={{ animationDelay: "180ms" }}
          >
            {FEATURES.map((f) => (
              <li
                key={f}
                className="rounded-full border border-border/60 bg-surface/60 px-3 py-1 text-xs text-fg-muted backdrop-blur"
              >
                {f}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <Reveal>
        <ChatInterface />
      </Reveal>
    </section>
  );
}
