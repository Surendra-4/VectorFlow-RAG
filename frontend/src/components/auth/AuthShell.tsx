import * as React from "react";
import { Constellation } from "@/components/brand/Constellation";
import { LogoMark, Wordmark } from "@/components/brand/Logo";
import { CheckIcon } from "@/components/ui/icons";

const HIGHLIGHTS = [
  "Hybrid retrieval with provenance-rich citations",
  "Run local models or connect any provider",
  "Build, benchmark & switch FAISS indexes live",
  "100% local-first — your documents never leave",
];

/**
 * Split-screen auth layout: an aurora brand panel (constellation + value prop)
 * beside a centered form card. The panel is hidden on small screens so the form
 * gets the full width.
 */
export function AuthShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid min-h-screen lg:grid-cols-[1.1fr_1fr]">
      {/* Brand panel */}
      <aside className="relative hidden overflow-hidden border-r border-border/60 lg:block">
        <div className="absolute inset-0 bg-aurora opacity-[0.10]" />
        <Constellation className="opacity-60" />
        <div className="absolute inset-0 bg-gradient-to-t from-bg/70 via-transparent to-transparent" />
        <div className="relative flex h-full flex-col justify-between p-10 xl:p-14">
          <div className="flex items-center gap-2.5">
            <span className="grid h-10 w-10 place-items-center rounded-xl border border-border/70 bg-surface-raised/50 shadow-glow-sm">
              <LogoMark className="h-6 w-6" />
            </span>
            <Wordmark className="text-lg" />
          </div>

          <div className="max-w-md">
            <h2 className="font-display text-3xl font-semibold leading-tight tracking-tight xl:text-4xl">
              Your documents,{" "}
              <span className="gradient-text">answered with sources.</span>
            </h2>
            <ul className="mt-7 space-y-3">
              {HIGHLIGHTS.map((h) => (
                <li key={h} className="flex items-start gap-3 text-sm text-fg-muted">
                  <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-accent/15 text-accent">
                    <CheckIcon className="h-3 w-3" />
                  </span>
                  {h}
                </li>
              ))}
            </ul>
          </div>

          <p className="text-xs text-fg-muted">
            Local-first hybrid RAG platform · production-grade retrieval
          </p>
        </div>
      </aside>

      {/* Form column */}
      <main className="flex items-center justify-center px-5 py-10 sm:px-8">
        <div className="w-full max-w-sm">
          {/* Mobile logo (panel is hidden) */}
          <div className="mb-8 flex items-center justify-center gap-2.5 lg:hidden">
            <span className="grid h-9 w-9 place-items-center rounded-xl border border-border/70 bg-surface-raised/50">
              <LogoMark className="h-5 w-5" />
            </span>
            <Wordmark />
          </div>
          {children}
        </div>
      </main>
    </div>
  );
}
