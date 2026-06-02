import type { Metadata } from "next";
import { Inter, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/layout/Header";
import { StatusBadge } from "@/components/layout/StatusBadge";
import { AuroraBackground } from "@/components/layout/AuroraBackground";
import { cn } from "@/lib/utils/cn";

// Self-hosted at build time by next/font (no runtime network, no layout shift).
const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const display = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["500", "600", "700"],
  display: "swap",
});
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

const APP_TITLE = process.env.NEXT_PUBLIC_APP_TITLE || "VectorFlow-RAG";

export const metadata: Metadata = {
  title: { default: APP_TITLE, template: `%s · ${APP_TITLE}` },
  description: "Local-first hybrid RAG with provenance-rich retrieval.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn(sans.variable, display.variable, mono.variable)}
    >
      <body className="min-h-screen bg-bg font-sans text-fg antialiased">
        <AuroraBackground />
        <Header />
        <main className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:py-12">{children}</main>
        <footer className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 pb-8 pt-4 text-xs text-fg-muted sm:px-6">
          <span className="flex items-center gap-1.5">
            <span className="font-display font-semibold text-fg">{APP_TITLE}</span>
            <span aria-hidden>·</span> local-first
          </span>
          <StatusBadge />
        </footer>
      </body>
    </html>
  );
}
