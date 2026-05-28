import type { Metadata } from "next";
import "./globals.css";
import { Header } from "@/components/layout/Header";
import { StatusBadge } from "@/components/layout/StatusBadge";

const APP_TITLE = process.env.NEXT_PUBLIC_APP_TITLE || "VectorFlow-RAG";

export const metadata: Metadata = {
  title: APP_TITLE,
  description: "Local-first hybrid RAG with provenance-rich retrieval.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-bg text-fg antialiased">
        <Header />
        <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
        <footer className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 text-xs text-fg-muted">
          <span>{APP_TITLE} · local-first</span>
          <StatusBadge />
        </footer>
      </body>
    </html>
  );
}
