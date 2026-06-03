import type { Metadata } from "next";
import { Inter, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { AuroraBackground } from "@/components/layout/AuroraBackground";
import { AppFrame } from "@/components/layout/AppFrame";
import { AuthProvider } from "@/lib/auth/AuthContext";
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
        <AuthProvider>
          <AppFrame>{children}</AppFrame>
        </AuthProvider>
      </body>
    </html>
  );
}
