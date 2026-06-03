"use client";

import * as React from "react";
import { usePathname, useRouter } from "next/navigation";
import { Header } from "@/components/layout/Header";
import { StatusBadge } from "@/components/layout/StatusBadge";
import { LogoMark } from "@/components/brand/Logo";
import { useAuth } from "@/lib/auth/AuthContext";
import { useTheme } from "@/lib/hooks/useTheme";

const APP_TITLE = process.env.NEXT_PUBLIC_APP_TITLE || "VectorFlow-RAG";

function isAuthPath(pathname: string): boolean {
  return (
    pathname === "/login" ||
    pathname === "/signup" ||
    pathname === "/reset" ||
    pathname.startsWith("/auth/")
  );
}

/**
 * Decides between the full-screen auth pages (no chrome) and the authenticated
 * app shell (header + content + footer). Unauthenticated visits to a protected
 * route bounce to /login with a ?next= return path.
 */
export function AppFrame({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { status, ready } = useAuth();
  const authRoute = isAuthPath(pathname);
  // Apply the saved theme on every route. The Header (which also calls
  // useTheme) is absent on the full-bleed auth pages, so without this the
  // login/signup screens would ignore the user's dark/light choice.
  useTheme();

  React.useEffect(() => {
    if (ready && status === "anonymous" && !authRoute) {
      const next = pathname && pathname !== "/" ? `?next=${encodeURIComponent(pathname)}` : "";
      router.replace(`/login${next}`);
    }
  }, [ready, status, authRoute, pathname, router]);

  // Auth screens render themselves full-bleed.
  if (authRoute) return <>{children}</>;

  // Loading the session, or about to redirect — show a calm splash.
  if (!ready || status !== "authenticated") return <Splash />;

  return (
    <>
      <Header />
      <main className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:py-12">{children}</main>
      <footer className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 pb-8 pt-4 text-xs text-fg-muted sm:px-6">
        <span className="flex items-center gap-1.5">
          <span className="font-display font-semibold text-fg">{APP_TITLE}</span>
          <span aria-hidden>·</span> local-first
        </span>
        <StatusBadge />
      </footer>
    </>
  );
}

function Splash() {
  return (
    <div className="grid min-h-[70vh] place-items-center">
      <div className="flex flex-col items-center gap-3 animate-fade-in">
        <span className="grid h-12 w-12 place-items-center rounded-2xl border border-border/70 bg-surface-raised/50 shadow-glow-sm">
          <LogoMark className="h-7 w-7" />
        </span>
        <span
          aria-hidden
          className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-r-transparent"
        />
        <span className="sr-only">Loading…</span>
      </div>
    </div>
  );
}
