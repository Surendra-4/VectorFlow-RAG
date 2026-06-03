"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { LogoMark } from "@/components/brand/Logo";
import { useAuth } from "@/lib/auth/AuthContext";

/**
 * OAuth landing: the backend redirects here with the JWT in the URL fragment
 * (`#access_token=...`). Fragments never reach a server, so the token stays
 * client-only. We adopt it, then replace history to drop the token from the URL.
 */
export default function AuthCallbackPage() {
  const router = useRouter();
  const { adoptToken } = useAuth();
  const [failed, setFailed] = React.useState(false);

  React.useEffect(() => {
    const hash = typeof window !== "undefined" ? window.location.hash : "";
    const token = new URLSearchParams(hash.replace(/^#/, "")).get("access_token");
    if (!token) {
      setFailed(true);
      const t = setTimeout(() => router.replace("/login?error=oauth_error"), 1200);
      return () => clearTimeout(t);
    }
    void adoptToken(token).then(() => router.replace("/"));
  }, [adoptToken, router]);

  return (
    <div className="grid min-h-screen place-items-center">
      <div className="flex flex-col items-center gap-3 text-center animate-fade-in">
        <span className="grid h-12 w-12 place-items-center rounded-2xl border border-border/70 bg-surface-raised/50 shadow-glow-sm">
          <LogoMark className="h-7 w-7" />
        </span>
        {failed ? (
          <p className="text-sm text-danger">Sign-in failed — redirecting…</p>
        ) : (
          <>
            <span aria-hidden className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-r-transparent" />
            <p className="text-sm text-fg-muted">Signing you in…</p>
          </>
        )}
      </div>
    </div>
  );
}
