"use client";

import * as React from "react";
import { authApi } from "@/lib/api";
import { GithubIcon, GoogleIcon } from "./brandIcons";

/**
 * OAuth sign-in buttons. Only providers the deployment has configured are
 * shown. These are real top-level navigations (full-page redirect to the
 * backend's /auth/{provider}), so the state cookie + provider redirect work.
 */
export function SocialButtons({
  google,
  github,
}: {
  google: boolean;
  github: boolean;
}) {
  if (!google && !github) return null;

  const go = (provider: "google" | "github") => {
    window.location.href = authApi.oauthStartUrl(provider);
  };

  return (
    <div className="grid gap-2">
      {google && (
        <button
          type="button"
          onClick={() => go("google")}
          className="flex w-full items-center justify-center gap-2.5 rounded-lg border border-border/80 bg-surface-raised/60 px-4 py-2.5 text-sm font-medium text-fg transition-colors hover:border-accent/40 hover:bg-surface-raised"
        >
          <GoogleIcon />
          Continue with Google
        </button>
      )}
      {github && (
        <button
          type="button"
          onClick={() => go("github")}
          className="flex w-full items-center justify-center gap-2.5 rounded-lg border border-border/80 bg-surface-raised/60 px-4 py-2.5 text-sm font-medium text-fg transition-colors hover:border-accent/40 hover:bg-surface-raised"
        >
          <GithubIcon />
          Continue with GitHub
        </button>
      )}
    </div>
  );
}

export function OrDivider({ label = "or" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-xs text-fg-muted">
      <span className="h-px flex-1 bg-border" />
      {label}
      <span className="h-px flex-1 bg-border" />
    </div>
  );
}
