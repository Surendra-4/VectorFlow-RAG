"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { ApiError, authApi } from "@/lib/api";
import { useAuth } from "@/lib/auth/AuthContext";
import { OrDivider, SocialButtons } from "./SocialButtons";

type Mode = "login" | "signup";

export function AuthCard({ mode }: { mode: Mode }) {
  const router = useRouter();
  const params = useSearchParams();
  const { login, signup } = useAuth();

  const [providers, setProviders] = React.useState<{ google: boolean; github: boolean }>({
    google: false,
    github: false,
  });
  const [name, setName] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  React.useEffect(() => {
    authApi
      .getProviders()
      .then((p) => setProviders({ google: p.google, github: p.github }))
      .catch(() => {});
  }, []);

  // Surface OAuth errors bounced back as ?error=...
  React.useEffect(() => {
    const e = params.get("error");
    if (e) setError(oauthErrorMessage(e));
  }, [params]);

  const submit: React.FormEventHandler = async (ev) => {
    ev.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "signup") await signup(email, password, name || undefined);
      else await login(email, password);
      router.replace(params.get("next") || "/");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const isSignup = mode === "signup";

  return (
    <div className="animate-fade-up">
      <h1 className="font-display text-2xl font-semibold tracking-tight">
        {isSignup ? "Create your account" : "Welcome back"}
      </h1>
      <p className="mt-1.5 text-sm text-fg-muted">
        {isSignup ? "Start exploring your documents in seconds." : "Sign in to continue to VectorFlow."}
      </p>

      <div className="mt-6 space-y-4">
        <SocialButtons google={providers.google} github={providers.github} />
        {(providers.google || providers.github) && <OrDivider label="or continue with email" />}

        <form onSubmit={submit} className="space-y-3">
          {isSignup && (
            <Field label="Name" htmlFor="name">
              <Input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Ada Lovelace" autoComplete="name" />
            </Field>
          )}
          <Field label="Email" htmlFor="email">
            <Input
              id="email" type="email" required value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com" autoComplete="email"
            />
          </Field>
          <Field
            label="Password" htmlFor="password"
            aside={
              !isSignup ? (
                <Link href="/reset" className="text-xs font-medium text-accent hover:underline">
                  Forgot?
                </Link>
              ) : undefined
            }
          >
            <Input
              id="password" type="password" required value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={isSignup ? "At least 8 characters" : "••••••••"}
              autoComplete={isSignup ? "new-password" : "current-password"}
              minLength={8}
            />
          </Field>

          {error && (
            <p role="alert" className="rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger">
              {error}
            </p>
          )}

          <Button type="submit" loading={busy} className="w-full" size="md">
            {isSignup ? "Create account" : "Sign in"}
          </Button>
        </form>
      </div>

      <p className="mt-6 text-center text-sm text-fg-muted">
        {isSignup ? "Already have an account? " : "New to VectorFlow? "}
        <Link
          href={isSignup ? "/login" : "/signup"}
          className="font-medium text-accent hover:underline"
        >
          {isSignup ? "Sign in" : "Create one"}
        </Link>
      </p>
    </div>
  );
}

function Field({
  label,
  htmlFor,
  aside,
  children,
}: {
  label: string;
  htmlFor: string;
  aside?: React.ReactNode;
  children: React.ReactNode;
}) {
  // The aside (e.g. "Forgot?") is a sibling of the <label>, not a child, so it
  // never pollutes the associated input's accessible name.
  return (
    <div className="block">
      <div className="mb-1 flex items-center justify-between">
        <label htmlFor={htmlFor} className="text-xs font-medium text-fg-muted">
          {label}
        </label>
        {aside}
      </div>
      {children}
    </div>
  );
}

function oauthErrorMessage(code: string): string {
  switch (code) {
    case "state_mismatch":
      return "Sign-in expired or was tampered with. Please try again.";
    case "access_denied":
      return "Sign-in was cancelled.";
    case "provider_unavailable":
      return "That sign-in method isn't available right now.";
    default:
      return "Sign-in failed. Please try again.";
  }
}
