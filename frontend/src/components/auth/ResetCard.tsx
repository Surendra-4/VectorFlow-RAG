"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { ApiError, authApi } from "@/lib/api";
import { useAuth } from "@/lib/auth/AuthContext";

export function ResetCard() {
  const params = useSearchParams();
  const token = params.get("token");
  return token ? <ConfirmReset token={token} /> : <RequestReset />;
}

function RequestReset() {
  const [email, setEmail] = React.useState("");
  const [sent, setSent] = React.useState(false);
  const [devLink, setDevLink] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  const submit: React.FormEventHandler = async (e) => {
    e.preventDefault();
    setBusy(true);
    try {
      const res = await authApi.resetRequest(email);
      setDevLink(res.reset_link ?? null);
      setSent(true);
    } catch {
      setSent(true); // opaque: never reveal whether the account exists
    } finally {
      setBusy(false);
    }
  };

  if (sent) {
    return (
      <div className="animate-fade-up">
        <h1 className="font-display text-2xl font-semibold tracking-tight">Check your email</h1>
        <p className="mt-1.5 text-sm text-fg-muted">
          If an account exists for <span className="font-medium text-fg">{email}</span>, a password-reset
          link is on its way.
        </p>
        {devLink && (
          <div className="mt-4 rounded-lg border border-warning/40 bg-warning/10 p-3 text-xs">
            <p className="mb-1 font-medium text-warning">Dev mode (no email configured)</p>
            <Link href={devLink.replace(/^https?:\/\/[^/]+/, "")} className="break-all text-accent hover:underline">
              {devLink}
            </Link>
          </div>
        )}
        <Link href="/login" className="mt-6 inline-block text-sm font-medium text-accent hover:underline">
          ← Back to sign in
        </Link>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="animate-fade-up">
      <h1 className="font-display text-2xl font-semibold tracking-tight">Reset password</h1>
      <p className="mt-1.5 text-sm text-fg-muted">
        Enter your email and we'll send you a link to reset it.
      </p>
      <div className="mt-6 space-y-3">
        <label htmlFor="email" className="block">
          <span className="mb-1 block text-xs font-medium text-fg-muted">Email</span>
          <Input
            id="email" type="email" required value={email}
            onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" autoComplete="email"
          />
        </label>
        <Button type="submit" loading={busy} className="w-full">Send reset link</Button>
      </div>
      <Link href="/login" className="mt-6 inline-block text-sm font-medium text-accent hover:underline">
        ← Back to sign in
      </Link>
    </form>
  );
}

function ConfirmReset({ token }: { token: string }) {
  const router = useRouter();
  const { adoptToken } = useAuth();
  const [password, setPassword] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  const submit: React.FormEventHandler = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await authApi.resetConfirm(token, password);
      await adoptToken(res.access_token);
      router.replace("/");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not reset your password.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="animate-fade-up">
      <h1 className="font-display text-2xl font-semibold tracking-tight">Set a new password</h1>
      <p className="mt-1.5 text-sm text-fg-muted">Choose a strong password you'll remember.</p>
      <div className="mt-6 space-y-3">
        <label htmlFor="password" className="block">
          <span className="mb-1 block text-xs font-medium text-fg-muted">New password</span>
          <Input
            id="password" type="password" required value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="At least 8 characters" autoComplete="new-password" minLength={8}
          />
        </label>
        {error && (
          <p role="alert" className="rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger">
            {error}
          </p>
        )}
        <Button type="submit" loading={busy} className="w-full">Reset password</Button>
      </div>
      <Link href="/login" className="mt-6 inline-block text-sm font-medium text-accent hover:underline">
        ← Back to sign in
      </Link>
    </form>
  );
}
