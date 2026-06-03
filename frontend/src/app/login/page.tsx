import { Suspense } from "react";
import { AuthShell } from "@/components/auth/AuthShell";
import { AuthCard } from "@/components/auth/AuthCard";

export const metadata = { title: "Sign in" };

export default function LoginPage() {
  return (
    <AuthShell>
      <Suspense>
        <AuthCard mode="login" />
      </Suspense>
    </AuthShell>
  );
}
