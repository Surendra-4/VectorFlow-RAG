import { Suspense } from "react";
import { AuthShell } from "@/components/auth/AuthShell";
import { AuthCard } from "@/components/auth/AuthCard";

export const metadata = { title: "Create account" };

export default function SignupPage() {
  return (
    <AuthShell>
      <Suspense>
        <AuthCard mode="signup" />
      </Suspense>
    </AuthShell>
  );
}
