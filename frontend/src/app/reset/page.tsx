import { Suspense } from "react";
import { AuthShell } from "@/components/auth/AuthShell";
import { ResetCard } from "@/components/auth/ResetCard";

export const metadata = { title: "Reset password" };

export default function ResetPage() {
  return (
    <AuthShell>
      <Suspense>
        <ResetCard />
      </Suspense>
    </AuthShell>
  );
}
