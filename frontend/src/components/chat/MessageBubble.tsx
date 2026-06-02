"use client";

import * as React from "react";
import { cn } from "@/lib/utils/cn";
import { LogoMark } from "@/components/brand/Logo";

interface MessageBubbleProps {
  role: "user" | "assistant";
  children: React.ReactNode;
}

export function MessageBubble({ role, children }: MessageBubbleProps) {
  const isUser = role === "user";
  return (
    <div className={cn("flex w-full animate-fade-up items-start gap-3", isUser ? "flex-row-reverse" : "flex-row")}>
      {/* Avatar */}
      <span
        aria-hidden="true"
        className={cn(
          "mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg border text-[0.7rem] font-semibold",
          isUser
            ? "border-accent/30 bg-accent/15 text-accent"
            : "border-border/70 bg-surface-raised/60"
        )}
      >
        {isUser ? "You" : <LogoMark className="h-4 w-4" animated={false} />}
      </span>
      <div
        // dir="auto" lets the browser pick LTR/RTL per the first strong
        // character — correct rendering for Arabic/Hebrew answers without
        // any language-specific branching.
        dir="auto"
        className={cn(
          "max-w-[85%] whitespace-pre-wrap rounded-xl2 px-4 py-3 text-sm leading-relaxed shadow-soft",
          isUser
            ? "rounded-tr-sm bg-aurora bg-[length:160%] text-accent-fg"
            : "rounded-tl-sm border border-border/70 bg-surface/85 text-fg backdrop-blur-sm"
        )}
        data-role={role}
      >
        {children}
      </div>
    </div>
  );
}
