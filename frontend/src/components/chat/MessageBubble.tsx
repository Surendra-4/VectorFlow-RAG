"use client";

import * as React from "react";
import { cn } from "@/lib/utils/cn";

interface MessageBubbleProps {
  role: "user" | "assistant";
  children: React.ReactNode;
}

export function MessageBubble({ role, children }: MessageBubbleProps) {
  const isUser = role === "user";
  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        // dir="auto" lets the browser pick LTR/RTL per the first strong
        // character — correct rendering for Arabic/Hebrew answers without
        // any language-specific branching.
        dir="auto"
        className={cn(
          "max-w-[85%] rounded-lg px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap",
          isUser
            ? "bg-accent text-accent-fg"
            : "bg-surface border border-border text-fg"
        )}
        data-role={role}
      >
        {children}
      </div>
    </div>
  );
}
