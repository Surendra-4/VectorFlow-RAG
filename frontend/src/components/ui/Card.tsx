"use client";

import * as React from "react";
import { cn } from "@/lib/utils/cn";

export function Card({
  className,
  children,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-surface p-4 shadow-sm",
        className
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

export function CardTitle({
  className,
  children,
  ...rest
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn(
        "mb-1 text-sm font-semibold uppercase tracking-wide text-fg-muted",
        className
      )}
      {...rest}
    >
      {children}
    </h3>
  );
}
