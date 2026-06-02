"use client";

import * as React from "react";
import { cn } from "@/lib/utils/cn";

type Variant = "default" | "glass" | "gradient";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: Variant;
  /** Lift + glow on hover (use for interactive/clickable cards). */
  interactive?: boolean;
}

/**
 * Surface container. `glass` frosts the background; `gradient` adds an aurora
 * hairline border. `interactive` adds a hover lift + glow for clickable cards.
 */
export function Card({
  className,
  variant = "default",
  interactive = false,
  children,
  ...rest
}: CardProps) {
  return (
    <div
      className={cn(
        "relative rounded-xl2 p-4 shadow-soft",
        variant === "glass" && "glass",
        variant === "gradient" && "glass border-gradient",
        variant === "default" && "border border-border/70 bg-surface/80 backdrop-blur-sm",
        interactive && "transition-all duration-300 hover:-translate-y-0.5 hover:shadow-glow",
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
        "mb-2 text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-fg-muted",
        className
      )}
      {...rest}
    >
      {children}
    </h3>
  );
}
