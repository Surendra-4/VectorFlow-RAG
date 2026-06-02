import * as React from "react";
import { cn } from "@/lib/utils/cn";

/**
 * Consistent page header: an accent eyebrow, a large display title (with an
 * optional gradient-highlighted word), a muted description, and an optional
 * actions slot. Animates in on mount.
 */
export function PageHeader({
  eyebrow,
  title,
  highlight,
  description,
  icon,
  actions,
  className,
}: {
  eyebrow?: string;
  title: string;
  highlight?: string;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("animate-fade-up", className)}>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          {eyebrow && (
            <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-accent">
              {icon && <span className="text-accent">{icon}</span>}
              {eyebrow}
            </p>
          )}
          <h1 className="font-display text-3xl font-semibold tracking-tight sm:text-4xl">
            {title}
            {highlight && <span className="gradient-text"> {highlight}</span>}
          </h1>
          {description && (
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-fg-muted sm:text-[0.95rem]">
              {description}
            </p>
          )}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </div>
      <div className="mt-5 h-px w-full bg-gradient-to-r from-border via-border/40 to-transparent" />
    </div>
  );
}
