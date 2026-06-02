"use client";

import * as React from "react";
import { Card } from "./Card";
import { AnimatedNumber } from "./motion";
import { cn } from "@/lib/utils/cn";

/**
 * Headline metric tile: an icon chip, a count-up value, a label, and an
 * optional hint. Used on the dashboard. Hover lifts + glows.
 */
export function StatCard({
  label,
  value,
  hint,
  icon,
  format,
  tone = "accent",
  className,
}: {
  label: string;
  value: number;
  hint?: React.ReactNode;
  icon?: React.ReactNode;
  format?: (n: number) => string;
  tone?: "accent" | "success" | "warning" | "danger";
  className?: string;
}) {
  const toneRing: Record<string, string> = {
    accent: "text-accent",
    success: "text-success",
    warning: "text-warning",
    danger: "text-danger",
  };
  return (
    <Card variant="gradient" interactive className={cn("flex items-start gap-3", className)}>
      {icon && (
        <span
          className={cn(
            "grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-border/60 bg-surface-raised/60 text-[1.3rem]",
            toneRing[tone]
          )}
        >
          {icon}
        </span>
      )}
      <div className="min-w-0">
        <div className="font-display text-2xl font-semibold leading-none tracking-tight">
          <AnimatedNumber value={value} format={format} />
        </div>
        <div className="mt-1.5 text-xs font-medium uppercase tracking-wide text-fg-muted">
          {label}
        </div>
        {hint != null && <div className="mt-0.5 text-xs text-fg-muted/80">{hint}</div>}
      </div>
    </Card>
  );
}

/**
 * Sparkline — a tiny inline SVG area chart for a series of numbers. Decorative
 * trend cue; gracefully renders nothing for <2 points.
 */
export function Sparkline({
  data,
  width = 96,
  height = 28,
  className,
}: {
  data: number[];
  width?: number;
  height?: number;
  className?: string;
}) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const stepX = width / (data.length - 1);
  const pts = data.map((d, i) => {
    const x = i * stepX;
    const y = height - ((d - min) / span) * (height - 4) - 2;
    return [x, y] as const;
  });
  const line = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const area = `${line} L${width} ${height} L0 ${height} Z`;
  const id = React.useId();
  return (
    <svg width={width} height={height} className={className} aria-hidden="true">
      <defs>
        <linearGradient id={`spark-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="rgb(var(--accent))" stopOpacity="0.35" />
          <stop offset="1" stopColor="rgb(var(--accent))" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#spark-${id})`} />
      <path d={line} fill="none" stroke="rgb(var(--accent))" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
