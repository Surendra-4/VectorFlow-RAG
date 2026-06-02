"use client";

import { cn } from "@/lib/utils/cn";

interface ProgressBarProps {
  value: number; // 0..100
  label?: string;
  tone?: "accent" | "success" | "danger";
  className?: string;
}

const TONES = {
  accent: "bg-accent",
  success: "bg-success",
  danger: "bg-danger",
} as const;

export function ProgressBar({ value, label, tone = "accent", className }: ProgressBarProps) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className={className}>
      {label && (
        <div className="mb-1 flex justify-between text-xs text-fg-muted">
          <span>{label}</span>
          <span>{pct.toFixed(0)}%</span>
        </div>
      )}
      <div
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-2 w-full overflow-hidden rounded-full bg-border"
      >
        <div
          className={cn("h-full rounded-full transition-[width] duration-300", TONES[tone])}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
