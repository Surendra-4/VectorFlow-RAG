"use client";

import * as React from "react";
import { cn } from "@/lib/utils/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
}

const VARIANTS: Record<Variant, string> = {
  // Aurora gradient + a sheen sweep that animates across on hover.
  primary:
    "group/btn relative overflow-hidden bg-aurora bg-[length:200%_auto] text-accent-fg shadow-glow-sm " +
    "hover:bg-[position:100%_50%] hover:shadow-glow disabled:opacity-50",
  secondary:
    "border border-border/80 bg-surface-raised/60 text-fg backdrop-blur-sm hover:border-accent/40 hover:text-fg disabled:opacity-50",
  ghost: "text-fg-muted hover:bg-surface-raised hover:text-fg disabled:opacity-40",
  danger: "bg-danger text-white hover:opacity-90 disabled:opacity-50",
};

const SIZES: Record<Size, string> = {
  sm: "px-3 py-1.5 text-sm",
  md: "px-4 py-2 text-sm",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    { className, variant = "primary", size = "md", loading, disabled, children, ...rest },
    ref
  ) {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-all duration-300",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg",
          "disabled:cursor-not-allowed disabled:shadow-none",
          VARIANTS[variant],
          SIZES[size],
          className
        )}
        {...rest}
      >
        {variant === "primary" && (
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/25 to-transparent transition-transform duration-700 group-hover/btn:translate-x-full"
          />
        )}
        {loading && (
          <span
            aria-hidden="true"
            className="h-3 w-3 animate-spin rounded-full border-2 border-current border-r-transparent"
          />
        )}
        <span className="relative">{children}</span>
      </button>
    );
  }
);
