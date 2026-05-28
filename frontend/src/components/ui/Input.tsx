"use client";

import * as React from "react";
import { cn } from "@/lib/utils/cn";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...rest }, ref) {
    return (
      <input
        ref={ref}
        className={cn(
          "w-full rounded border border-border bg-surface px-3 py-2 text-fg",
          "placeholder:text-fg-muted",
          "focus:border-accent focus:outline-none",
          "disabled:opacity-50",
          className
        )}
        {...rest}
      />
    );
  }
);

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(function Textarea({ className, ...rest }, ref) {
  return (
    <textarea
      ref={ref}
      className={cn(
        "w-full rounded border border-border bg-surface px-3 py-2 text-fg",
        "placeholder:text-fg-muted",
        "focus:border-accent focus:outline-none",
        "disabled:opacity-50",
        className
      )}
      {...rest}
    />
  );
});
