"use client";

import * as React from "react";
import { cn } from "@/lib/utils/cn";

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "children"> {
  options: SelectOption[];
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  function Select({ className, options, ...rest }, ref) {
    return (
      <select
        ref={ref}
        className={cn(
          "w-full rounded border border-border bg-surface px-3 py-2 text-fg",
          "focus:border-accent focus:outline-none disabled:opacity-50",
          className
        )}
        {...rest}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
          </option>
        ))}
      </select>
    );
  }
);
