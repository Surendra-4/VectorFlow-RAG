// src/lib/utils/cn.ts

/**
 * Tiny `clsx`-style class concatenator. Drops falsy values so callers can
 * write `cn("base", isActive && "active")` without ternaries.
 */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}
