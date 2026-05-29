// src/lib/hooks/useTheme.ts

"use client";

import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";

const STORAGE_KEY = "vfr_theme";

function applyTheme(theme: Theme): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  const resolved =
    theme === "system"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
      : theme;
  if (resolved === "dark") root.classList.add("dark");
  else root.classList.remove("dark");
}

export function useTheme(): {
  theme: Theme;
  setTheme: (t: Theme) => void;
} {
  const [theme, setThemeState] = useState<Theme>("system");

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY) as Theme | null;
      const initial: Theme = stored ?? "system";
      setThemeState(initial);
      applyTheme(initial);
    } catch {
      applyTheme("system");
    }

    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      // Only react when in system mode.
      const current = (window.localStorage.getItem(STORAGE_KEY) as Theme | null) ?? "system";
      if (current === "system") applyTheme("system");
    };
    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, []);

  const setTheme = useCallback((next: Theme) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore (private mode, etc.)
    }
    setThemeState(next);
    applyTheme(next);
  }, []);

  return { theme, setTheme };
}
