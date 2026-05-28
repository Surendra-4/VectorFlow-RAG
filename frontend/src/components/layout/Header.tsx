"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "@/lib/hooks/useTheme";
import { cn } from "@/lib/utils/cn";

const NAV_ITEMS = [
  { href: "/", label: "Chat" },
  { href: "/search", label: "Search" },
  { href: "/ingest", label: "Ingest" },
  { href: "/documents", label: "Documents" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/traces", label: "Traces" },
  { href: "/settings", label: "Settings" },
];

const APP_TITLE = process.env.NEXT_PUBLIC_APP_TITLE || "VectorFlow-RAG";

export function Header() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();

  return (
    <header className="sticky top-0 z-20 border-b border-border bg-surface/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
        <Link href="/" className="text-base font-semibold text-fg">
          {APP_TITLE}
        </Link>

        <nav aria-label="Primary" className="hidden gap-1 md:flex">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "rounded px-3 py-1.5 text-sm transition-colors",
                  active
                    ? "bg-accent/15 text-accent"
                    : "text-fg-muted hover:bg-surface-raised hover:text-fg"
                )}
                aria-current={active ? "page" : undefined}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <select
            aria-label="Theme"
            value={theme}
            onChange={(e) => setTheme(e.target.value as "light" | "dark" | "system")}
            className="rounded border border-border bg-surface px-2 py-1 text-xs text-fg"
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
            <option value="system">System</option>
          </select>
        </div>
      </div>

      {/* Mobile nav — visible below md breakpoint */}
      <nav
        aria-label="Primary (mobile)"
        className="border-t border-border md:hidden"
      >
        <ul className="mx-auto flex max-w-7xl gap-1 overflow-x-auto px-2 py-2">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={cn(
                    "block whitespace-nowrap rounded px-3 py-1.5 text-sm",
                    active
                      ? "bg-accent/15 text-accent"
                      : "text-fg-muted hover:bg-surface-raised"
                  )}
                  aria-current={active ? "page" : undefined}
                >
                  {item.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
