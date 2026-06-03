"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "@/lib/hooks/useTheme";
import { cn } from "@/lib/utils/cn";
import { LogoMark, Wordmark } from "@/components/brand/Logo";
import { MoonIcon, NAV_ICONS, SunIcon, SystemIcon } from "@/components/ui/icons";
import { useAuth } from "@/lib/auth/AuthContext";

const NAV_ITEMS = [
  { href: "/", label: "Chat" },
  { href: "/search", label: "Search" },
  { href: "/ingest", label: "Ingest" },
  { href: "/documents", label: "Documents" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/traces", label: "Traces" },
  { href: "/settings", label: "Settings" },
] as const;

export function Header() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-30 border-b border-border/60 glass">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-2.5 sm:px-6">
        <Link
          href="/"
          className="group flex items-center gap-2.5 rounded-lg outline-none"
          aria-label="VectorFlow home"
        >
          <span className="grid h-9 w-9 place-items-center rounded-xl border border-border/70 bg-surface-raised/60 shadow-glow-sm transition-transform group-hover:scale-105">
            <LogoMark className="h-5 w-5" />
          </span>
          <Wordmark className="hidden sm:inline" />
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-0.5 md:flex">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            const Icon = NAV_ICONS[item.href];
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "group relative flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors",
                  active ? "text-fg" : "text-fg-muted hover:text-fg"
                )}
              >
                {active && (
                  <span className="absolute inset-0 -z-10 rounded-lg bg-accent/12 shadow-[inset_0_0_0_1px_rgb(var(--accent)/0.25)] animate-scale-in" />
                )}
                <Icon className={cn(active ? "text-accent" : "opacity-70 group-hover:opacity-100")} />
                {item.label}
                {active && (
                  <span className="absolute -bottom-[9px] left-1/2 h-[2px] w-7 -translate-x-1/2 rounded-full bg-aurora" />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <ThemeToggle />
          <UserMenu />
        </div>
      </div>

      {/* Mobile nav — icon rail */}
      <nav aria-label="Primary (mobile)" className="border-t border-border/60 md:hidden">
        <ul className="mx-auto flex max-w-7xl gap-1 overflow-x-auto px-2 py-2">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            const Icon = NAV_ICONS[item.href];
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "flex items-center gap-1.5 whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-medium",
                    active ? "bg-accent/15 text-accent" : "text-fg-muted"
                  )}
                >
                  <Icon /> {item.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}

const THEMES = [
  { id: "light", Icon: SunIcon, label: "Light" },
  { id: "dark", Icon: MoonIcon, label: "Dark" },
  { id: "system", Icon: SystemIcon, label: "System" },
] as const;

function UserMenu() {
  const { user, logout } = useAuth();
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  if (!user) return null;
  const initials = (user.name || user.email).slice(0, 2).toUpperCase();

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="grid h-8 w-8 place-items-center overflow-hidden rounded-full border border-border/70 bg-accent/15 text-xs font-semibold text-accent transition-transform hover:scale-105"
      >
        {user.avatar_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={user.avatar_url} alt="" className="h-full w-full object-cover" />
        ) : (
          initials
        )}
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 mt-2 w-56 origin-top-right animate-scale-in rounded-xl border border-border/70 bg-surface/95 p-1.5 shadow-lift backdrop-blur"
        >
          <div className="border-b border-border/60 px-3 py-2">
            <p className="truncate text-sm font-medium">{user.name || user.email}</p>
            <p className="truncate text-xs text-fg-muted">{user.email}</p>
          </div>
          <Link
            href="/dashboard"
            role="menuitem"
            onClick={() => setOpen(false)}
            className="block rounded-lg px-3 py-2 text-sm text-fg-muted hover:bg-surface-raised hover:text-fg"
          >
            Dashboard & stats
          </Link>
          <button
            type="button"
            role="menuitem"
            onClick={() => { setOpen(false); logout(); }}
            className="block w-full rounded-lg px-3 py-2 text-left text-sm text-danger hover:bg-danger/10"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <div
      role="radiogroup"
      aria-label="Theme"
      className="flex items-center gap-0.5 rounded-xl border border-border/70 bg-surface-raised/50 p-0.5"
    >
      {THEMES.map(({ id, Icon, label }) => {
        const active = theme === id;
        return (
          <button
            key={id}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={label}
            title={label}
            onClick={() => setTheme(id)}
            className={cn(
              "grid h-7 w-7 place-items-center rounded-lg text-sm transition-colors",
              active
                ? "bg-accent/15 text-accent shadow-[inset_0_0_0_1px_rgb(var(--accent)/0.3)]"
                : "text-fg-muted hover:text-fg"
            )}
          >
            <Icon />
          </button>
        );
      })}
    </div>
  );
}
