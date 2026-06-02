import { cn } from "@/lib/utils/cn";

/**
 * VectorFlow mark: a node retrieving from a small field of points — an arrow of
 * connected vectors converging on a highlighted result. Uses the aurora
 * gradient; the result node gently pulses (disabled under reduced-motion).
 */
export function LogoMark({ className, animated = true }: { className?: string; animated?: boolean }) {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      role="img"
      aria-label="VectorFlow logo"
      className={cn("h-7 w-7", className)}
    >
      <defs>
        <linearGradient id="vf-grad" x1="2" y1="4" x2="30" y2="28" gradientUnits="userSpaceOnUse">
          <stop stopColor="rgb(var(--accent-3))" />
          <stop offset="0.5" stopColor="rgb(var(--accent))" />
          <stop offset="1" stopColor="rgb(var(--accent-2))" />
        </linearGradient>
      </defs>
      {/* connecting vectors */}
      <path
        d="M6 23 L14 9 L20 19 L26 7"
        stroke="url(#vf-grad)"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* field nodes */}
      <circle cx="6" cy="23" r="2.1" fill="rgb(var(--accent-3))" />
      <circle cx="14" cy="9" r="2.1" fill="rgb(var(--accent))" />
      <circle cx="20" cy="19" r="2.1" fill="rgb(var(--accent))" />
      {/* highlighted result */}
      <circle
        cx="26"
        cy="7"
        r="3.2"
        fill="rgb(var(--accent-2))"
        className={animated ? "origin-center animate-pulse-glow" : undefined}
        style={{ transformBox: "fill-box" }}
      />
    </svg>
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn("font-display text-[1.05rem] font-semibold tracking-tight", className)}>
      Vector<span className="gradient-text">Flow</span>
    </span>
  );
}
