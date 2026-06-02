/**
 * Fixed, decorative app background: a faint vector-space grid, a top glow, and
 * slow-drifting aurora blobs. Purely visual (aria-hidden, pointer-events-none),
 * GPU-cheap (a few blurred gradients), and disabled under prefers-reduced-motion
 * via the global CSS guard. No hooks → renders on the server.
 */
export function AuroraBackground() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden"
    >
      {/* Vector-space grid, faded toward the edges. */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage:
            "linear-gradient(to right, rgb(var(--grid-line) / var(--grid-alpha)) 1px, transparent 1px)," +
            "linear-gradient(to bottom, rgb(var(--grid-line) / var(--grid-alpha)) 1px, transparent 1px)",
          backgroundSize: "44px 44px",
          maskImage: "radial-gradient(ellipse 75% 60% at 50% 0%, #000 35%, transparent 80%)",
          WebkitMaskImage: "radial-gradient(ellipse 75% 60% at 50% 0%, #000 35%, transparent 80%)",
        }}
      />
      {/* Top aurora glow. */}
      <div className="absolute inset-x-0 top-0 h-[60vh] bg-grid-fade" />

      {/* Drifting aurora blobs. */}
      <div
        className="absolute -left-[10%] top-[-15%] h-[42rem] w-[42rem] rounded-full opacity-[0.22] blur-[120px] animate-float-slow"
        style={{ background: "radial-gradient(circle, rgb(var(--accent-3)), transparent 60%)" }}
      />
      <div
        className="absolute right-[-12%] top-[8%] h-[38rem] w-[38rem] rounded-full opacity-[0.20] blur-[120px] animate-float-slow"
        style={{
          background: "radial-gradient(circle, rgb(var(--accent-2)), transparent 60%)",
          animationDelay: "-6s",
        }}
      />
      <div
        className="absolute bottom-[-20%] left-[30%] h-[40rem] w-[40rem] rounded-full opacity-[0.16] blur-[130px] animate-float-slow"
        style={{
          background: "radial-gradient(circle, rgb(var(--accent)), transparent 60%)",
          animationDelay: "-12s",
        }}
      />
    </div>
  );
}
