import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Surface + accent tokens — themed via CSS variables (see globals.css)
        // so dark mode is one class toggle, not a Tailwind variant tree.
        bg: "rgb(var(--bg) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        "surface-raised": "rgb(var(--surface-raised) / <alpha-value>)",
        border: "rgb(var(--border) / <alpha-value>)",
        fg: "rgb(var(--fg) / <alpha-value>)",
        "fg-muted": "rgb(var(--fg-muted) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        "accent-fg": "rgb(var(--accent-fg) / <alpha-value>)",
        "accent-2": "rgb(var(--accent-2) / <alpha-value>)",
        "accent-3": "rgb(var(--accent-3) / <alpha-value>)",
        glow: "rgb(var(--glow) / <alpha-value>)",
        success: "rgb(var(--success) / <alpha-value>)",
        warning: "rgb(var(--warning) / <alpha-value>)",
        danger: "rgb(var(--danger) / <alpha-value>)",
      },
      fontFamily: {
        // Provided by next/font as CSS variables (see layout.tsx).
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "var(--font-sans)", "ui-sans-serif", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      borderRadius: {
        DEFAULT: "0.625rem",
        xl2: "1.125rem",
      },
      boxShadow: {
        soft: "0 1px 2px rgb(0 0 0 / 0.04), 0 8px 24px -12px rgb(0 0 0 / 0.18)",
        lift: "0 2px 4px rgb(0 0 0 / 0.06), 0 18px 40px -20px rgb(0 0 0 / 0.35)",
        glow: "0 0 0 1px rgb(var(--accent) / 0.25), 0 12px 40px -12px rgb(var(--glow) / 0.45)",
        "glow-sm": "0 8px 28px -14px rgb(var(--glow) / 0.55)",
      },
      backgroundImage: {
        aurora:
          "linear-gradient(120deg, rgb(var(--accent-3)) 0%, rgb(var(--accent)) 45%, rgb(var(--accent-2)) 100%)",
        "grid-fade":
          "radial-gradient(ellipse 80% 60% at 50% -10%, rgb(var(--glow) / 0.16), transparent 70%)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(14px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "scale-in": {
          "0%": { opacity: "0", transform: "scale(0.96)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
        "gradient-x": {
          "0%, 100%": { "background-position": "0% 50%" },
          "50%": { "background-position": "100% 50%" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        "float-slow": {
          "0%, 100%": { transform: "translate(0,0) scale(1)" },
          "33%": { transform: "translate(3%, -4%) scale(1.05)" },
          "66%": { transform: "translate(-2%, 3%) scale(0.97)" },
        },
        "pulse-glow": {
          "0%, 100%": { opacity: "0.55", transform: "scale(1)" },
          "50%": { opacity: "1", transform: "scale(1.15)" },
        },
        "spin-slow": {
          to: { transform: "rotate(360deg)" },
        },
        "draw-in": {
          to: { "stroke-dashoffset": "0" },
        },
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.2" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.6s cubic-bezier(0.22,1,0.36,1) both",
        "fade-in": "fade-in 0.6s ease-out both",
        "scale-in": "scale-in 0.4s cubic-bezier(0.22,1,0.36,1) both",
        shimmer: "shimmer 1.6s infinite",
        "gradient-x": "gradient-x 6s ease infinite",
        float: "float 6s ease-in-out infinite",
        "float-slow": "float-slow 18s ease-in-out infinite",
        "pulse-glow": "pulse-glow 3s ease-in-out infinite",
        "spin-slow": "spin-slow 14s linear infinite",
        blink: "blink 1s step-start infinite",
      },
    },
  },
  plugins: [],
};

export default config;
