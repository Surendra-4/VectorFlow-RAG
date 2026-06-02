import * as React from "react";

/**
 * Inline stroke-icon set — no icon dependency. 24px grid, currentColor stroke,
 * 1.75 weight to match the UI. Each icon takes standard SVG props.
 */
type IconProps = React.SVGProps<SVGSVGElement>;

function Base({ children, ...p }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="h-[1.15em] w-[1.15em]"
      {...p}
    >
      {children}
    </svg>
  );
}

export const ChatIcon = (p: IconProps) => (
  <Base {...p}><path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-3.7-.84L3 20.5l1.34-5.8A8.5 8.5 0 1 1 21 11.5Z" /></Base>
);
export const SearchIcon = (p: IconProps) => (
  <Base {...p}><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></Base>
);
export const UploadIcon = (p: IconProps) => (
  <Base {...p}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v13" /></Base>
);
export const DocsIcon = (p: IconProps) => (
  <Base {...p}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6M9 13h6M9 17h6" /></Base>
);
export const ActivityIcon = (p: IconProps) => (
  <Base {...p}><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></Base>
);
export const TraceIcon = (p: IconProps) => (
  <Base {...p}><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="18" r="2.5" /><path d="M8.5 6H15a3 3 0 0 1 3 3v6.5M6 8.5V15a3 3 0 0 0 3 3h6.5" /></Base>
);
export const SettingsIcon = (p: IconProps) => (
  <Base {...p}><path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6" /></Base>
);
export const SunIcon = (p: IconProps) => (
  <Base {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></Base>
);
export const MoonIcon = (p: IconProps) => (
  <Base {...p}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" /></Base>
);
export const SystemIcon = (p: IconProps) => (
  <Base {...p}><rect x="2" y="4" width="20" height="13" rx="2" /><path d="M8 21h8M12 17v4" /></Base>
);
export const SparkIcon = (p: IconProps) => (
  <Base {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18" /></Base>
);
export const CheckIcon = (p: IconProps) => (
  <Base {...p}><path d="M20 6 9 17l-5-5" /></Base>
);
export const ArrowRightIcon = (p: IconProps) => (
  <Base {...p}><path d="M5 12h14M13 6l6 6-6 6" /></Base>
);

export const NAV_ICONS = {
  "/": ChatIcon,
  "/search": SearchIcon,
  "/ingest": UploadIcon,
  "/documents": DocsIcon,
  "/dashboard": ActivityIcon,
  "/traces": TraceIcon,
  "/settings": SettingsIcon,
} as const;
