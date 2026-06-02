"use client";

import * as React from "react";
import { cn } from "@/lib/utils/cn";
import { UploadIcon } from "@/components/ui/icons";

const SUPPORTED_EXTENSIONS = [
  ".txt", ".md", ".log", ".json",
  ".csv", ".tsv", ".xlsx", ".xlsm",
  ".db", ".sqlite", ".sqlite3",
  ".pdf", ".docx",
  ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
];

interface DropZoneProps {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
}

export function DropZone({ onFiles, disabled }: DropZoneProps) {
  const [hover, setHover] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleFiles = React.useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      onFiles(Array.from(files));
    },
    [onFiles]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setHover(true);
      }}
      onDragLeave={() => setHover(false)}
      onDrop={(e) => {
        e.preventDefault();
        setHover(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (disabled) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label="Upload files by clicking or dropping"
      className={cn(
        "group relative flex min-h-[200px] cursor-pointer flex-col items-center justify-center gap-3 overflow-hidden rounded-xl2 border-2 border-dashed px-4 py-8 text-center transition-all duration-300",
        hover
          ? "border-accent/70 bg-accent/[0.07] shadow-glow"
          : "border-border/70 bg-surface/60 backdrop-blur-sm hover:border-accent/40",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <span
        className={cn(
          "grid h-14 w-14 place-items-center rounded-2xl border border-border/60 bg-surface-raised/60 text-accent transition-transform duration-300",
          hover ? "scale-110 animate-float" : "group-hover:scale-105"
        )}
      >
        <UploadIcon className="h-6 w-6" />
      </span>
      <p className="text-sm font-semibold text-fg">
        Drop files here or click to browse
      </p>
      <p className="max-w-sm text-xs text-fg-muted">
        Supported: TXT · MD · JSON · CSV · XLSX · SQLite · PDF · DOCX · images (OCR)
      </p>
      <input
        ref={inputRef}
        type="file"
        multiple
        className="hidden"
        accept={SUPPORTED_EXTENSIONS.join(",")}
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
