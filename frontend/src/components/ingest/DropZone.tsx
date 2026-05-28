"use client";

import * as React from "react";
import { cn } from "@/lib/utils/cn";

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
        "flex min-h-[180px] cursor-pointer flex-col items-center justify-center gap-2 rounded border-2 border-dashed bg-surface px-4 py-8 text-center",
        hover ? "border-accent bg-accent/5" : "border-border",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <p className="text-sm font-medium text-fg">
        Drop files here or click to browse
      </p>
      <p className="text-xs text-fg-muted">
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
