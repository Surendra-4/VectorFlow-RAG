import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { DropZone } from "@/components/ingest/DropZone";

function makeFile(name: string, content = "x"): File {
  return new File([content], name, { type: "text/plain" });
}

describe("DropZone", () => {
  it("renders the drop hint", () => {
    render(<DropZone onFiles={vi.fn()} />);
    expect(screen.getByText(/Drop files here or click to browse/i)).toBeInTheDocument();
  });

  it("calls onFiles with dropped files", () => {
    const onFiles = vi.fn();
    render(<DropZone onFiles={onFiles} />);
    const zone = screen.getByRole("button");
    const file = makeFile("hello.txt");
    fireEvent.drop(zone, {
      dataTransfer: { files: [file] },
    });
    expect(onFiles).toHaveBeenCalledWith([file]);
  });

  it("does not call onFiles when disabled", () => {
    const onFiles = vi.fn();
    render(<DropZone onFiles={onFiles} disabled />);
    const zone = screen.getByRole("button");
    fireEvent.drop(zone, {
      dataTransfer: { files: [makeFile("hello.txt")] },
    });
    expect(onFiles).not.toHaveBeenCalled();
  });

  it("is keyboard-focusable when enabled (a11y)", () => {
    render(<DropZone onFiles={vi.fn()} />);
    const zone = screen.getByRole("button");
    expect(zone).toHaveAttribute("tabindex", "0");
    expect(zone).not.toHaveAttribute("aria-disabled", "true");
  });

  it("removes tab-stop and marks aria-disabled when disabled (a11y)", () => {
    render(<DropZone onFiles={vi.fn()} disabled />);
    const zone = screen.getByRole("button");
    expect(zone).toHaveAttribute("tabindex", "-1");
    expect(zone).toHaveAttribute("aria-disabled", "true");
  });
});
