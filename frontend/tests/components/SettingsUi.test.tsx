import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Toggle } from "@/components/ui/Toggle";
import { Tabs, TabPanel } from "@/components/ui/Tabs";
import { Select } from "@/components/ui/Select";
import { ProgressBar } from "@/components/ui/ProgressBar";

describe("Toggle", () => {
  it("renders as a switch with aria-checked and toggles", () => {
    const onChange = vi.fn();
    render(<Toggle checked={false} onChange={onChange} label="Reranker" />);
    const sw = screen.getByRole("switch", { name: "Reranker" });
    expect(sw).toHaveAttribute("aria-checked", "false");
    fireEvent.click(sw);
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("does not fire when disabled", () => {
    const onChange = vi.fn();
    render(<Toggle checked onChange={onChange} label="X" disabled />);
    fireEvent.click(screen.getByRole("switch"));
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("Tabs", () => {
  it("marks the active tab selected and switches panels", () => {
    const onChange = vi.fn();
    render(
      <>
        <Tabs
          tabs={[
            { id: "a", label: "Alpha" },
            { id: "b", label: "Beta" },
          ]}
          active="a"
          onChange={onChange}
        />
        <TabPanel id="a" active="a">
          <p>Panel A</p>
        </TabPanel>
        <TabPanel id="b" active="a">
          <p>Panel B</p>
        </TabPanel>
      </>
    );
    expect(screen.getByRole("tab", { name: "Alpha" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Panel A")).toBeInTheDocument();
    expect(screen.queryByText("Panel B")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Beta" }));
    expect(onChange).toHaveBeenCalledWith("b");
  });
});

describe("Select", () => {
  it("renders options and reports change", () => {
    const onChange = vi.fn();
    render(
      <Select
        aria-label="Model"
        value="a"
        onChange={onChange}
        options={[
          { value: "a", label: "Model A" },
          { value: "b", label: "Model B" },
        ]}
      />
    );
    const sel = screen.getByLabelText("Model") as HTMLSelectElement;
    expect(sel.value).toBe("a");
    fireEvent.change(sel, { target: { value: "b" } });
    expect(onChange).toHaveBeenCalled();
  });
});

describe("ProgressBar", () => {
  it("clamps and exposes aria values", () => {
    render(<ProgressBar value={150} label="Installing" />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "100");
    expect(screen.getByText("Installing")).toBeInTheDocument();
  });
});
