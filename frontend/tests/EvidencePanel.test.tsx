import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import EvidencePanel from "../src/components/EvidencePanel";
import type { EvidenceItem, EvidenceStatus } from "../src/types/verification";

/**
 * The evidence panel is where the officer actually reads the system's findings,
 * so its failure mode is not a broken layout — it is showing a green tick a
 * check has not earned.
 *
 * Five separate defects in this project came from exactly that: a check that
 * could not run being presented as one that passed. These tests exist to catch
 * the sixth.
 */

function item(overrides: Partial<EvidenceItem> = {}): EvidenceItem {
  return {
    stage: "forensics",
    check: "error_level_analysis",
    status: "pass" as EvidenceStatus,
    detail: "Compression error consistent across the document.",
    confidence: 0.9,
    regions: [],
    ...overrides,
  };
}

describe("EvidencePanel", () => {
  it("renders every check it is given", () => {
    render(
      <EvidencePanel
        evidence={[
          item({ check: "mrz_checksum", stage: "ocr_mrz" }),
          item({ check: "copy_move_detection" }),
        ]}
      />,
    );

    expect(screen.getByText(/MRZ checksum and OCR agreement/i)).toBeInTheDocument();
    expect(screen.getByText(/Duplicated region detection/i)).toBeInTheDocument();
  });

  it("labels a not-applicable check as such, never as a pass", () => {
    render(
      <EvidencePanel
        evidence={[
          item({
            check: "noise_consistency",
            status: "not_applicable",
            detail: "Advisory only: this check is not yet calibrated.",
          }),
        ]}
      />,
    );

    expect(screen.getByText("Not applicable")).toBeInTheDocument();
    expect(screen.queryByText("Pass")).not.toBeInTheDocument();
  });

  it("distinguishes weak from fail", () => {
    render(
      <EvidencePanel
        evidence={[
          item({ check: "face_match", stage: "face", status: "weak" }),
          item({ check: "mrz_checksum", stage: "ocr_mrz", status: "fail" }),
        ]}
      />,
    );

    expect(screen.getByText("Weak")).toBeInTheDocument();
    expect(screen.getByText("Fail")).toBeInTheDocument();
  });

  it("orders failures before passes so the problem is not below the fold", () => {
    render(
      <EvidencePanel
        evidence={[
          item({ check: "copy_move_detection", status: "pass" }),
          item({ check: "error_level_analysis", status: "pass" }),
          item({ check: "mrz_checksum", stage: "ocr_mrz", status: "fail" }),
        ]}
      />,
    );

    const rendered = screen.getAllByRole("button").map((b) => b.textContent ?? "");
    const failIndex = rendered.findIndex((t) => t.includes("Fail"));
    const passIndex = rendered.findIndex((t) => t.includes("Pass"));

    expect(failIndex).toBeGreaterThanOrEqual(0);
    expect(failIndex).toBeLessThan(passIndex);
  });

  it("summarises the counts by status", () => {
    render(
      <EvidencePanel
        evidence={[
          item({ check: "mrz_checksum", stage: "ocr_mrz", status: "fail" }),
          item({ check: "copy_move_detection", status: "pass" }),
          item({ check: "noise_consistency", status: "not_applicable" }),
        ]}
      />,
    );

    const summary = screen.getByText(/3 checks/);
    expect(summary.textContent).toContain("1 failed");
    expect(summary.textContent).toContain("1 passed");
    expect(summary.textContent).toContain("1 not applicable");
  });

  it("states that a not-applicable check is not evidence of authenticity", () => {
    render(<EvidencePanel evidence={[item()]} />);

    expect(
      screen.getByText(/not evidence that it is genuine/i),
    ).toBeInTheDocument();
  });

  it("does not promise pixel-level localization", () => {
    render(<EvidencePanel evidence={[item()]} />);

    expect(screen.getByText(/region-level/i)).toBeInTheDocument();
    expect(screen.getByText(/not at an exact pixel boundary/i)).toBeInTheDocument();
  });

  it("renders an unknown check name rather than dropping it", () => {
    render(<EvidencePanel evidence={[item({ check: "some_future_check" })]} />);

    expect(screen.getByText(/some future check/i)).toBeInTheDocument();
  });
});
