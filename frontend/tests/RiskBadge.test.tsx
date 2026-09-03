import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import RiskBadge from "../src/components/RiskBadge";

/**
 * The badge is the one element an officer reads in a second, so its wording
 * carries the system's central claim: it describes the document's evidence, and
 * the human decides. Those sentences are tested because a re-skin could quietly
 * drop them.
 */

describe("RiskBadge", () => {
  it.each([
    ["clear", "Clear"],
    ["review", "Review"],
    ["high_risk", "High risk"],
  ] as const)("renders the %s band as %s", (band, label) => {
    render(<RiskBadge band={band} score={0.5} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("shows the score to two decimals", () => {
    render(<RiskBadge band="high_risk" score={0.9512} />);
    expect(screen.getByText(/risk 0\.95/)).toBeInTheDocument();
  });

  it("says the score describes the document, not the traveller", () => {
    render(<RiskBadge band="high_risk" score={0.95} />);
    expect(
      screen.getByText(/describes the evidence found on the document, not the traveller/i),
    ).toBeInTheDocument();
  });

  it("states that the system does not accept or reject on its own", () => {
    render(<RiskBadge band="clear" score={0} />);
    expect(
      screen.getByText(/does not accept or reject on its own/i),
    ).toBeInTheDocument();
  });

  it("recommends inspection rather than rejection at high risk", () => {
    render(<RiskBadge band="high_risk" score={0.95} />);

    const text = document.body.textContent ?? "";
    expect(text).toMatch(/secondary inspection/i);

    // The badge must never *instruct* a rejection. It does legitimately contain
    // the word "reject" — in the sentence saying the system does not do it — so
    // match the instruction rather than the word.
    expect(text).not.toMatch(/reject (the )?traveller/i);
    expect(text).not.toMatch(/\bdeny\b|\brefuse entry\b/i);
    expect(text).toMatch(/does not accept or reject on its own/i);
  });

  it("compact mode drops the explanation but keeps the band", () => {
    render(<RiskBadge band="review" score={0.4} compact />);

    expect(screen.getByText("Review")).toBeInTheDocument();
    expect(screen.queryByText(/describes the evidence/i)).not.toBeInTheDocument();
  });
});
