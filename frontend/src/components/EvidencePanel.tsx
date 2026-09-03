import { useState } from "react";
import type { EvidenceItem, EvidenceStatus } from "../types/verification";
import { CHECK_LABELS, STAGE_LABELS } from "../types/verification";

/**
 * Renders the `evidence[]` array — the contract the whole system exists to
 * produce. Every check states its own outcome and its reasoning in a sentence
 * an officer can read aloud.
 *
 * `not_applicable` is shown, never hidden. A check that could not run on this
 * document is not evidence of authenticity, and quietly dropping it would let
 * an officer believe more was verified than actually was.
 */

const STATUS_STYLE: Record<EvidenceStatus, { label: string; chip: string; dot: string }> = {
  pass: {
    label: "Pass",
    chip: "bg-emerald-500/10 text-emerald-300 ring-emerald-500/25",
    dot: "bg-emerald-400",
  },
  weak: {
    label: "Weak",
    chip: "bg-amber-500/10 text-amber-300 ring-amber-500/25",
    dot: "bg-amber-400",
  },
  fail: {
    label: "Fail",
    chip: "bg-red-500/10 text-red-300 ring-red-500/25",
    dot: "bg-red-400",
  },
  not_applicable: {
    label: "Not applicable",
    chip: "bg-slate-500/10 text-slate-400 ring-slate-500/25",
    dot: "bg-slate-500",
  },
};

// Failures first: the officer should not have to scroll to find the problem.
const STATUS_ORDER: Record<EvidenceStatus, number> = {
  fail: 0,
  weak: 1,
  pass: 2,
  not_applicable: 3,
};

function label(check: string): string {
  return CHECK_LABELS[check] ?? check.replace(/_/g, " ");
}

interface Props {
  evidence: EvidenceItem[];
  onShowRegion?: (item: EvidenceItem) => void;
}

export default function EvidencePanel({ evidence, onShowRegion }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const sorted = [...evidence].sort(
    (a, b) => STATUS_ORDER[a.status] - STATUS_ORDER[b.status],
  );

  const counts = evidence.reduce<Record<EvidenceStatus, number>>(
    (acc, item) => ({ ...acc, [item.status]: (acc[item.status] ?? 0) + 1 }),
    { pass: 0, weak: 0, fail: 0, not_applicable: 0 },
  );

  const summary = [
    `${evidence.length} checks`,
    counts.fail ? `${counts.fail} failed` : null,
    counts.weak ? `${counts.weak} weak` : null,
    counts.pass ? `${counts.pass} passed` : null,
    counts.not_applicable ? `${counts.not_applicable} not applicable` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/40">
      <header className="flex items-baseline justify-between border-b border-slate-800 px-5 py-3.5">
        <h3 className="text-sm font-medium text-slate-200">Evidence</h3>
        <span className="text-xs text-slate-500">{summary}</span>
      </header>

      <ul className="divide-y divide-slate-800/70">
        {sorted.map((item) => {
          const style = STATUS_STYLE[item.status];
          const key = `${item.stage}:${item.check}`;
          const isOpen = expanded === key;

          return (
            <li key={key} className="px-5 py-4">
              <button
                type="button"
                onClick={() => setExpanded(isOpen ? null : key)}
                className="flex w-full items-start gap-3 text-left"
                aria-expanded={isOpen}
              >
                <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${style.dot}`} />

                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span className="text-sm font-medium text-slate-100">
                      {label(item.check)}
                    </span>
                    <span
                      className={`rounded px-1.5 py-0.5 text-[11px] font-medium ring-1 ${style.chip}`}
                    >
                      {style.label}
                    </span>
                    <span className="text-[11px] uppercase tracking-wide text-slate-600">
                      {STAGE_LABELS[item.stage] ?? item.stage}
                    </span>
                  </span>

                  <p
                    className={`mt-1.5 text-sm leading-relaxed text-slate-400 ${
                      isOpen ? "" : "line-clamp-2"
                    }`}
                  >
                    {item.detail}
                  </p>
                </span>
              </button>

              {isOpen && (
                <div className="mt-3 flex flex-wrap items-center gap-4 pl-5 text-xs text-slate-500">
                  {item.status !== "not_applicable" && (
                    <span>
                      Check confidence{" "}
                      <span className="font-mono text-slate-400">
                        {item.confidence.toFixed(2)}
                      </span>
                    </span>
                  )}
                  {item.regions.length > 0 && (
                    <>
                      <span>
                        {item.regions.length} region
                        {item.regions.length === 1 ? "" : "s"} on the scan
                      </span>
                      {onShowRegion && (
                        <button
                          type="button"
                          onClick={() => onShowRegion(item)}
                          className="text-sky-400 underline-offset-2 hover:underline"
                        >
                          View region on scan
                        </button>
                      )}
                    </>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>

      <footer className="border-t border-slate-800 px-5 py-3">
        <p className="text-xs leading-relaxed text-slate-600">
          Findings are region-level: they point at an area of the document, not at
          an exact pixel boundary. Checks marked not applicable could not run on
          this document and are not evidence that it is genuine.
        </p>
      </footer>
    </section>
  );
}
