import type { VerifyResponse } from "../types/verification";

/**
 * Case strip above the verification workspace.
 *
 * Mirrors the team's Verify Screen mockup: case reference, pipeline stage, and
 * capture time on the left; the two actions an officer actually has at this
 * point on the right.
 *
 * The stage wording is deliberate. Before a result it reads "awaiting capture",
 * and after one "analysis complete — awaiting officer": the machine's work is
 * finished, the decision has not been made. A console that said "complete"
 * would imply the system had decided something it has not.
 */

interface Props {
  result: VerifyResponse | null;
  running: boolean;
  hasDocument: boolean;
  onReset: () => void;
}

/** Short, human-readable case reference derived from the verification id. */
function caseReference(result: VerifyResponse | null): string {
  if (!result) return "—";
  const date = new Date().toISOString().slice(0, 10).replace(/-/g, "");
  return `VD-${date}-${result.verification_id.slice(0, 4).toUpperCase()}`;
}

function stageLabel(result: VerifyResponse | null, running: boolean, hasDocument: boolean) {
  if (running) return "Analysis in progress";
  if (result) return "Analysis complete — awaiting officer";
  if (hasDocument) return "Capture ready — not yet analysed";
  return "Awaiting capture";
}

export default function CaseHeader({ result, running, hasDocument, onReset }: Props) {
  const captured = result
    ? new Date().toLocaleString(undefined, {
        day: "2-digit",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "—";

  const fields: [string, string][] = [
    ["Case", caseReference(result)],
    ["Stage", stageLabel(result, running, hasDocument)],
    ["Captured", captured],
  ];

  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-x-10 gap-y-4 border-b border-slate-800 pb-5">
      <dl className="flex flex-wrap gap-x-10 gap-y-3">
        {fields.map(([label, value]) => (
          <div key={label}>
            <dt className="text-[11px] uppercase tracking-wider text-slate-600">
              {label}
            </dt>
            <dd
              className={`mt-1 text-sm text-slate-200 ${
                label === "Case" ? "font-mono" : ""
              }`}
            >
              {value}
            </dd>
          </div>
        ))}
      </dl>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={onReset}
          disabled={running || (!result && !hasDocument)}
          className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 transition hover:border-slate-600 hover:text-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
        >
          New verification
        </button>
      </div>
    </header>
  );
}
