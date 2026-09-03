import type { RiskBand } from "../types/verification";
import { BAND_LABELS } from "../types/verification";

/**
 * Traffic-light band indicator.
 *
 * The wording is deliberately about what the officer should do, not about the
 * traveller. "High risk" describes the document's evidence, and the subtitle
 * always says the decision remains the officer's.
 */

const BAND_STYLES: Record<RiskBand, { ring: string; dot: string; text: string; bg: string }> = {
  clear: {
    ring: "ring-emerald-500/30",
    dot: "bg-emerald-400",
    text: "text-emerald-300",
    bg: "bg-emerald-500/10",
  },
  review: {
    ring: "ring-amber-500/30",
    dot: "bg-amber-400",
    text: "text-amber-300",
    bg: "bg-amber-500/10",
  },
  high_risk: {
    ring: "ring-red-500/30",
    dot: "bg-red-400",
    text: "text-red-300",
    bg: "bg-red-500/10",
  },
};

const BAND_SUBTITLE: Record<RiskBand, string> = {
  clear: "No tamper indicators found — the officer retains the final decision",
  review: "Officer review recommended before clearing",
  high_risk: "Recommend secondary inspection before the traveller is cleared",
};

interface Props {
  band: RiskBand;
  score: number;
  compact?: boolean;
}

export default function RiskBadge({ band, score, compact = false }: Props) {
  const style = BAND_STYLES[band];

  if (compact) {
    return (
      <span
        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ${style.bg} ${style.ring} ${style.text}`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
        {BAND_LABELS[band]}
      </span>
    );
  }

  return (
    <div className={`rounded-xl p-5 ring-1 ${style.bg} ${style.ring}`}>
      <div className="flex items-baseline gap-3">
        <span className={`h-2.5 w-2.5 rounded-full ${style.dot}`} />
        <h2 className={`text-2xl font-semibold tracking-tight ${style.text}`}>
          {BAND_LABELS[band]}
        </h2>
        <span className="ml-auto font-mono text-sm text-slate-400">
          risk {score.toFixed(2)}
        </span>
      </div>
      <p className="mt-2 pl-[22px] text-sm text-slate-300">{BAND_SUBTITLE[band]}</p>
      <p className="mt-3 pl-[22px] text-xs text-slate-500">
        This score describes the evidence found on the document, not the traveller.
        The system does not accept or reject on its own.
      </p>
    </div>
  );
}
