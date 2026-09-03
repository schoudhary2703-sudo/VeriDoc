import { useCallback, useEffect, useState } from "react";
import { ApiError, getAuditLog } from "../api/client";
import RiskBadge from "../components/RiskBadge";
import type { AuditEntry } from "../types/verification";
import { CHECK_LABELS } from "../types/verification";

/**
 * The audit trail: what the system recommended, and what the officer decided.
 *
 * Both are shown side by side on purpose. The record has to be able to answer
 * "the model said review — what did the human do?", which is the question that
 * matters when a decision is challenged after the fact.
 */

function relativeTime(iso: string): string {
  const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

const ACTION_STYLE: Record<string, string> = {
  cleared: "text-emerald-300",
  referred: "text-amber-300",
  escalated: "text-sky-300",
};

export default function AuditLog() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      setEntries(await getAuditLog(100));
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load the audit log.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const id = setInterval(load, 10_000);
    return () => clearInterval(id);
  }, [load]);

  if (loading) {
    return <p className="text-sm text-slate-500">Loading audit log…</p>;
  }

  if (error) {
    return (
      <p className="rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-300 ring-1 ring-red-500/25">
        {error}
      </p>
    );
  }

  if (!entries.length) {
    return (
      <div className="rounded-xl border border-dashed border-slate-800 p-10 text-center">
        <p className="text-sm text-slate-500">
          No verifications recorded yet. Every run of the pipeline appends an entry
          here, whether or not an officer decision follows.
        </p>
      </div>
    );
  }

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/40">
      <header className="flex items-baseline justify-between border-b border-slate-800 px-5 py-3.5">
        <h2 className="text-sm font-medium text-slate-200">Audit log</h2>
        <span className="text-xs text-slate-500">
          {entries.length} entr{entries.length === 1 ? "y" : "ies"} · refreshes every 10s
        </span>
      </header>

      <ul className="divide-y divide-slate-800/70">
        {entries.map((entry) => (
          <li key={entry.verification_id} className="px-5 py-4">
            <div className="flex flex-wrap items-center gap-3">
              <RiskBadge band={entry.band} score={entry.score} compact />

              <span className="font-mono text-sm text-slate-300">
                {entry.document_number ?? "no document number"}
              </span>

              <span className="text-xs text-slate-600">
                {relativeTime(entry.recorded_at)}
              </span>

              <span className="ml-auto text-xs text-slate-600">
                {(entry.processing_time_ms / 1000).toFixed(1)}s
              </span>
            </div>

            {(entry.failed_checks.length > 0 || entry.weak_checks.length > 0) && (
              <p className="mt-2 text-xs text-slate-500">
                {entry.failed_checks.length > 0 && (
                  <span className="text-red-400">
                    Failed: {entry.failed_checks.map((c) => CHECK_LABELS[c] ?? c).join(", ")}
                  </span>
                )}
                {entry.failed_checks.length > 0 && entry.weak_checks.length > 0 && " · "}
                {entry.weak_checks.length > 0 && (
                  <span className="text-amber-400">
                    Weak: {entry.weak_checks.map((c) => CHECK_LABELS[c] ?? c).join(", ")}
                  </span>
                )}
              </p>
            )}

            <p className="mt-2 text-xs">
              {entry.officer_action ? (
                <span className="text-slate-400">
                  Officer{" "}
                  <span className="font-medium text-slate-300">{entry.officer_id}</span>{" "}
                  <span className={ACTION_STYLE[entry.officer_action] ?? ""}>
                    {entry.officer_action}
                  </span>{" "}
                  this traveller
                  {entry.decided_at && ` · ${relativeTime(entry.decided_at)}`}
                  {entry.officer_note && (
                    <span className="mt-1 block italic text-slate-500">
                      “{entry.officer_note}”
                    </span>
                  )}
                </span>
              ) : (
                <span className="text-slate-600">Awaiting officer decision</span>
              )}
            </p>
          </li>
        ))}
      </ul>

      <footer className="border-t border-slate-800 px-5 py-3">
        <p className="text-xs text-slate-600">
          Entries are append-only. No images and no personal fields beyond the
          document number are stored.
        </p>
      </footer>
    </section>
  );
}
