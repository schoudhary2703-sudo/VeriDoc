import { useEffect, useState } from "react";
import { getHealth } from "./api/client";
import AuditLog from "./pages/AuditLog";
import VerifyDocument from "./pages/VerifyDocument";
import type { HealthResponse } from "./types/health";

type Page = "verify" | "audit";

type ConnState =
  | { kind: "checking" }
  | { kind: "connected"; health: HealthResponse }
  | { kind: "error"; message: string };

const POLL_MS = 15_000;

function ConnectionPill({ state }: { state: ConnState }) {
  if (state.kind === "checking") {
    return (
      <span className="flex items-center gap-2 text-xs text-slate-500">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
        connecting…
      </span>
    );
  }

  if (state.kind === "error") {
    return (
      <span
        className="flex items-center gap-2 text-xs text-red-400"
        title={state.message}
      >
        <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
        backend unreachable
      </span>
    );
  }

  const deps = state.health.dependencies;
  const degraded = Object.values(deps).filter((v) => v !== "ok").length;

  return (
    <span className="flex items-center gap-2 text-xs text-slate-400">
      <span
        className={`h-1.5 w-1.5 rounded-full ${degraded ? "bg-amber-400" : "bg-emerald-400"}`}
      />
      {degraded ? `${degraded} dependency degraded` : "all engines online"}
    </span>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("verify");
  const [conn, setConn] = useState<ConnState>({ kind: "checking" });

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const health = await getHealth();
        if (!cancelled) setConn({ kind: "connected", health });
      } catch (err) {
        if (!cancelled)
          setConn({ kind: "error", message: (err as Error).message });
      }
    };

    void check();
    const id = setInterval(check, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const tabs: { id: Page; label: string }[] = [
    { id: "verify", label: "Verify" },
    { id: "audit", label: "Audit log" },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/85 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-3 px-6 py-3">
          <div className="flex items-baseline gap-2.5">
            <span className="flex h-6 w-6 items-center justify-center rounded bg-sky-600 text-sm font-bold text-white">
              V
            </span>
            <span className="text-base font-semibold tracking-tight">VeriDoc</span>
            <span className="text-xs text-slate-500">SIH26188 · MHA</span>
          </div>

          <nav className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setPage(tab.id)}
                className={`rounded-lg px-3 py-1.5 text-sm transition ${
                  page === tab.id
                    ? "bg-slate-800 text-slate-100"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-5">
            <ConnectionPill state={conn} />
            <span className="text-xs text-slate-500">
              Insp. R. Nair · <span className="text-slate-600">BSF-2291</span>
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-7">
        {page === "verify" ? <VerifyDocument /> : <AuditLog />}
      </main>

      <footer className="mx-auto max-w-7xl px-6 pb-8">
        <p className="text-xs leading-relaxed text-slate-600">
          Decision support for a human officer. VeriDoc never accepts or rejects a
          traveller on its own. Record cross-checks run against a simulated local
          record set — this prototype has no connection to any live watchlist.
        </p>
      </footer>
    </div>
  );
}
