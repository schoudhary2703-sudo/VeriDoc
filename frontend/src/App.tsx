import { useEffect, useState } from "react";
import { getHealth } from "./api/client";
import type { HealthResponse } from "./types/health";

type ConnState =
  | { kind: "checking" }
  | { kind: "connected"; health: HealthResponse }
  | { kind: "error"; message: string };

const POLL_MS = 5000;

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span
      className={`inline-block h-2.5 w-2.5 rounded-full ${
        ok ? "bg-emerald-500" : "bg-red-500"
      }`}
    />
  );
}

function BackendStatus({ state }: { state: ConnState }) {
  if (state.kind === "checking") {
    return (
      <div className="flex items-center gap-2 text-slate-400">
        <span className="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-amber-400" />
        <span className="text-sm">backend: checking…</span>
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="flex items-center gap-2 text-red-400">
        <StatusDot ok={false} />
        <span className="text-sm">backend: disconnected — {state.message}</span>
      </div>
    );
  }

  const { dependencies } = state.health;
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-emerald-400">
        <StatusDot ok />
        <span className="text-sm font-medium">backend: connected</span>
      </div>
      <div className="flex gap-4 pl-[18px] text-xs text-slate-400">
        <span className="flex items-center gap-1.5">
          <StatusDot ok={dependencies.database === "ok"} />
          postgres {dependencies.database === "ok" ? "" : `(${dependencies.database})`}
        </span>
        <span className="flex items-center gap-1.5">
          <StatusDot ok={dependencies.redis === "ok"} />
          redis {dependencies.redis === "ok" ? "" : `(${dependencies.redis})`}
        </span>
      </div>
    </div>
  );
}

export default function App() {
  const [state, setState] = useState<ConnState>({ kind: "checking" });

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const health = await getHealth();
        if (!cancelled) setState({ kind: "connected", health });
      } catch (err) {
        if (!cancelled) {
          setState({ kind: "error", message: (err as Error).message });
        }
      }
    };

    check();
    const id = setInterval(check, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-3xl px-6 py-16">
        <header className="mb-10">
          <h1 className="text-3xl font-semibold tracking-tight">VeriDoc</h1>
          <p className="mt-1 text-sm text-slate-400">
            AI-based fake identity &amp; document screening — SIH26188
          </p>
        </header>

        <section className="rounded-lg border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
            System status
          </h2>
          <BackendStatus state={state} />
        </section>

        <section className="mt-6 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
            Pipeline
          </h2>
          <p className="text-sm text-slate-400">
            Verification modules are not built yet — Phase 0 scaffolding only. The
            officer dashboard arrives in Phase 5.
          </p>
        </section>
      </div>
    </div>
  );
}
