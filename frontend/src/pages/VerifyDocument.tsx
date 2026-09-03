import { useState } from "react";
import { ApiError, submitDecision, verifyDocument } from "../api/client";
import CaseHeader from "../components/CaseHeader";
import DocumentUpload from "../components/DocumentUpload";
import EvidencePanel from "../components/EvidencePanel";
import LiveCameraCapture from "../components/LiveCameraCapture";
import RiskBadge from "../components/RiskBadge";
import type { OfficerAction, VerifyResponse } from "../types/verification";

const OFFICER_ID = "BSF-2291";

const DEMO_SAMPLES = [
  { file: "specimen_passport_genuine.png", label: "Genuine specimen" },
  { file: "specimen_passport_tampered_dob.png", label: "Tampered date of birth" },
] as const;

const ACTIONS: { action: OfficerAction; label: string; className: string }[] = [
  {
    action: "cleared",
    label: "Clear traveller",
    className: "bg-emerald-600 hover:bg-emerald-500 text-white",
  },
  {
    action: "referred",
    label: "Refer for secondary inspection",
    className: "bg-amber-600 hover:bg-amber-500 text-white",
  },
  {
    action: "escalated",
    label: "Escalate to supervisor",
    className: "bg-slate-700 hover:bg-slate-600 text-slate-100",
  },
];

function StageTimings({ timings, total }: { timings: Record<string, number>; total: number }) {
  const entries = Object.entries(timings).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
      <h3 className="mb-3 text-xs font-medium uppercase tracking-wide text-slate-500">
        Stage timings
      </h3>
      <ul className="space-y-1.5">
        {entries.map(([stage, ms]) => (
          <li key={stage} className="flex items-center gap-3 text-xs">
            <span className="w-28 shrink-0 text-slate-400">
              {stage.replace(/_/g, " ")}
            </span>
            <span className="h-1 flex-1 overflow-hidden rounded bg-slate-800">
              <span
                className="block h-full bg-sky-600/70"
                style={{ width: `${Math.max((ms / Math.max(total, 1)) * 100, 1)}%` }}
              />
            </span>
            <span className="w-16 shrink-0 text-right font-mono text-slate-500">
              {ms} ms
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-slate-600">
        Total {(total / 1000).toFixed(1)} s on CPU.
      </p>
    </div>
  );
}

function ExtractedFieldsCard({ result }: { result: VerifyResponse }) {
  const f = result.extracted_fields;
  const rows: [string, string | null][] = [
    ["Surname", f.surname],
    ["Given names", f.given_names],
    ["Document number", f.document_number],
    ["Date of birth", f.dob],
    ["Nationality", f.nationality],
    ["Sex", f.sex],
    ["Date of expiry", f.expiry_date],
  ];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
      <h3 className="mb-3 text-xs font-medium uppercase tracking-wide text-slate-500">
        Extracted fields
      </h3>
      <dl className="space-y-1.5">
        {rows.map(([label, value]) => (
          <div key={label} className="flex gap-3 text-sm">
            <dt className="w-36 shrink-0 text-slate-500">{label}</dt>
            <dd className="font-medium text-slate-200">
              {value ?? <span className="text-slate-600">—</span>}
            </dd>
          </div>
        ))}
      </dl>

      {result.mrz_check.raw_lines.length > 0 && (
        <div className="mt-4 border-t border-slate-800 pt-3">
          <p className="mb-2 text-xs uppercase tracking-wide text-slate-500">
            Machine-readable zone
          </p>
          <pre className="overflow-x-auto whitespace-pre font-mono text-[11px] leading-relaxed text-slate-400">
            {result.mrz_check.raw_lines.join("\n")}
          </pre>
          <p className="mt-2 text-xs text-slate-500">
            {result.mrz_check.checks.filter((c) => c.passed).length} of{" "}
            {result.mrz_check.checks.length} check digits recompute correctly.
          </p>
        </div>
      )}
    </div>
  );
}

export default function VerifyDocument() {
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [faceBlob, setFaceBlob] = useState<Blob | null>(null);
  const [fast, setFast] = useState(true);

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [note, setNote] = useState("");
  const [decision, setDecision] = useState<OfficerAction | null>(null);
  const [decisionError, setDecisionError] = useState<string | null>(null);

  const loadDemo = async (file: string, label: string) => {
    setError(null);
    try {
      const resp = await fetch(`/samples/${file}`);
      if (!resp.ok) throw new Error(`${resp.status}`);
      const blob = await resp.blob();
      setDocumentFile(new File([blob], file, { type: "image/png" }));
      setResult(null);
      setDecision(null);
    } catch {
      setError(`Could not load the ${label.toLowerCase()} sample.`);
    }
  };

  const run = async () => {
    if (!documentFile) return;
    setRunning(true);
    setError(null);
    setResult(null);
    setDecision(null);
    setDecisionError(null);

    try {
      const response = await verifyDocument({
        documentImage: documentFile,
        liveFaceImage: faceBlob,
        fast,
      });
      setResult(response);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Verification failed unexpectedly.",
      );
    } finally {
      setRunning(false);
    }
  };

  const reset = () => {
    setDocumentFile(null);
    setFaceBlob(null);
    setResult(null);
    setError(null);
    setNote("");
    setDecision(null);
    setDecisionError(null);
  };

  const decide = async (action: OfficerAction) => {
    if (!result) return;
    setDecisionError(null);
    try {
      await submitDecision(result.verification_id, action, OFFICER_ID, note);
      setDecision(action);
    } catch (err) {
      setDecisionError(
        err instanceof ApiError ? err.message : "Could not record the decision.",
      );
    }
  };

  return (
    <>
      <CaseHeader
        result={result}
        running={running}
        hasDocument={documentFile !== null}
        onReset={reset}
      />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
      {/* Capture column */}
      <div className="space-y-5">
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <h2 className="mb-4 text-sm font-medium text-slate-200">Captured sources</h2>

          <div className="space-y-5">
            <DocumentUpload
              label="Document scan"
              hint="Passport or ID card, JPEG or PNG, up to 25 MB"
              file={documentFile}
              onChange={setDocumentFile}
              disabled={running}
            />

            <LiveCameraCapture
              onCapture={setFaceBlob}
              captured={faceBlob}
              disabled={running}
            />
          </div>

          {/* Demo mode: load a curated specimen without a scanner attached.
              Both are synthetic SPECIMEN documents — no real document is ever
              used in this project. */}
          <div className="mt-5 border-t border-slate-800 pt-4">
            <p className="mb-2 text-xs uppercase tracking-wide text-slate-500">
              Demo documents
            </p>
            <div className="flex flex-wrap gap-2">
              {DEMO_SAMPLES.map((sample) => (
                <button
                  key={sample.file}
                  type="button"
                  disabled={running}
                  onClick={() => void loadDemo(sample.file, sample.label)}
                  className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 transition hover:border-slate-600 hover:text-slate-100 disabled:opacity-50"
                >
                  {sample.label}
                </button>
              ))}
            </div>
            <p className="mt-2 text-xs text-slate-600">
              Synthetic specimen documents. No real document is used anywhere in
              this project.
            </p>
          </div>

          <label className="mt-5 flex items-start gap-2.5 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={fast}
              onChange={(e) => setFast(e.target.checked)}
              disabled={running}
              className="mt-0.5"
            />
            <span>
              Fast mode — read only the machine-readable zone.
              <span className="block text-slate-600">
                Around ten times faster and still validates every check digit, but
                the printed fields are not read.
              </span>
            </span>
          </label>

          <button
            type="button"
            onClick={run}
            disabled={!documentFile || running}
            className="mt-5 w-full rounded-lg bg-sky-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
          >
            {running ? "Analysing…" : "Run verification"}
          </button>

          {running && (
            <p className="mt-3 text-xs text-slate-500">
              OCR, forensics and face analysis run on CPU — this takes a few
              seconds.
            </p>
          )}
          {error && (
            <p className="mt-3 rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-300 ring-1 ring-red-500/25">
              {error}
            </p>
          )}
        </section>

        {result && (
          <StageTimings
            timings={result.stage_timings_ms}
            total={result.processing_time_ms}
          />
        )}
      </div>

      {/* Result column */}
      <div className="space-y-5">
        {!result && !running && (
          <div className="flex h-full min-h-64 items-center justify-center rounded-xl border border-dashed border-slate-800 p-10 text-center">
            <p className="max-w-sm text-sm text-slate-500">
              Upload a document to begin. Every check reports its own outcome and
              the reasoning behind it — the system recommends, the officer decides.
            </p>
          </div>
        )}

        {result && (
          <>
            <RiskBadge band={result.verdict.band} score={result.verdict.score} />

            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
              <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                System recommendation
              </h3>
              <p className="text-sm leading-relaxed text-slate-200">
                {result.verdict.recommendation}
              </p>
            </div>

            <EvidencePanel evidence={result.verdict.evidence} />

            <ExtractedFieldsCard result={result} />

            {/* Officer decision */}
            <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
              <h3 className="text-sm font-medium text-slate-200">Officer decision</h3>
              <p className="mt-1 text-xs text-slate-500">
                Recorded against {OFFICER_ID} with a timestamp and written to the
                audit log.
              </p>

              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Notes — physical inspection findings, recapture attempts, anything the next officer should know."
                rows={3}
                disabled={decision !== null}
                className="mt-3 w-full rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-600 focus:outline-none disabled:opacity-60"
              />

              {decision ? (
                <p className="mt-3 rounded-lg bg-emerald-500/10 px-3 py-2 text-sm text-emerald-300 ring-1 ring-emerald-500/25">
                  Decision recorded: {decision}. This is now in the audit log.
                </p>
              ) : (
                <div className="mt-3 flex flex-wrap gap-2">
                  {ACTIONS.map(({ action, label, className }) => (
                    <button
                      key={action}
                      type="button"
                      onClick={() => decide(action)}
                      className={`rounded-lg px-3.5 py-2 text-sm font-medium transition ${className}`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              )}

              {decisionError && (
                <p className="mt-3 text-xs text-red-400">{decisionError}</p>
              )}
            </section>
          </>
        )}
      </div>
    </div>
    </>
  );
}
