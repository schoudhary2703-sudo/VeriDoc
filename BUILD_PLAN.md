# DocSentry — Build Plan

**Project:** AI-Based Fake Identity & Document Screening System
**SIH Problem Statement:** SIH26188 · Ministry of Home Affairs · Software · Smart Automation

This file is written to be fed directly to Claude Code (or any coding agent) to scaffold and build the project phase by phase. Drop it at the repo root — as `CLAUDE.md` if you want Claude Code to auto-load it as project context on every session, or keep it as `BUILD_PLAN.md` and paste the relevant phase section in when you start each one.

**How to use this doc with Claude Code:**
1. `git init docsentry && cd docsentry`, then paste this whole file in as `BUILD_PLAN.md` (or `CLAUDE.md`).
2. Open Claude Code in that folder.
3. Start with the prompt at the end of **Phase 0** below. Do not ask it to build everything at once — feed one phase's prompt at a time, let it finish, verify against that phase's "Definition of Done," then move to the next.

---

## 1 · What we're building, in one paragraph

A layered document-and-identity verification pipeline for border checkpoints. A document image (and optionally a live face capture) goes in; the system runs OCR/MRZ extraction, image-forensics tamper detection, face matching, and a database cross-check in sequence; a risk-scoring stage combines all four signals into one explainable verdict (clear / review / high-risk) with the specific evidence behind it; an officer-facing dashboard shows that verdict and lets a human make the final call. Nothing is fully automated — this is a decision-support tool, not an auto-reject system.

---

## 2 · Architecture

```
                    ┌─────────────────────┐
  document image ─▶ │ 1. Capture &        │
  live face photo─▶ │    Pre-processing   │
                    └──────────┬──────────┘
                               │ clean, cropped image
                 ┌─────────────┼─────────────┐
                 ▼                           ▼
       ┌───────────────────┐     ┌───────────────────────┐
       │ 2. OCR & MRZ       │     │ 3. Image Forensics     │
       │    Extraction      │     │    Engine               │
       └─────────┬──────────┘     └───────────┬────────────┘
                 │ fields + MRZ check                │ tamper evidence
                 ▼                                    │
       ┌───────────────────┐                          │
       │ 4. Face Match &    │                          │
       │    Liveness        │                          │
       └─────────┬──────────┘                          │
                 │ match confidence                     │
                 ▼                                      ▼
       ┌───────────────────┐              ┌──────────────────────┐
       │ 5. DB Cross-       │─────────────▶│ 6. Risk Scoring       │
       │    Verification     │              │    Engine             │
       └───────────────────┘              └───────────┬────────────┘
                                                        │ verdict object
                                                        ▼
                                          ┌──────────────────────────┐
                                          │ 7. Officer Review          │
                                          │    Dashboard (frontend)    │
                                          └───────────┬────────────────┘
                                                        │ officer decision
                                                        ▼
                                          ┌──────────────────────────┐
                                          │ 8. Audit & Sync Layer      │
                                          └──────────────────────────┘
```

Every stage after pre-processing produces **evidence, not just a pass/fail** — that's what stage 6 scores and what stage 7 renders. Nothing downstream ever sees a bare probability without the reasoning behind it.

| # | Module | Backend location | Responsibility |
|---|--------|-------------------|-----------------|
| 1 | Capture & Pre-processing | `backend/app/modules/preprocessing/` | Deskew, crop to document boundary, glare/noise cleanup |
| 2 | OCR & MRZ Extraction | `backend/app/modules/ocr_mrz/` | Field OCR, MRZ parse + checksum cross-check |
| 3 | Image Forensics | `backend/app/modules/forensics/` | ELA, copy-move/splicing detection, CNN tamper classifier |
| 4 | Face Match & Liveness | `backend/app/modules/face/` | Document-photo vs. live-photo match, passive liveness |
| 5 | DB Cross-Verification | `backend/app/modules/db_crosscheck/` | Lookup against seeded mock records/watchlist |
| 6 | Risk Scoring | `backend/app/core/risk_scoring.py` | Combine all signals into one explainable verdict |
| 7 | Officer Dashboard | `frontend/` | Upload, live verdict, evidence panel, accept/escalate |
| 8 | Audit & Sync | `backend/app/modules/audit/` | Immutable log of every check + officer decision |

---

## 3 · Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Backend / API | Python 3.11, FastAPI, Pydantic v2 | Async, easy to containerize |
| OCR | PaddleOCR (primary), Tesseract (fallback) | Keep both behind one interface so the engine is swappable |
| MRZ parsing | Custom rules-based parser + checksum validator | MRZ format is a fixed spec — no ML needed here |
| Image forensics | OpenCV (ELA, noise/compression analysis) + a fine-tuned EfficientNet-B0 (via `timm`/PyTorch) | Start classical (explainable), add the CNN once a baseline exists |
| Face match & liveness | InsightFace (ArcFace embeddings) | Runs fine on CPU for a demo-scale dataset |
| Risk scoring | Rules-weighted ensemble to start; optional LightGBM once labelled evidence accumulates | Judges reward explainability — don't hide behind a black-box model early |
| Database | PostgreSQL 15 (SQLAlchemy + Alembic migrations) | Documents, verifications, audit log |
| Cache / session | Redis 7 | |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS | Fast to build, fast to demo |
| Deployment | Docker Compose (dev + demo) | Documented path to edge deployment in `docs/ARCHITECTURE.md` |
| Testing | Pytest (backend), Vitest + React Testing Library (frontend) | |

---

## 4 · Repository Structure

```
docsentry/
├── README.md
├── BUILD_PLAN.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── backend/
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── alembic.ini
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py
│   │   │   └── routes/
│   │   │       ├── verify.py          # POST /api/verify
│   │   │       ├── documents.py       # document CRUD
│   │   │       ├── audit.py           # GET /api/audit-log
│   │   │       └── health.py
│   │   ├── core/
│   │   │   ├── pipeline.py            # orchestrates stages 1-6
│   │   │   ├── risk_scoring.py
│   │   │   └── schemas.py             # Pydantic request/response models
│   │   ├── modules/
│   │   │   ├── preprocessing/
│   │   │   │   ├── __init__.py
│   │   │   │   └── normalize.py
│   │   │   ├── ocr_mrz/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ocr_engine.py
│   │   │   │   └── mrz_parser.py
│   │   │   ├── forensics/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ela.py
│   │   │   │   ├── copy_move.py
│   │   │   │   └── cnn_classifier.py
│   │   │   ├── face/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── face_match.py
│   │   │   │   └── liveness.py
│   │   │   ├── db_crosscheck/
│   │   │   │   ├── __init__.py
│   │   │   │   └── lookup.py
│   │   │   └── audit/
│   │   │       ├── __init__.py
│   │   │       └── logger.py
│   │   ├── models/                    # SQLAlchemy ORM
│   │   │   ├── document.py
│   │   │   ├── verification.py
│   │   │   └── audit_log.py
│   │   └── db/
│   │       ├── session.py
│   │       └── migrations/            # alembic versions
│   ├── ml/
│   │   ├── train_forensics_cnn.py
│   │   ├── evaluate_face_match.py
│   │   ├── data_prep/
│   │   │   ├── download_midv.py
│   │   │   ├── generate_synthetic_forgeries.py
│   │   │   └── split_dataset.py
│   │   └── notebooks/
│   │       └── eda.ipynb
│   └── tests/
│       ├── test_ocr_mrz.py
│       ├── test_forensics.py
│       ├── test_face_match.py
│       ├── test_risk_scoring.py
│       └── test_api.py
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── VerifyDocument.tsx
│   │   │   └── AuditLog.tsx
│   │   ├── components/
│   │   │   ├── EvidencePanel.tsx
│   │   │   ├── RiskBadge.tsx
│   │   │   ├── DocumentUpload.tsx
│   │   │   └── LiveCameraCapture.tsx
│   │   ├── api/
│   │   │   └── client.ts
│   │   └── types/
│   │       └── verification.ts
│   └── tests/
├── data/
│   ├── raw/                           # gitignored — MIDV-500/2020 downloads
│   ├── synthetic/                     # gitignored — generated forgeries
│   └── samples/                       # small, committed demo samples
└── docs/
    ├── ARCHITECTURE.md
    ├── API_CONTRACTS.md
    └── DATASETS.md
```

---

## 5 · Core API Contract

**`POST /api/verify`**

Request — `multipart/form-data`:
```
document_image: file (required)
live_face_image: file (optional — enables face-match stage)
```

Response — `200 OK`:
```json
{
  "verification_id": "uuid",
  "verdict": {
    "band": "clear | review | high_risk",
    "score": 0.0,
    "evidence": [
      {
        "stage": "forensics",
        "check": "photo_splice_detection",
        "passed": false,
        "detail": "Compression artifact mismatch around face boundary",
        "confidence": 0.83
      }
    ]
  },
  "extracted_fields": { "name": "...", "dob": "...", "document_number": "...", "expiry_date": "..." },
  "mrz_check": { "valid": true, "checksum_match": true },
  "face_match": { "match_score": 0.91, "liveness_passed": true },
  "db_crosscheck": { "found": true, "blacklisted": false, "status": "active" },
  "processing_time_ms": 1830
}
```

Keep every stage's output as its own object in the response — the frontend evidence panel renders directly off this shape, and it's what makes the verdict explainable instead of a bare score.

---

## 6 · Phase-by-Phase Build Plan

Each phase below has a goal, a task checklist, the files it touches, a Definition of Done, and a ready-to-paste **Claude Code prompt**. Work through them in order — each phase assumes the previous one's Definition of Done is met.

### Phase 0 — Scaffolding & Environment
**Goal:** A running skeleton: empty FastAPI app, empty React app, Postgres + Redis in Docker, health-check green end to end.

Tasks:
- [ ] Initialize backend (`FastAPI`, `pyproject.toml`, `requirements.txt`, `Dockerfile`)
- [ ] Initialize frontend (`Vite` + React + TS + Tailwind)
- [ ] `docker-compose.yml` wiring backend, frontend, Postgres, Redis
- [ ] `.env.example` with all required variables (see Section 7)
- [ ] `GET /api/health` returns `{"status": "ok"}`; frontend calls it on load and shows a green/red badge
- [ ] Alembic initialized, first empty migration applied

**Definition of Done:** `docker compose up` brings up all four services; the frontend homepage shows a live "backend: connected" status.

**Claude Code prompt:**
> Read `BUILD_PLAN.md` in this repo. Implement Phase 0 only: scaffold the backend (FastAPI) and frontend (React + TypeScript + Vite + Tailwind) per the repository structure in Section 4, wire up Docker Compose with Postgres and Redis, add a `/api/health` endpoint, and have the frontend homepage display live backend connection status. Do not implement any of the verification modules yet. Stop and summarize when the Definition of Done in Section 6, Phase 0 is met.

---

### Phase 1 — OCR & MRZ Pipeline
**Goal:** Given a document image, extract text fields and validate the MRZ checksum.

Tasks:
- [ ] `modules/preprocessing/normalize.py` — deskew, crop, glare reduction (OpenCV)
- [ ] `modules/ocr_mrz/ocr_engine.py` — wrap PaddleOCR behind a small interface (`extract_text(image) -> list[Field]`)
- [ ] `modules/ocr_mrz/mrz_parser.py` — parse MRZ lines, verify checksum digits per the ICAO 9303 spec
- [ ] `core/schemas.py` — `ExtractedFields`, `MRZCheckResult` Pydantic models
- [ ] Unit tests against `data/samples/` (a handful of MIDV-style sample images, genuine + one with a tampered MRZ digit)

**Definition of Done:** A standalone script (and a test) that takes a sample image path and prints extracted fields + MRZ validity; correctly flags the one sample with a tampered checksum.

**Claude Code prompt:**
> Implement Phase 1 from `BUILD_PLAN.md`: the pre-processing and OCR/MRZ modules. Use PaddleOCR for text extraction and write a rules-based MRZ parser with ICAO 9303 checksum validation. Add the Pydantic schemas from Section 5. Write pytest tests against sample images in `data/samples/`. Do not wire this into the API yet — a standalone callable pipeline function is enough for this phase.

---

### Phase 2 — Image Forensics Engine
**Goal:** Detect photo substitution, splicing, and re-compression artifacts.

Tasks:
- [ ] `modules/forensics/ela.py` — Error Level Analysis
- [ ] `modules/forensics/copy_move.py` — copy-move/splice detection (classical CV first)
- [ ] `ml/data_prep/download_midv.py` — fetch MIDV-500/2020 samples
- [ ] `ml/data_prep/generate_synthetic_forgeries.py` — programmatically apply photo-splice, DOB-digit-edit, stamp-overlay, re-compression to clean templates
- [ ] `ml/train_forensics_cnn.py` — fine-tune EfficientNet-B0 on the synthetic + MIDV set
- [ ] `modules/forensics/cnn_classifier.py` — load the trained model, expose `classify(image) -> ForensicsResult`
- [ ] Report precision/recall **per tamper type**, not one blended number

**Definition of Done:** Forensics module correctly flags each of the four synthetic tamper types on a held-out test split, each with its own evidence string (not just "tampered: true").

**Claude Code prompt:**
> Implement Phase 2 from `BUILD_PLAN.md`: the image forensics engine. Start with classical CV checks (ELA, copy-move detection) in `modules/forensics/`, then build the synthetic forgery data-generation script and train an EfficientNet-B0 classifier per Section 3's tech stack. Wire the trained model into `cnn_classifier.py`. Report per-tamper-type precision/recall in a short results file. Do not touch the API layer yet.

---

### Phase 3 — Face Match & Liveness
**Goal:** Compare the document photo against a live capture; detect spoofing.

Tasks:
- [ ] `modules/face/face_match.py` — InsightFace/ArcFace embedding extraction + cosine similarity scoring
- [ ] `modules/face/liveness.py` — passive liveness classifier (texture/reflection cues)
- [ ] Tune the match threshold against a low-quality-photo subset (per the risk register in the project blueprint — this is a known failure mode)
- [ ] Report equal-error-rate at the chosen threshold

**Definition of Done:** Given a genuine document+face pair, the module returns a high match score; given a mismatched pair, a low one; a printed photo held up to the camera is correctly flagged by liveness.

**Claude Code prompt:**
> Implement Phase 3 from `BUILD_PLAN.md`: face match and liveness. Use InsightFace (ArcFace embeddings) for matching and a lightweight passive-liveness classifier. Tune and report the equal-error-rate at your chosen threshold, including performance on deliberately low-quality document photos. Keep this module standalone and testable independent of the API.

---

### Phase 4 — Risk Scoring & Pipeline Integration
**Goal:** Wire stages 1–5 together behind one orchestrator that returns the verdict object from Section 5.

Tasks:
- [ ] `modules/db_crosscheck/lookup.py` — seeded mock Postgres table of sample records + a blacklist flag
- [ ] `core/risk_scoring.py` — rules-weighted ensemble combining OCR/MRZ, forensics, face-match, and DB-check signals into `band` + `score` + `evidence[]`
- [ ] `core/pipeline.py` — orchestrates all five stages in sequence, assembles the final response object
- [ ] `api/routes/verify.py` — `POST /api/verify` per the contract in Section 5

**Definition of Done:** A `curl` request with a genuine sample document image returns a `"clear"` verdict; the same request with a tampered sample returns `"high_risk"` with the correct evidence entry.

**Claude Code prompt:**
> Implement Phase 4 from `BUILD_PLAN.md`: integrate the modules built in Phases 1–3 into `core/pipeline.py`, add the mock database cross-check, build the rules-weighted risk-scoring ensemble in `core/risk_scoring.py`, and expose it via `POST /api/verify` matching the exact response contract in Section 5. Add an integration test that posts a genuine and a tampered sample document and asserts the expected verdict band for each.

---

### Phase 5 — Officer Dashboard (Frontend)
**Goal:** A working UI: upload a document (+ optional live photo), see the verdict and evidence live.

Tasks:
- [ ] `api/client.ts` — typed fetch wrapper for `/api/verify`, matching `types/verification.ts`
- [ ] `components/DocumentUpload.tsx`, `components/LiveCameraCapture.tsx`
- [ ] `components/RiskBadge.tsx` — traffic-light band indicator
- [ ] `components/EvidencePanel.tsx` — renders the `evidence[]` array, highlighting failed checks
- [ ] `pages/VerifyDocument.tsx` — the main demo flow
- [ ] `pages/AuditLog.tsx` — reads from `GET /api/audit-log`

**Definition of Done:** End-to-end in the browser: upload a sample document, see a live verdict with evidence within a few seconds, no console errors.

**Claude Code prompt:**
> Implement Phase 5 from `BUILD_PLAN.md`: the officer dashboard frontend. Build the upload flow, risk badge, and evidence panel components, wired to the live `POST /api/verify` endpoint from Phase 4. Match the response shape in Section 5 exactly in `types/verification.ts`. Keep the UI clean and demo-ready — this is what judges will watch live.

---

### Phase 6 — Audit Log & Hardening
**Goal:** Every verification and officer decision is durably logged; edge cases from earlier phases are fixed.

Tasks:
- [ ] `modules/audit/logger.py` + `models/audit_log.py` — immutable log entry per verification + officer action
- [ ] `api/routes/audit.py` — `GET /api/audit-log`
- [ ] Fix false-positive/negative edge cases surfaced in Phases 1–3 testing
- [ ] Add rate-limiting / basic input validation (file size/type checks) on `/api/verify`

**Definition of Done:** Every call to `/api/verify` produces a queryable audit-log row; the audit page in the frontend lists recent verifications.

**Claude Code prompt:**
> Implement Phase 6 from `BUILD_PLAN.md`: audit logging for every verification and officer decision, exposed via `GET /api/audit-log` and rendered in the frontend's Audit Log page. Then do a hardening pass — review the test results from Phases 1–3 and fix any false-positive/negative edge cases you find, and add basic input validation to the `/api/verify` endpoint.

---

### Phase 7 — Testing, Benchmarking & Dockerization
**Goal:** Full test suite green, accuracy benchmarks documented, one-command demo startup.

Tasks:
- [ ] Full pytest suite passing (`backend/tests/`)
- [ ] Frontend test suite passing (`frontend/tests/`)
- [ ] `docs/DATASETS.md` — final accuracy numbers (OCR/MRZ accuracy, forensics precision/recall per tamper type, face-match EER, end-to-end latency)
- [ ] `docker-compose.yml` finalized — `docker compose up` is the entire demo setup, no manual steps
- [ ] `README.md` — quickstart, architecture summary, screenshot

**Definition of Done:** A clean machine can clone the repo, run `docker compose up`, and have a working demo in under five minutes with no manual configuration.

**Claude Code prompt:**
> Implement Phase 7 from `BUILD_PLAN.md`: fill out the test suites, finalize `docker-compose.yml` for a true one-command startup, and write `docs/DATASETS.md` documenting final accuracy numbers per Section 9 of the project blueprint (OCR/MRZ accuracy, per-tamper-type forensics precision/recall, face-match EER, end-to-end latency). Write a clear `README.md` with quickstart instructions.

---

### Phase 8 — Demo Polish
**Goal:** Rehearsal-ready for the grand finale.

Tasks:
- [ ] Seed 2–3 curated demo documents (one clean, one with each major tamper type) into `data/samples/`
- [ ] Add a "demo mode" toggle that pre-loads these without needing a live camera/scanner
- [ ] Polish loading states, error states, and the evidence panel's wording (see the pitch-strategy section of the project blueprint for exact language judges respond to)
- [ ] Record a fallback demo video in case of live A/V issues on stage

**Definition of Done:** The full narrated demo (from `docx` blueprint, Section 10) runs start to finish without a single rough edge.

**Claude Code prompt:**
> Implement Phase 8 from `BUILD_PLAN.md`: add a demo mode with curated sample documents covering each tamper type, polish all loading/error states and evidence-panel copy, and make sure the full pitch demo flow runs smoothly end to end.

---

## 7 · Environment Variables (`.env.example`)

```
# Backend
DATABASE_URL=postgresql://docsentry:docsentry@db:5432/docsentry
REDIS_URL=redis://redis:6379/0
ENV=development

# ML
FORENSICS_MODEL_PATH=./ml/checkpoints/forensics_cnn.pt
FACE_MATCH_THRESHOLD=0.55
LIVENESS_THRESHOLD=0.5

# Frontend
VITE_API_BASE_URL=http://localhost:8000
```

---

## 8 · Testing Strategy

- **Unit tests** per module (Phases 1–3) — test against `data/samples/`, never against unreviewed real documents.
- **Integration tests** (Phase 4) — full pipeline, asserting verdict bands on known genuine/tampered samples.
- **Accuracy benchmarks** (Phase 7) — reported per tamper type, not blended, per the project blueprint's Section 9.
- **No real personal documents, ever** — all training/test data is MIDV-500/2020 (public benchmark) or synthetically generated from clean templates.

---

## 9 · Reference

This build plan implements the architecture and phase timeline from the SIH26188 project blueprint (`SIH26188_Project_Blueprint.docx`) — see that document for the problem-statement background, risk register, success metrics, and grand-finale pitch strategy. This file is the engineering companion: what to actually build, in what order, and how to hand each step to a coding agent.
