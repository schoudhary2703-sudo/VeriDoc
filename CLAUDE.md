# VeriDoc

AI-Based Fake Identity & Document Screening System.
SIH Problem Statement **SIH26188** · Ministry of Home Affairs · Software · Smart Automation.

## Engineering plan

The full architecture, API contract, and phase-by-phase build plan live in
[BUILD_PLAN.md](BUILD_PLAN.md). Read it before starting work. Build strictly one
phase at a time and verify that phase's "Definition of Done" before moving on.

## Data strategy

[docs/DATA_STRATEGY.md](docs/DATA_STRATEGY.md) names the concrete dataset per
module and supersedes the vaguer data notes in `BUILD_PLAN.md`. Read it before
starting Phases 2 or 3. It changes the plan in four ways worth knowing up front:

- **Forensics training data is IDNet (CC-BY-4.0, Hugging Face)** plus specimen
  documents we generate ourselves with DocXPand for Indian formats -- not
  hand-rolled forgeries from scratch as `BUILD_PLAN.md` Phase 2 implies.
- **SIDTD is the held-out cross-dataset validation set.** If forensics accuracy
  collapses on SIDTD, the model learned IDNet's generation signature rather than
  general tamper cues. Report both numbers, always.
- **No pixel-level tamper localization.** Tampered regions on ID documents are
  0.27-4.17% of the image, and SOTA detectors score near-zero on pixel-level
  localization. Forensics output is field/region-level classification plus a
  coarse bounding box. The evidence panel must not promise a precise heatmap.
- **Face match needs no training** -- InsightFace `buffalo_l` pretrained, tune
  the threshold only. Liveness does need a trained classifier (CelebA-Spoof),
  and every public anti-spoofing dataset is **research-licensed only**. Say so
  in the README and the submission; do not imply a commercially deployable model.

## Naming

`BUILD_PLAN.md` was written with the working title *DocSentry*. The project is
named **VeriDoc**. Wherever the plan says `docsentry`, use `veridoc`:

- Repo root is `VeriDoc/` itself — there is no nested project folder. The tree in
  Section 4 of the plan maps directly onto this directory (`backend/`,
  `frontend/`, `data/`, `docs/` at the root).
- Python package / module naming, Docker Compose service names, image names,
  Postgres database name, user, and password: `veridoc`.
- So `DATABASE_URL=postgresql://veridoc:veridoc@db:5432/veridoc`.

Everything else in `BUILD_PLAN.md` — architecture, tech stack, API contract,
phase gates — applies as written.

## Non-negotiables

- **No real personal documents, ever.** All training and test data is MIDV-500/2020
  (public benchmark) or synthetically generated from clean templates.
- **Every stage emits evidence, not a bare pass/fail.** The `evidence[]` array in
  the `POST /api/verify` response (Section 5) is the contract the risk scorer and
  the frontend both depend on.
- **Explainable first.** Classical CV before the forensics CNN; rules-weighted
  scoring before any learned ensemble.
- **Accuracy is reported per tamper type**, never as one blended number.
- This is decision support for a human officer, never an auto-reject system.
