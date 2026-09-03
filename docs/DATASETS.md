# Datasets and Measured Accuracy

Phase 7 deliverable. Every figure here was produced by a script in this repo and
can be reproduced; the command is given beside each one.

**Read the caveats.** Some numbers are strong, several are weak, and one whole
component failed. All of them are here, because a per-type accuracy table whose
weak entries have been quietly removed is worth less than no table at all.

---

## 1 · Datasets used

| Dataset | Role | Licence | In repo? |
|---|---|---|---|
| **FantasyID** (Idiap, IJCB 2025) | Primary evaluation. 362 fantasy ID cards printed on a card printer and re-captured with an iPhone 15 Pro, a Huawei Mate 30, and a Kyocera scanner. Official train/test split whose halves use **different card templates**. | CC-BY-4.0 / CC0 via [Zenodo](https://zenodo.org/records/17063366) | `data/raw/` (gitignored, 2.5 GB) |
| **Generated specimens** | Phase 1 unit tests and the demo. Fully fabricated: fictional names, drawn placeholder portraits, SPECIMEN watermark. | Ours | `data/samples/` (committed) |
| **Synthetic forgeries** | Phase 2 smoke test. Four tamper types applied to a generated specimen. | Ours | `data/synthetic/` (gitignored, regenerable) |
| IDNet | Considered, **not used**: 124.93 GB total, and the machine had 32 GB free. | CC-BY-4.0 | no |
| CelebA-Spoof / OULU-NPU | Required for liveness, **not obtained**. Research-licence only. | research-only | no |

**No real personal document was used at any stage.** One nuance to state
accurately rather than overclaim: FantasyID's cards are fictional, but the
*faces* printed on the bonafide cards are real people from consented research
datasets (AMFD, Face Research Lab London, HQ-WMCA). "No real documents" holds
exactly as written; "no real faces" does not.

---

## 2 · Forensics — held-out accuracy

Measured on FantasyID's official test split. Different templates from training,
real device captures, and **no threshold in the engine was tuned on any of it**.

```bash
cd backend && python -m ml.evaluate_fantasyid --limit 150
```

450 images (150 bonafide, 300 attack):

| Metric | Result |
|---|---|
| **False positives on genuine documents** | **0 / 150 — 0%** |
| Face-swap detection | **46%** (69/150) |
| Text-manipulation detection | **6%** (9/150) |
| Overall attack detection | 26% (78/300) |
| Mean analysis time | 2.2 s per image (CPU) |

### Per capture device

| Device | Detection |
|---|---|
| Huawei Mate 30 | 46% |
| iPhone 15 | 46% |
| Kyocera scanner | 8% |
| **iPhone 15 Pro** | **0%** |

Detection is strongly device-dependent, and that is a deployment risk worth
naming: a checkpoint standardised on the wrong capture hardware would get far
less from this pipeline than the headline number suggests. A single blended
figure would have hidden it completely.

### Where the 46% comes from

Two orthogonal signals, with **zero overlap** in what they catch:

| Signal | Face-swap detection | False positives |
|---|---|---|
| Classical checks (ELA + copy-move) | 29% | 0% |
| Intra-document face consistency | 11% | 0% |
| **Combined** | **46%** | **0%** |

The face check exploits a counter-intuitive property: a generative swap
re-renders *both* the main portrait and the ghost image from one model, so the
two become **unnaturally alike**. Genuine cards score 0.729 median cosine
similarity between their two portraits; face-swapped cards score 0.844. AUC
0.839. Too much consistency is the tamper signal.

Thresholds are set at each check's **zero-false-positive** operating point, not
its accuracy-optimal one. The face check reaches 91.9% recall at a threshold that
also flags 38.5% of genuine documents — indefensible at a border, so recall was
traded away deliberately.

---

## 3 · The learned classifier failed

Three EfficientNet-B0 configurations were trained and evaluated. **All scored at
or below chance.** Full detail in [FORENSICS_CNN_ATTEMPTS.md](FORENSICS_CNN_ATTEMPTS.md).

| Configuration | Best macro-F1 | Trivial baseline |
|---|---|---|
| Whole image, 384 px | 0.306 | 0.439 — *worse than trivial* |
| 256 px native patches, same-image negatives | 0.519 | 0.360 |
| 256 px native patches, bonafide-only negatives | 0.461 | 0.392 |

The CNN ships **disabled** (`CHECK_WEIGHTS["cnn_classifier"] = 0.0`, no checkpoint
at the production path). Everything reported in this file comes from explainable
checks only.

---

## 4 · OCR and MRZ

```bash
cd backend && python scripts/run_ocr_mrz.py ../data/samples/specimen_passport_genuine.png
```

| Metric | Result |
|---|---|
| MRZ check-digit validation | Verified against the **published ICAO 9303 worked example**, not against itself |
| Genuine specimen | `VALID` — 5/5 check digits, exit 0 |
| DOB-tampered specimen | `INVALID` — names `date_of_birth` and `composite`, exit 1 |
| MRZ read latency | **~1.8 s** (band-cropped, upscaled 2×) |

Field-level OCR accuracy against MIDV-2020 is **not measured** — MIDV was never
downloaded. Reporting an OCR accuracy figure would mean inventing one.

---

## 5 · Face matching

| Capability | Status |
|---|---|
| Intra-document consistency | **Measured**, AUC 0.839 (section 2) |
| Document photo vs live capture | **Implemented, uncalibrated.** Threshold 0.40 is InsightFace's published default for `w600k_r50`. No EER is quoted because we have no document+live capture pairs to compute one from. |
| Liveness / anti-spoofing | **Implemented, uncalibrated, disabled.** Four passive cues are measured and reported; no pass or fail is claimed. See below. |

### Liveness — measured but not evaluated

`app/modules/face/liveness.py` measures four passive presentation-attack cues:
moire interference (screen replay), specular concentration (glossy print or
screen), micro-texture energy (print), and chroma spread (both).

**`LIVENESS_CALIBRATED = False`, and every result returns `passed=None`.** There
is no labelled data to fit thresholds against: CelebA-Spoof, OULU-NPU and
Replay-Attack are research-licence only and were not obtained, and every face in
FantasyID is already a printed-and-recaptured card -- a single class with nothing
to discriminate against. `passed=None` means "not established", never "passed".

A mechanism check confirms the cues respond, using synthetic print and replay
transforms. This tests the implementation, **not** its accuracy:

| Cue | baseline | simulated replay | simulated print | reading |
|---|---|---|---|---|
| `micro_texture_energy` | 28.2 | 24.1 | **16.6** | responds strongly, correct direction |
| `chroma_spread` | 6.11 | 6.11 | **2.79** | correct: unmoved by replay, collapses on print |
| `moire_interference` | 0.43 | 0.52 | **0.87** | **confounded** -- fires harder on print than replay |
| `specular_concentration` | 0.07 | 0.09 | 0.09 | barely moves at this crop size |

Two cues behave as designed. The moire cue is confounded by blur and would need
rework before it is trustworthy; specular concentration is close to useless.

**To calibrate without a licensed dataset**, photograph roughly 30 consented
faces directly and re-present the same faces as printed photos and on a screen,
using the same camera and lighting, then run:

```bash
cd backend && python -m ml.calibrate_liveness --data ../data/liveness
```

It fits each threshold at its zero-false-positive point on the live set and
reports per-cue AUC, refusing to recommend any cue below 0.70.

---

## 6 · End-to-end latency

```bash
docker compose up -d && curl -X POST "http://localhost:8000/api/verify?fast=true" -F "document_image=@data/samples/specimen_passport_genuine.png"
```

| Stage | Warm |
|---|---|
| Pre-processing | 40 ms |
| OCR + MRZ | 1,750 ms |
| Forensics (incl. face consistency) | 140–550 ms |
| Record cross-check + scoring | <5 ms |
| **Total** | **~1.9 s** |

First request after startup takes 45–60 s while the OCR and face models load.
Worth warming the engine before a demo.

---

## 7 · Reproducing everything

```bash
cd backend
python -m ml.data_prep.generate_specimen_documents     # demo specimens
python -m ml.data_prep.generate_synthetic_forgeries    # 4 tamper types + control
python -m ml.evaluate_forensics                        # synthetic smoke test
python -m ml.evaluate_fantasyid --limit 150            # held-out accuracy
python -m pytest tests/ -q                             # 68 tests
```

FantasyID must be downloaded to `data/raw/FantasyID` first (2.5 GB; verify the
MD5 published on Zenodo — the link drops mid-transfer, so fetch it with
`curl -C -` rather than a tool that cannot resume).
