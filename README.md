# VeriDoc

**AI-based fake identity & document screening for border checkpoints.**
SIH Problem Statement **SIH26188** · Ministry of Home Affairs

A document image goes in. OCR/MRZ extraction, image forensics, face matching and
a record cross-check each run independently, and a rules-weighted scorer combines
them into one verdict with the specific evidence behind it. An officer reads the
evidence and decides.

**VeriDoc never accepts or rejects a traveller on its own.**

---

## Quickstart

```bash
docker compose up -d
```

Then open **http://localhost:5173**. That is the entire setup — Postgres, Redis,
the API and the dashboard all come up together.

The first verification takes 45–60 seconds while the OCR and face models load;
every one after that takes about two seconds. Warm it before a demo:

```bash
curl -X POST "http://localhost:8000/api/verify?fast=true" \
     -F "document_image=@data/samples/specimen_passport_genuine.png"
```

The dashboard has two demo documents built in — a genuine specimen and one with a
tampered date of birth. Both are synthetic; **no real document is used anywhere in
this project.**

API docs: **http://localhost:8000/docs**

---

## What it actually detects

Measured on [FantasyID](https://zenodo.org/records/17063366)'s official held-out
test split — different card templates from training, real captures from three
devices, and no threshold in the engine tuned on any of it.

| | Result |
|---|---|
| **False positives on 150 genuine documents** | **0%** |
| Face-swap detection | **47%** |
| Text-manipulation detection | **8%** |
| Overall attack detection | 27% |
| Latency, warm | ~2 s per document (full pipeline, specimen-sized image) |

Detection is strongly device-dependent: 49% on Huawei and 48% on iPhone 15
captures, 8% on scanner captures, **0% on iPhone 15 Pro**. That is a deployment
risk, and a single blended accuracy figure would have hidden it.

Full numbers, and the commands that reproduce every one of them, in
[docs/DATASETS.md](docs/DATASETS.md).

### The interesting part

Face-swap detection comes from two signals whose detections have **zero overlap**:
the classical checks catch 29%, and an intra-document face check catches a further
18% that the classical checks miss entirely.

That second check works on a counter-intuitive principle. An ID card carries two
portraits — the main photograph and a smaller ghost image. A generative face swap
re-renders **both** from the same model, so they become *unnaturally alike*; on a
genuine card they are physically different renderings of one photograph and agree
well but imperfectly. Genuine cards score 0.729 median similarity between their
two portraits, face-swapped cards 0.844. **Too much consistency is the tamper
signal.**

---

## Architecture

```
document image ─┐
                ├─▶ 1. pre-processing ─┬─▶ 2. OCR + MRZ ────┐
live capture ───┘   deskew, crop,      │   ICAO 9303        │
                    glare              │   checksums        │
                                       │                    │
                                       ├─▶ 3. forensics ────┤
                                       │   ELA, copy-move,  ├─▶ 6. risk scoring
                                       │   face consistency │   noisy-OR over
                                       │                    │   weighted evidence
                                       ├─▶ 4. face match ───┤        │
                                       │   ArcFace + liveness│        ▼
                                       │                    │   verdict + evidence[]
                                       └─▶ 5. record check ─┘        │
                                           simulated lookup          ▼
                                                            7. officer dashboard
                                                                     │
                                                                     ▼
                                                            8. append-only audit
```

Stage 1 returns **two** images, and the distinction matters: glare reduction and
denoising rewrite exactly the pixel statistics the forensics engine measures, so
forensics reads the unenhanced crop while OCR reads the enhanced one.

Signals combine by **noisy-OR, not averaging**. A clean MRZ is not evidence that
the portrait was not substituted, and averaging let one passing check cancel
another's finding — measured, when it dropped held-out detection from 18% to 15%.

---

## Evidence, not a score

Every check reports its own outcome, its confidence, and its reasoning in a
sentence an officer can read aloud. Evidence has **four** states, not two:

| State | Meaning |
|---|---|
| `pass` | The check ran and found nothing |
| `weak` | Borderline — worth a look, not a referral |
| `fail` | The check ran and found something |
| `not_applicable` | **The check could not run on this document** |

That last state is load-bearing. A document with no machine-readable zone, no
record match and no live capture can otherwise come back "clear" having had
almost nothing verified about it. When fewer than 60% of checks could assess a
document, the verdict says so explicitly and names what did not run.

---

## What does not work, stated plainly

- **Text manipulation is close to undetected (8%).** Diffusion-based inpainting
  does not touch the photograph, which is where our strongest signal lives.
- **The learned tamper classifier failed.** Three EfficientNet-B0 configurations
  scored at or below chance on held-out data (macro-F1 0.306 / 0.519 / 0.461
  against a trivial baseline of 0.439). It ships **disabled**, and the failures
  are documented in [docs/FORENSICS_CNN_ATTEMPTS.md](docs/FORENSICS_CNN_ATTEMPTS.md)
  rather than deleted.
- **Liveness is uncalibrated and disabled.** The cues are implemented and
  measured, but there is no labelled live-versus-spoof data to fit thresholds
  against — every public anti-spoofing dataset is research-licence only.
  `ml/calibrate_liveness.py` fits them from photographs a team can take
  themselves, in about twenty minutes.
- **Face-match threshold is not calibrated.** It uses InsightFace's published
  default because we have no document-plus-live-capture pairs, so no EER is
  quoted.
- **The record cross-check is simulated.** Four seeded rows. The API says so in
  every response (`source: "simulated local record set (no live watchlist
  access)"`), and there is a test asserting it. There is no connection to
  Interpol SLTD or any national system.

---

## Development

```bash
cd backend
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
python -m pytest tests/ -q                    # 89 tests

cd ../frontend
npm install && npm run dev
```

Regenerate the demo data and re-measure accuracy:

```bash
cd backend
python -m ml.data_prep.generate_specimen_documents     # demo specimens
python -m ml.data_prep.generate_synthetic_forgeries    # 4 tamper types + control
python -m ml.evaluate_fantasyid --limit 150            # held-out accuracy
```

The evaluation needs FantasyID in `data/raw/FantasyID` (2.5 GB, CC-BY-4.0).
Fetch it with `curl -C -` — the link drops mid-transfer and pip-style downloaders
that cannot resume will never finish it.

---

## Non-negotiables

- **No real personal documents, ever.** All data is public benchmark or
  synthetic. One nuance stated accurately rather than overclaimed: FantasyID's
  cards are fictional, but the faces printed on them are real people from
  consented research datasets.
- **Every stage emits evidence, not a pass/fail.**
- **Explainable first.** Classical CV before the CNN; rules-weighted scoring
  before any learned ensemble. Nothing that cannot justify itself ships enabled.
- **Accuracy is reported per tamper type**, never as one blended number.
- **Decision support for a human officer, never an auto-reject system.**

---

## Documentation

| File | Contents |
|---|---|
| [docs/DATASETS.md](docs/DATASETS.md) | Every measured number and how to reproduce it |
| [docs/FORENSICS_FANTASYID.md](docs/FORENSICS_FANTASYID.md) | Held-out accuracy, per attack type and device |
| [docs/FORENSICS_CNN_ATTEMPTS.md](docs/FORENSICS_CNN_ATTEMPTS.md) | Three failed CNN configurations and why each failed |
| [docs/DATA_STRATEGY.md](docs/DATA_STRATEGY.md) | Dataset choices and licensing |
| [docs/UI_REVIEW.md](docs/UI_REVIEW.md) | Design review against the built system |
| [BUILD_PLAN.md](BUILD_PLAN.md) | Original phase-by-phase engineering plan |
