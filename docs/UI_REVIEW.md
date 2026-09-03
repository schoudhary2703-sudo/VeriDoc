# Verify Screen — review against the built system

Review of `VeriDoc Verify Screen.html` against the implemented backend, 2026-09-03.

The design is good and several of its choices are now baked into the API. What
follows is the short list of places where the screen claims something the system
does not do, because those are the ones a judge can test live.

---

## 1. The mock MRZ is invalid — fix this first

The screen states *"All five check digits recompute correctly."* Running its MRZ
through our ICAO 9303 validator:

```
line 2 is 38 characters — TD3 requires 44

[FAIL] document_number   expected '9'  found '4'
[FAIL] date_of_birth     expected '7'  found '5'
[PASS] expiry_date       expected '9'  found '9'
[FAIL] personal_number   expected '2'  found '<'
[FAIL] composite         expected '9'  found '4'
```

Four of five fail, and the one that passes does so by coincidence. If anyone
checks — and MRZ checksums are exactly the kind of thing a judge with a phone can
check — the headline claim on the screen collapses.

**Corrected MRZ for the same fictional traveller** (ANANYA SHARMA, Z3541287,
DOB 14 Mar 1992, expiry 31 May 2031). Both lines are exactly 44 characters and
all five check digits validate:

```
P<INDSHARMA<<ANANYA<<<<<<<<<<<<<<<<<<<<<<<<<
Z3541287<9IND9203147F3105319<<<<<<<<<<<<<<08
```

Generated with `build_td3_mrz` in `backend/ml/data_prep/generate_specimen_documents.py`
— ask for any other name or number and it takes one call.

---

## 2. Claims that outrun the system

| The screen says | What the system does |
|---|---|
| "Interpol SLTD, the national lookout circular, or state BOLO lists" | A **seeded local table of four records**. The API returns `source: "simulated local record set (no live watchlist access)"` on every response, and there is a test asserting it. Name it as simulated in the UI too. |
| Face threshold **0.75**, score 0.62 shown as "weak" | Our threshold is **0.40** — InsightFace's published default, *not* calibrated on our data, because we have no document+live capture pairs. At 0.40, a score of 0.62 **passes**. |
| "Stamp and overprint consistency" check | **This check does not exist.** Ours are: MRZ checksum, compression consistency (ELA), duplicated-region detection, sensor-noise consistency (advisory), intra-document face consistency, face match, record cross-check. |
| "Model confidence 0.71 · below the 0.85 auto-clear threshold" | There is no model confidence, and **no auto-clear**. This also contradicts the screen's own — correct — line that the system never accepts or rejects on its own. Recommend deleting it. |
| "Analysis took 6.4s" | Measured **~1.9 s** warm, ~2.2 s with face analysis. The screen is pessimistic; use the real number. |

---

## 3. What the screen is missing

**Intra-document face consistency** is our strongest face-swap signal and has no
row on the screen. It takes face-swap detection from 29% to **46%** at no cost in
false positives, and the mechanism is the most memorable thing in the project:

> A generative face swap re-renders *both* the main portrait and the ghost image
> from the same model, so the two become **unnaturally alike**. On a genuine card
> they are physically different renderings of one photograph and agree well but
> imperfectly. Measured on held-out data: genuine 0.729 median similarity,
> face-swapped 0.844. **Too much consistency is the tamper signal.**

It deserves its own evidence row.

---

## 4. What the design got right, and is now in the API

These were adopted rather than argued with:

- **"The finding is region-level: it points at the portrait window, not at an
  exact pixel boundary."** Exactly right, and matches the published finding that
  tampered regions occupy 0.27–4.17% of an ID image and that state-of-the-art
  detectors score near-zero on pixel-level localization. Most teams promise a
  heatmap they cannot deliver.
- **"Confidence describes the model, not the traveller."** Kept verbatim in the
  risk badge.
- **"The system does not accept or reject on its own."** Kept verbatim in the
  footer.
- **Pass / weak / fail as three distinct states.** The API now returns a
  four-state `status` per evidence item rather than a boolean — the fourth being
  `not_applicable`, for checks that could not run on this document. A check that
  did not run is not evidence that a document is genuine, and hiding it would let
  an officer believe more was verified than actually was.
- **Glare and recapture guidance** ("a recapture under diffuse light may resolve
  this"). The risk scorer distinguishes a near-miss face score from a large gap
  for this reason, and says so in the evidence text.

---

## 5. Honest accuracy numbers for the pitch

From `docs/FORENSICS_FANTASYID.md`, measured on FantasyID's official held-out
test split — different card templates from training, real device captures, no
threshold tuned on any of it:

- **0% false positives** on 150 genuine documents
- **46%** face-swap detection, **6%** text manipulation, 26% overall
- 46% on Huawei and iPhone 15 captures, 8% on scanner, **0% on iPhone 15 Pro**

Quote the weak numbers alongside the strong ones. "6% on text manipulation and
0% on one capture device" is a far stronger position under questioning than a
single blended figure, because it shows the per-type reporting is real.
