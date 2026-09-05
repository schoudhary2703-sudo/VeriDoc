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

**The second traveller on the screen was invalid too.** The genuine-case card
(PRIYA VENKATESH, M7712854, DOB 28 Jul 1988, expiry 21 Nov 2034) had the same
problem — line 2 was 38 characters and its document-number, date-of-birth,
expiry and composite check digits all failed. Corrected, all five validate:

```
P<INDVENKATESH<<PRIYA<<<<<<<<<<<<<<<<<<<<<<<
M7712854<0IND8807283F3411218<<<<<<<<<<<<<<00
```

> **A trap worth naming, because the first fix fell into it.** Rebuilding the
> check digits around the *old* expiry produced
> `M7712854<0IND8807283F3402111<<<<<<<<<<<<<<04`, which passes all five check
> digits and is still wrong: it encodes 11 Feb 2034 while the printed page says
> 21 Nov 2034. The broken MRZ's expiry field had been wrong all along, and
> recomputing checksums around it just made the error self-consistent. On a
> screen whose own caption reads "every printed field agrees with the
> machine-readable zone", that is the exact forgery our cross-check is built to
> catch — the screen would have been arguing against itself. **Always read the
> field values off the printed page, never off the MRZ you are repairing.**

This one matters more than it looks: that card is the screen's *genuine*
example, and it carries the claim "all five check digits recompute correctly
and every printed field agrees with the machine-readable zone." A judge who
checks the genuine card and finds it fails is a worse outcome than one who
checks the flagged card.

**Status: both fixed.** All six MRZ blocks across `VeriDoc Verify Screen.html`
and `VeriDoc Officer Console.html` now carry 44-character lines with all five
check digits passing, and each agrees with the printed fields beside it. The
Officer Console was not part of the original review but contained two copies of
the same invalid Ananya MRZ.

**Check it yourself before any slide or demo:**

```bash
python backend/scripts/audit_mockup_claims.py "VeriDoc Verify Screen.html"
```

That script validates every MRZ, cross-checks it against the printed dates on
the same page, and greps for the stale claims in section 2. It exits non-zero on
any finding. Run it on anything a judge will see.

---

## 2. Claims that outrun the system

**All five are now fixed** in both `VeriDoc Verify Screen.html` and
`VeriDoc Officer Console.html`. What was wrong and what it became:

| The screen said | What the system does | Fix applied |
|---|---|---|
| "Interpol SLTD, the national lookout circular, or state BOLO lists" | A **seeded local table of four records**. The API returns `source: "simulated local record set (no live watchlist access)"` on every response, and there is a test asserting it. | Reworded to "the simulated local record set", stating plainly that this build has no live watchlist access and those feeds are not queried. The Officer Console's "Exact hit on Interpol SLTD" now reads as a seeded demo record. |
| Face threshold **0.75**, score 0.62 shown as "weak" | Our threshold is **0.40** — InsightFace's published default, *not* calibrated on our data, because we have no document+live capture pairs. At 0.40, a score of 0.62 **passes**. | Threshold corrected to 0.40 everywhere. The review-case score moved 0.62 → **0.31** so the "weak, a recapture may resolve this" narrative stays true against the real threshold. The genuine case keeps 0.94, which passes either way. |
| "Stamp and overprint consistency" check | **This check does not exist.** Ours are: MRZ checksum, compression consistency (ELA), duplicated-region detection, sensor-noise consistency (advisory), intra-document face consistency, face match, record cross-check. | Repurposed into the **intra-document face consistency** row, which is a real check and was missing from the screen entirely (see section 3). Two birds. |
| "Model confidence 0.71 · below the 0.85 auto-clear threshold" | There is no model confidence, and **no auto-clear**. This also contradicts the screen's own — correct — line that the system never accepts or rejects on its own. | Confidence bar and threshold removed from both screens, replaced with a line stating the recommendation is advisory and there is no auto-clear. The summary badge is now "Review", not "Review · confidence 0.71". |
| "Analysis took 6.4s" | Measured **~1.9 s** warm, ~2.2 s with face analysis. | Every timing on both screens was rewritten into the real 1.9–2.7 s band. It was not only the 6.4 s the review spotted: the Verify Screen also carried a 4.2 s, and the Officer Console queue ran 3.8 s to 10.4 s per case with a **"p95 11.3s"** latency figure on the header. A p95 five times the true warm run is the kind of number a judge asks you to reproduce live. |

Note on the confidence bar: it could not simply be relabelled "risk score",
because the mock values invert. A confidence of 0.96 on the genuine case would
have become a risk of 0.96 — the opposite reading. Removing it was the only
safe fix.

---

## 3. What the screen is missing

**Intra-document face consistency** is our strongest face-swap signal and has no
row on the screen. It takes face-swap detection from 29% to **47%** at no cost in
false positives, and the mechanism is the most memorable thing in the project:

> A generative face swap re-renders *both* the main portrait and the ghost image
> from the same model, so the two become **unnaturally alike**. On a genuine card
> they are physically different renderings of one photograph and agree well but
> imperfectly. Measured on held-out data: genuine 0.729 median similarity,
> face-swapped 0.844. **Too much consistency is the tamper signal.**

It deserves its own evidence row.

**Done** — it took over the row vacated by the non-existent "stamp and overprint
consistency" check. The review case now reads: portrait and ghost image agree at
0.731, normal for two physically different renderings of one photograph, and a
face swap pushes this above the 0.884 threshold in
`backend/app/modules/face/face_match.py:75`. The genuine case shows 0.742.

**The inconsistency raised here is now resolved** (2026-09-05). Both figures were
real, from different runs: **28.7% → 40.0%** was the Phase 3 pilot on 80 genuine +
80 face swaps, and **29% → 46%** was the full 450-image evaluation. Neither was
wrong; the code comment simply never said which sample it came from.

Checking it surfaced a third problem the review could not have seen: *every* doc
still carried pre-`bd0af40` numbers, because the exposure fix changed detection
and the evaluation was never re-run. It has been now, on the identical sample
(`--limit 150 --seed 11`), and the current figures are **29% → 47%** face swap,
**8%** text, **27%** overall, still at **0/150** false positives. `engine.py` now
names its sample size and points at the full evaluation.

The generator itself was the deeper fault: `evaluate_fantasyid.py` hardcoded its
prose percentages, so the regenerated file claimed 46% on Huawei directly beside
a table it had just computed saying 49%. Those figures are now derived from the
run. This is the same defect class the audit script exists to catch, one level
further up — a report contradicting its own data.

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
- **47%** face-swap detection, **8%** text manipulation, 27% overall
- 49% on Huawei, 48% on iPhone 15, 8% on scanner, **0% on iPhone 15 Pro**

Quote the weak numbers alongside the strong ones. "8% on text manipulation and
0% on one capture device" is a far stronger position under questioning than a
single blended figure, because it shows the per-type reporting is real.
