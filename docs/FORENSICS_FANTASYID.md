# Forensics Results — FantasyID Held-Out Test Split

Generated 2026-09-03 by `python -m ml.evaluate_fantasyid`.

## Why this file exists separately

`FORENSICS_RESULTS.md` reports the synthetic smoke test, whose thresholds
were fitted on the very images it scores. This file does not have that
problem. FantasyID's official test split uses **different card templates**
from its train split, every image is a real print captured on an iPhone 15
Pro, a Huawei Mate 30, or a Kyocera scanner, and no threshold in the engine
was tuned on any of it.

These are therefore the first honest accuracy numbers in the project.

## Headline

- Images evaluated: **450** (150 bonafide, 300 attack)  — random sample of the split
- **False-positive rate on genuine documents: 0/150 (0%)**
- **Overall attack detection rate: 78/300 (26%)**
- Mean analysis time: 2181 ms per image

## Detection rate per attack type

| Attack type | Detected | Rate |
|---|---|---|
| `face` | 69/150 | **46%** |
| `text` | 9/150 | **6%** |

## Detection rate per capture device

| Device | Detected | Rate |
|---|---|---|
| `huawei` | 47/103 | 46% |
| `iphone15` | 23/50 | 46% |
| `iphone15pro` | 0/45 | 0% |
| `scan` | 8/102 | 8% |

## Interpretation — read this before drawing conclusions

**Face-swap detection comes from two orthogonal signals.** The classical checks
alone reach 29% on face swaps; adding intra-document face consistency lifts that
to 46% with no new false positives, because the two catch disjoint sets of
forgeries. The face check exploits a counter-intuitive property: a generative
swap re-renders both the main portrait and the ghost image from one model, so
they become *unnaturally alike*, whereas a genuine card's two portraits are
physically different renderings that agree well but imperfectly.

**Text manipulation remains close to undetected (6%).** Diffusion-based
inpainting (DiffSTE, TextDiffuser-2) blends into the host image's noise and
compression statistics, and the on-card photograph it does not touch is exactly
where our strongest signal lives. This is the honest weak point of the system.

**Detection is strongly device-dependent** and this is a deployment risk worth
naming: 46% on Huawei and iPhone 15 captures, 8% on scanner captures, 0% on
iPhone 15 Pro. A checkpoint standardised on the wrong capture hardware would get
far less from this pipeline than these headline numbers suggest. A single blended
accuracy figure would have hidden that completely.

**The zero false-positive rate is the property to protect.** Every threshold here
was set at its zero-false-positive operating point rather than its
accuracy-optimal one. The intra-document check, for instance, reaches 91.9%
face-swap recall at a threshold that also flags 38.5% of genuine documents --
which would be indefensible at a border, where a false accusation costs a real
traveller real time. Recall was traded away deliberately.

**No learned model contributes to these numbers.** Three CNN training attempts
all scored at or below chance on this split; see `FORENSICS_CNN_ATTEMPTS.md`.
Everything reported here comes from explainable checks, each of which states its
reasoning in a sentence an officer can read.

