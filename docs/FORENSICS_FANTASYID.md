# Forensics Results — FantasyID Held-Out Test Split

Generated 2026-09-05 by `python -m ml.evaluate_fantasyid`.

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
- **Overall attack detection rate: 82/300 (27%)**
- Mean analysis time: 5362 ms per image

> The timing above is `engine.analyze` only -- forensics, including the ~1.5 s
> face-consistency check. It is **not** the end-to-end verification latency, and
> it is not comparable to the ~2 s warm figure in the README: that measures the
> full pipeline on a 900x600 specimen, while these are full-resolution phone
> photographs. It is also whatever the machine had spare -- two runs of this
> identical sample measured 3644 ms and 5362 ms while a video player held the
> top CPU slot. Treat it as an upper bound, and re-run on an idle machine before
> quoting a latency anywhere. Detection rates are deterministic and unaffected.

## Detection rate per attack type

| Attack type | Detected | Rate |
|---|---|---|
| `face` | 70/150 | **47%** |
| `text` | 12/150 | **8%** |

## Detection rate per capture device

| Device | Detected | Rate |
|---|---|---|
| `huawei` | 50/103 | 49% |
| `iphone15` | 24/50 | 48% |
| `iphone15pro` | 0/45 | 0% |
| `scan` | 8/102 | 8% |

## Interpretation — read this before drawing conclusions

**Face-swap detection comes from two orthogonal signals.** The classical checks
alone reach 29% on face swaps; adding intra-document face consistency lifts that
to 47% with no new false positives, because the two catch disjoint sets of
forgeries. The face check exploits a counter-intuitive property: a generative
swap re-renders both the main portrait and the ghost image from one model, so
they become *unnaturally alike*, whereas a genuine card's two portraits are
physically different renderings that agree well but imperfectly.

**Text manipulation remains close to undetected (8%).** Diffusion-based
inpainting blends into the host image's noise and compression statistics, and the
on-card photograph it does not touch is exactly where our strongest signal lives.
This is the honest weak point of the system.

**Detection is strongly device-dependent** and this is a deployment risk worth
naming: 49% on huawei, 48% on iphone15, 0% on iphone15pro, 8% on scan. A checkpoint standardised on the wrong capture hardware
would get far less from this pipeline than the headline suggests, and a single
blended accuracy figure would hide that completely.

**The zero false-positive rate is the property to protect.** Every threshold was
set at its zero-false-positive operating point rather than its accuracy-optimal
one. The intra-document check reaches 91.9% face-swap recall at a threshold that
also flags 38.5% of genuine documents -- indefensible at a border. Recall was
traded away deliberately.

**No learned model contributes to these numbers.** Three CNN training attempts
all scored at or below chance on this split; see `FORENSICS_CNN_ATTEMPTS.md`.