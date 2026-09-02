# Forensics Results — FantasyID Held-Out Test Split

Generated 2026-09-02 by `python -m ml.evaluate_fantasyid`.

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
- **Overall attack detection rate: 53/300 (18%)**
- Mean analysis time: 681 ms per image

## Detection rate per attack type

| Attack type | Detected | Rate |
|---|---|---|
| `face` | 44/150 | **29%** |
| `text` | 9/150 | **6%** |

## Detection rate per capture device

| Device | Detected | Rate |
|---|---|---|
| `huawei` | 36/103 | 35% |
| `iphone15` | 17/50 | 34% |
| `iphone15pro` | 0/45 | 0% |
| `scan` | 0/102 | 0% |

## Interpretation — read this before drawing conclusions

**The classical detectors are close to blind against modern generative
manipulation.** FantasyID's attacks are face swaps (Inswapper, FaceDancer)
and diffusion-based text inpainting (DiffSTE, TextDiffuser-2). These blend
into the host image's noise and compression statistics far too well for
error-level analysis or keypoint matching to catch reliably.

**The zero false-positive rate is the part worth keeping.** A detector that
never accuses a genuine traveller, and catches some fraction of attacks, is
a usable first stage in a layered pipeline. One that catches more attacks by
flagging genuine documents would be worse than useless at a border.

**This is the evidence for the CNN, not an argument against the classical
layer.** The classical checks stay because they are explainable and cost
almost nothing; the learned model exists to cover exactly the gap measured
here. Expect the CNN to carry detection and the classical checks to carry
the explanation.
