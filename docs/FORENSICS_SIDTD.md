# Forensics Results — SIDTD (cross-dataset validation)

Generated 2026-09-06 by `python -m ml.evaluate_sidtd`.

## What this file answers

Every other accuracy number in this project comes from FantasyID. This is
the only evidence about whether the detectors generalize.

SIDTD is built by a different group (CVC, Universitat Autònoma de Barcelona)
from a different source (MIDV-2020 document scans) using different forgery
methods (crop-and-replace inpainting on real photographs) than FantasyID's
generative face swaps and diffusion text edits. No threshold in this engine
was tuned on any of it, and since the engine fits nothing at run time, every
image here is held out by construction.

## Headline

- Images evaluated: **300** (150 genuine, 150 forged)  — random sample, seed 11
- **False-positive rate on genuine documents: 0/150 (0%)**
- **Forgery detection rate: 0/150 (0%)**
- Mean analysis time: 959 ms per image

> The timing is `engine.analyze` only and reflects whatever the machine had
> spare; it is not a benchmark. See the same note in `FORENSICS_FANTASYID.md`.

## Which checks fired on forged documents

| Check | Times flagged |
|---|---|
| *(none)* | 0 |

## False positives per nationality — read this before the averages

| Nationality | False positives | Rate |
|---|---|---|
| `alb` | 0/21 | 0% |
| `aze` | 0/14 | 0% |
| `esp` | 0/14 | 0% |
| `est` | 0/11 | 0% |
| `fin` | 0/19 | 0% |
| `grc` | 0/13 | 0% |
| `lva` | 0/18 | 0% |
| `rus` | 0/13 | 0% |
| `srb` | 0/15 | 0% |
| `svk` | 0/12 | 0% |

## Which checks fired on genuine documents

| Check | Times flagged |
|---|---|
| *(none)* | 0 |

## Detection rate per document nationality

| Nationality | Detected | Rate |
|---|---|---|
| `alb` | 0/17 | 0% |
| `aze` | 0/25 | 0% |
| `esp` | 0/15 | 0% |
| `est` | 0/13 | 0% |
| `fin` | 0/11 | 0% |
| `grc` | 0/12 | 0% |
| `lva` | 0/13 | 0% |
| `rus` | 0/14 | 0% |
| `srb` | 0/16 | 0% |
| `svk` | 0/14 | 0% |

## Compared with FantasyID

| | FantasyID | SIDTD |
|---|---|---|
| False positives on genuine | 0/150 (0%) | 0/150 (0%) |
| Forgery detection | 82/300 (27%) | 0/150 (0%) |

FantasyID's 27% is over face swaps and text manipulations in equal number;
SIDTD's forgeries are a different mix, so the two detection figures are not
like for like and should not be averaged. **The false-positive rate is the
comparable one**, and it is the one that matters: it is the property this
project protects, and a rate that holds on an unseen dataset is evidence the
thresholds are conservative rather than fitted.

## How to read a low detection rate here

A number well below FantasyID's would not, by itself, mean the engine failed.
SIDTD's forgeries are region-level crop-and-replace edits on already-scanned
documents, and rescanning destroys exactly the compression and sensor-noise
traces ELA and copy-move rely on. What it *would* mean is that the honest
claim is narrower than one dataset suggested — which is the reason to run it.

A false-positive rate well above zero is the serious outcome. That would say
the zero we quote is a property of FantasyID's captures rather than of the
thresholds, and the headline claim would have to be withdrawn.