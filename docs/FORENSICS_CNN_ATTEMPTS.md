# Forensics CNN — Three Attempts, Three Failures

Generated 2026-09-03.

**Summary: the learned tamper classifier does not work, and is disabled.** Three
training configurations were tried against FantasyID's held-out test split. None
beat chance. This file records what was tried, what it scored, and what was
measured about why — so the next person does not repeat it, and so nothing here
is quoted as a working result.

The forensics engine ships **classical-only**. `CHECK_WEIGHTS["cnn_classifier"]`
is `0.0`, and no checkpoint sits at the production path, so `engine.analyze`
degrades to the explainable checks rather than loading a model that cannot
justify its verdict.

---

## Results

| # | Configuration | Best macro-F1 | Trivial baseline | Verdict |
|---|---|---|---|---|
| 1 | Whole image resized to 384px | **0.306** | 0.439 | **worse than trivial** |
| 2 | 256px native patches, negatives from attack + bonafide images | **0.519** | 0.360 | chance |
| 3 | 256px native patches, negatives from bonafide images only | **0.461** | 0.392 | chance |

"Trivial baseline" is a classifier that always answers *manipulated*. Random
balanced guessing scores about 0.49, so attempts 2 and 3 learned nothing
transferable.

Every attempt shows the same signature: **training loss collapses while
validation stays flat.**

| Attempt | Train loss, first → last | Validation macro-F1 across epochs |
|---|---|---|
| 1 | 1.033 → 0.0055 | 0.303, 0.305, **0.306**, 0.285, … flat/declining |
| 2 | 1.330 → 0.062 | 0.478, **0.519**, 0.497, 0.477, 0.480, 0.494, 0.487 |
| 3 | 0.597 → 0.024 | 0.435, 0.457, **0.461**, 0.445 |

In attempts 1 and 3 the manipulated-class recall was **0.196** and **0.247** —
the model calls most manipulated documents genuine, which is the worst possible
failure direction for this application.

---

## Attempt 1 — whole image at 384px

**What was measured, not guessed.** Sampling 120 test images and their region
annotations:

- Median image size: **2784 × 1757**
- Manipulated regions: **median 1.15% of image area** (p10 0.78%, p90 3.48%),
  closely matching the 0.27–4.17% that DocForge-Bench reports
- At a 384px whole-image resize (scale 0.138), a median manipulated region
  shrinks from **~56,000 px to ~1,067 px** — roughly 33 × 33

A diffusion-inpainted text field rendered into 33 × 33 pixels retains no
forensic artifact. There was nothing to learn, so the network memorized template
identity instead.

Worth recording as a process failure, not only a technical one: the training
script's own docstring said *"Input size 384, not the usual 224 — downscaling to
224 destroys the very artifacts the model is meant to see"* — and 384 was then
applied to 2784px images, a 7.2× downscale. 384 is large relative to ImageNet,
not relative to a document scan.

## Attempt 2 — native-resolution patches, same-image negatives

256 × 256 crops at native resolution. Positives from `altered` regions of attack
images; negatives from the `original` regions **of those same images**, so that
positives and negatives shared a template, printer, camera and lighting and the
model could not separate classes by recognising the card design.

Result: chance. A hypothesis for why — manipulating an image and re-saving it
re-encodes the *whole* image, so an "original" region of an attack image carries
the same global re-compression signature as the altered one, and labelling it
genuine teaches the model that the manipulation signature means genuine.

## Attempt 3 — native-resolution patches, clean negatives

Same as attempt 2, but negatives taken **only** from bonafide images, removing
the suspected label noise.

Result: **worse** (0.461 vs 0.519). The hypothesis is refuted — negative
sampling was not the limiting factor.

---

## What this means

Three distinct configurations, each failing the same way, points at the data and
compute budget rather than at any single bug:

- **1,899 training images** is very little for a model that must generalize to
  unseen card templates.
- **The split is deliberately hard.** FantasyID's train and test halves use
  different templates, and train contains only *combined* face+text
  manipulations while test contains each manipulation *in isolation*.
- **The manipulations are current-generation.** Inswapper and FaceDancer face
  swaps, DiffSTE and TextDiffuser-2 inpainting. These are built to be
  imperceptible and they blend into host noise and compression statistics.
- **CPU-only training** rules out the larger backbones and longer schedules that
  published results on this benchmark rely on.

## What would be tried next, given more resources

1. **More data.** IDNet is 124.93 GB across 23 per-country archives; even two or
   three of them would multiply the training set. This machine had 32 GB free at
   the time and a link that stalled on 100 MB downloads.
2. **A GPU.** 0.486 s/image on CPU at 384px made each configuration a multi-hour
   commitment, which is why only three were tried.
3. **A backbone pretrained on forensics**, rather than ImageNet features that
   describe objects and say nothing about compression history.
4. **Frequency-domain input** — DCT coefficient histograms rather than RGB. Most
   published document-forgery detectors operate there, and it is a far smaller
   hypothesis space than fine-tuning a general vision backbone.

## What ships instead

The classical engine, whose honest numbers are in `FORENSICS_FANTASYID.md`:
**0% false positives on 150 genuine documents, 29% detection on face swaps, 6%
on text manipulation, 18% overall.**

That is a weak detector and a strong *first stage*: it never accuses a genuine
traveller, every finding it makes is explainable in a sentence, and it costs
~680 ms per document. In a layered pipeline where a human officer makes the
final call, a low-recall, zero-false-positive, fully-explainable stage is
defensible. A black-box model scoring at chance is not.

The three failed checkpoints are kept in `backend/ml/checkpoints/` as
`failed_*.pt` rather than deleted. A measured negative result is evidence; a
missing one is a gap in the record.
