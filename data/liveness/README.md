# Liveness calibration captures

This folder holds the photos that calibrate the passive liveness cues in
`backend/app/modules/face/liveness.py`. **The images are never committed** — the
`.gitignore` keeps this README and ignores everything else in here. Only
photograph people who have agreed to it.

Until these are collected and the calibration script is run, the module ships
with `LIVENESS_CALIBRATED = False` and every result returns `passed=None`. That
is deliberate: a threshold fitted on no data is a guess, and this project does
not ship guesses (see the five ground rules).

## What to collect

Aim for **~30 people**. For each person, three photos on the **same camera in
the same lighting**:

| Folder | What | Class |
|---|---|---|
| `live/`  | Their real face, photographed directly by the camera | live |
| `spoof/` | A **printed** photo of their face, held up to the camera | spoof |
| `spoof/` | Their photo shown **on a phone or laptop screen**, photographed | spoof |

So one person contributes **1 live + 2 spoof** images. Thirty people ≈ 30 live
and 60 spoof.

```
data/liveness/
├── live/     <- real faces
└── spoof/    <- printed photos AND screen photos (both attack types together)
```

## The one rule that ruins everything if broken

**Shoot live and spoof on the SAME camera, in the SAME lighting.** If you
photograph real faces on your phone and spoof photos on someone else's, the
thresholds will learn to tell the two *cameras* apart, not real faces from fake
ones. It will look brilliant in testing and fail completely in the demo. This is
the exact mistake that made an earlier model in this project memorise card
templates instead of detecting forgery.

Practical way to guarantee it: one phone, one room, one sitting. Take the live
shot, then immediately print that same photo (or open it on a second screen) and
re-photograph it with the same phone from the same spot.

## Then calibrate

```bash
cd backend
.venv\Scripts\python -m ml.calibrate_liveness --data ../data/liveness
```

It measures every cue on both classes and reports, per cue:

- **AUC** — separability, independent of any threshold. Keep a cue only if
  **AUC ≥ 0.70**.
- **recall @ 0 FPR** — how many spoofs it catches at the threshold that flags
  **zero** live faces. The threshold is fitted at the zero-false-positive point
  on purpose: refusing a genuine traveller is worse than missing an attack.

The script prints a ready-to-paste `PLACEHOLDER_THRESHOLDS` block. If — and only
if — at least one cue reaches AUC ≥ 0.70:

1. Paste the thresholds into `backend/app/modules/face/liveness.py`.
2. Set `LIVENESS_CALIBRATED = True`.
3. Record in `docs/DATASETS.md`: how many photos, what camera, what lighting,
   and the per-cue AUC — including the cues that failed.

If nothing reaches 0.70, **leave `LIVENESS_CALIBRATED = False`** and write down
that it did not separate. A documented negative result is an acceptable, honest
outcome — not a defeat.

## Note on the moire cue

The `moire_interference` cue was reworked (see its docstring and
`docs/DATASETS.md` §5) so it is no longer confounded by blur — a soft print used
to out-score an actual screen replay, and now it does not. That is a *mechanism*
fix verified on synthetic transforms; whether it actually separates real screens
from real faces still has to be established here, on your captures, like every
other cue. Do not treat the rework as permission to enable it without an AUC.
