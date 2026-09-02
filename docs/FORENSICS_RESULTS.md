# Forensics Results — Classical Detectors (Phase 2)

Generated 2026-09-02 by `python -m ml.evaluate_forensics`.

## Read this before quoting any number below

These figures come from **one synthetic specimen per tamper type plus one
genuine control** — five images in total. That is a smoke test, not a
benchmark. Detection rates on a sample of one are either 0% or 100% and
carry no confidence interval whatsoever.

The thresholds were also fitted against this same tiny set, so these
numbers describe the tuning data, not held-out performance. Real accuracy
figures require IDNet for training/validation and SIDTD as a held-out
cross-dataset check (see `docs/DATA_STRATEGY.md`). Nothing here should
appear in a submission as an accuracy claim.

## Per tamper type

| Tamper type | Detected | Score | Flagged by |
|---|---|---|---|
| `field_edit` | yes | 0.250 | error_level_analysis |
| `photo_splice` | yes | 0.600 | error_level_analysis |
| `recompression` | yes | 0.600 | error_level_analysis |
| `stamp_overlay` | yes | 0.600 | error_level_analysis |

## Negative control

| Sample | Flagged | Score |
|---|---|---|
| `genuine` | no (correct) | 0.000 |

## Summary

- Tamper types detected: **4/4** (100%)
- False positives on genuine documents: **0/1**
- Decision threshold: `0.2`

## Active checks and their weights

| Check | Weight | Status |
|---|---|---|
| `error_level_analysis` | 0.60 | active |
| `copy_move_detection` | 0.40 | active |
| `noise_consistency` | 0.00 | **advisory — not calibrated on real captures** |
| `cnn_classifier` | 0.00 | not yet trained (Phase 2 CNN) |

## Known limitations

**Noise consistency is disabled.** On synthetic renders it does not
discriminate: the genuine control peaked at 47.2 SD above the median block
energy while the four forgeries reached 19.6–46.1, meaning the clean
document looked *more* anomalous than every forgery. Normalizing for local
image activity did not help (51.1 vs 37.3–50.1). The cause is that
synthetic documents have no real sensor-noise field, so JPEG residual
energy tracks local content instead. The method is sound on genuine camera
captures and should be re-validated once IDNet/FantasyID data is available;
see `NOISE_DETECTOR_VALIDATED` in `app.modules.forensics.copy_move`.

**Copy-move is dormant on these samples.** It correctly stays silent on the
genuine control — an earlier version matched the MRZ's repeated `<` fillers
against themselves and reported 73 'cloned' keypoints sharing a (+104, 0)
offset. None of the four synthetic forgeries is a clone attack, so the
detector has no positive case here and is currently unverified in the
positive direction.

**No pixel-level localization is claimed.** Regions are coarse bounding
boxes. Tampered areas on ID documents occupy 0.27–4.17% of the image and
state-of-the-art detectors score near-zero on pixel-level localization
(DocForge-Bench, 2026).
