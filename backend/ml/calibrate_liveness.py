"""Fit liveness thresholds from your own captures. No licensed dataset needed.

CelebA-Spoof, OULU-NPU and Replay-Attack are all research-licence only, and
without one of them the liveness module cannot be calibrated -- so it ships
disabled. This script closes that gap with photographs a team can take in about
twenty minutes.

What to collect (30+ of each is enough to fit a usable threshold):

    data/liveness/live/    a person's face, photographed directly by the camera
    data/liveness/spoof/   the SAME faces re-presented as an attack:
                             - a printed photo held up to the camera
                             - a photo displayed on a phone or laptop screen

Collect both classes on the same camera, in the same lighting. Otherwise the
thresholds will separate *cameras* rather than attacks, which is the same trap
that made the tamper CNN memorise card templates instead of learning forgery.

Only photograph people who have agreed to it, and keep the images out of the
repository -- `data/` is gitignored for exactly this reason.

Each threshold is fitted at its zero-false-positive point on the live set:
refusing a genuine traveller is a worse failure than missing an attack, and this
is a first-stage check with a human behind it.

Usage:
    python -m ml.calibrate_liveness --data ../data/liveness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.modules.face.liveness import CUE_FUNCTIONS, _prepare

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _face_crops(folder: Path) -> list[np.ndarray]:
    """Largest detected face from every image in `folder`."""
    from app.modules.face import face_match

    crops: list[np.ndarray] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue

        faces = face_match.detect_faces(image)
        if not faces:
            print(f"  no face found in {path.name}, skipped")
            continue

        x1, y1, x2, y2 = (int(v) for v in faces[0].bbox)
        h, w = image.shape[:2]
        crop = image[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)]
        if crop.size and min(crop.shape[:2]) >= 32:
            crops.append(_prepare(crop))
    return crops


def _measure_all(crops: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.array([function(c) for c in crops], dtype=float)
        for name, (function, _, _) in CUE_FUNCTIONS.items()
    }


def _fit(live: np.ndarray, spoof: np.ndarray, direction: str) -> dict:
    """Threshold at the zero-false-positive point on the live set."""
    if live.size == 0 or spoof.size == 0:
        return {"threshold": None, "recall": 0.0, "false_positive_rate": 0.0}

    if direction == "above":
        threshold = float(live.max())
        recall = float((spoof > threshold).mean())
        fpr = float((live > threshold).mean())
    else:
        threshold = float(live.min())
        recall = float((spoof < threshold).mean())
        fpr = float((live < threshold).mean())

    # Separability, independent of any threshold: the probability that a random
    # spoof scores more extreme than a random live capture.
    if direction == "above":
        auc = float((spoof[:, None] > live[None, :]).mean())
    else:
        auc = float((spoof[:, None] < live[None, :]).mean())

    return {
        "threshold": round(threshold, 4),
        "recall_at_zero_fpr": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "auc": round(auc, 4),
        "live_median": round(float(np.median(live)), 4),
        "spoof_median": round(float(np.median(spoof)), 4),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "liveness")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    live_dir, spoof_dir = args.data / "live", args.data / "spoof"
    for folder in (live_dir, spoof_dir):
        if not folder.is_dir():
            raise SystemExit(
                f"Missing {folder}.\n"
                "Expected data/liveness/live and data/liveness/spoof — see the "
                "module docstring for what to collect."
            )

    print("measuring live captures...")
    live = _measure_all(_face_crops(live_dir))
    print("measuring spoof captures...")
    spoof = _measure_all(_face_crops(spoof_dir))

    n_live = len(next(iter(live.values()), []))
    n_spoof = len(next(iter(spoof.values()), []))
    print(f"\nusable faces: {n_live} live, {n_spoof} spoof")
    if n_live < 10 or n_spoof < 10:
        print(
            "WARNING: fewer than 10 of a class. A threshold fitted on this little "
            "data describes these photographs, not presentation attacks."
        )

    results = {}
    print(f"\n{'cue':24} {'live med':>9} {'spoof med':>10} {'AUC':>6} {'recall@0FPR':>12}")
    print("-" * 66)
    for name, (_, direction, _) in CUE_FUNCTIONS.items():
        fitted = _fit(live[name], spoof[name], direction)
        results[name] = fitted
        print(
            f"{name:24} {fitted['live_median']:9.3f} {fitted['spoof_median']:10.3f} "
            f"{fitted['auc']:6.3f} {fitted['recall_at_zero_fpr']:11.0%}"
        )

    usable = {k: v for k, v in results.items() if v.get("auc", 0) >= 0.70}
    print(
        f"\n{len(usable)} of {len(results)} cues separate the classes (AUC >= 0.70)."
    )
    if not usable:
        print(
            "None are usable. Leave LIVENESS_CALIBRATED = False rather than "
            "enabling a check that cannot tell the classes apart."
        )
    else:
        print("\nPaste into app/modules/face/liveness.py:\n")
        print("PLACEHOLDER_THRESHOLDS = {")
        for name, fitted in results.items():
            note = "" if name in usable else "  # AUC too low to rely on"
            print(f'    "{name}": {fitted["threshold"]},{note}')
        print("}")
        print("\nThen set LIVENESS_CALIBRATED = True and record in docs/DATASETS.md")
        print("how many captures, on what camera, and under what lighting.")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
