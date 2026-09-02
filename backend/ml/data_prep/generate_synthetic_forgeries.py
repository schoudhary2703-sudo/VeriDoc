"""Apply known tamper operations to clean specimen documents.

Produces one forged variant per tamper type, each with recorded ground truth, so
every forensic detector can be measured per type rather than on one blended
"tampered: yes/no" number.

The four operations mirror the attacks named in the problem statement:

    photo_splice    a face region replaced with content from a different source
    field_edit      a printed date/number digit painted over and rewritten
    stamp_overlay   a fake seal composited onto the document
    recompression   the document re-saved at lower JPEG quality after editing

These are deliberately *naive* forgeries -- the kind produced with consumer image
editing, which is what the classical detectors should catch. Sophisticated,
generation-based forgeries are what IDNet supplies for training the CNN; see
docs/DATA_STRATEGY.md. Both matter, and conflating them would overstate what the
classical layer can do.

Usage:
    python -m ml.data_prep.generate_synthetic_forgeries
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SOURCE = "specimen_passport_genuine.png"

# Photo box on the generated specimens (see generate_specimen_documents.py).
PHOTO_BOX = (40, 120, 280, 430)
# The printed date-of-birth row.
DOB_BOX = (330, 346, 560, 386)

RNG = np.random.default_rng(20260902)

# Capture simulation. A clean PNG render has no sensor noise and no compression
# history, so every noise/compression statistic on it is degenerate -- the
# detectors end up measuring text edges. Real input arrives through a camera or
# scanner, so the genuine baseline must too, or the whole forensic premise
# (regions that deviate from a consistent capture) has nothing to deviate from.
CAPTURE_NOISE_SIGMA = 3.0
CAPTURE_JPEG_QUALITY = 88


def simulate_capture(
    image: np.ndarray,
    *,
    noise_sigma: float = CAPTURE_NOISE_SIGMA,
    quality: int = CAPTURE_JPEG_QUALITY,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Model photographing the document: sensor noise, slight blur, JPEG save."""
    rng = rng or RNG
    captured = image.astype(np.float32)
    captured = cv2.GaussianBlur(captured, (3, 3), 0.6)
    captured += rng.normal(0.0, noise_sigma, captured.shape)
    captured = np.clip(captured, 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(".jpg", captured, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("capture simulation failed to encode")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


@dataclass
class Forgery:
    """Ground truth for one synthetic forgery."""

    filename: str
    tamper_type: str
    source: str
    description: str
    # Region the tamper was applied to, for localization scoring.
    region: tuple[int, int, int, int] | None


def photo_splice(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Replace the portrait with content of a visibly different origin.

    The pasted patch carries its own noise characteristics -- brighter, grainier,
    independently generated -- which is exactly the mismatch a noise-consistency
    check is designed to find.
    """
    out = image.copy()
    x1, y1, x2, y2 = PHOTO_BOX
    height, width = y2 - y1, x2 - x1

    # A synthetic "different camera": different base tone plus stronger grain.
    patch = np.full((height, width, 3), (150, 145, 138), dtype=np.float32)
    patch += RNG.normal(0.0, 14.0, patch.shape)

    # A crude head shape so it still reads as a portrait at a glance.
    centre = (width // 2, height // 3)
    cv2.circle(patch, centre, width // 6, (120, 116, 110), -1)
    cv2.ellipse(
        patch, (width // 2, height), (int(width * 0.38), height // 2), 0, 180, 360,
        (120, 116, 110), -1,
    )

    out[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)
    return out, PHOTO_BOX


def field_edit(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Paint over the printed date of birth and rewrite it.

    The classic document forgery: alter the visible date, hope nobody checks it
    against the MRZ. Phase 1's checksum catches the MRZ side; this is the
    printed side.
    """
    out = image.copy()
    x1, y1, x2, y2 = DOB_BOX

    # Sample the background so the patch blends at a glance.
    background = out[y1 - 12:y1 - 2, x1:x2].reshape(-1, 3).mean(axis=0)
    out[y1:y2, x1:x2] = background.astype(np.uint8)

    cv2.putText(
        out, "12 JUN 1988", (x1 + 2, y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (28, 32, 44), 2, cv2.LINE_AA,
    )
    return out, DOB_BOX


def stamp_overlay(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Composite a fake official seal onto the document."""
    out = image.copy()
    centre = (760, 300)
    radius = 92
    box = (centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius)

    overlay = out.copy()
    cv2.circle(overlay, centre, radius, (46, 70, 140), 6)
    cv2.circle(overlay, centre, radius - 16, (46, 70, 140), 2)
    cv2.putText(
        overlay, "VERIFIED", (centre[0] - 68, centre[1] + 8),
        cv2.FONT_HERSHEY_DUPLEX, 0.82, (46, 70, 140), 2, cv2.LINE_AA,
    )
    cv2.putText(
        overlay, "2026", (centre[0] - 32, centre[1] + 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 70, 140), 2, cv2.LINE_AA,
    )

    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    return out, box


def recompression(image: np.ndarray) -> tuple[np.ndarray, None]:
    """Re-save the document at low JPEG quality after a local edit.

    Models the forger who edits in one tool and exports from another. The whole
    image is affected, so there is no meaningful region -- which is itself worth
    encoding: not every tamper type is localizable.
    """
    edited, _ = field_edit(image)
    ok, encoded = cv2.imencode(".jpg", edited, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
    if not ok:
        raise RuntimeError("JPEG re-encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), None


OPERATIONS = {
    "photo_splice": (photo_splice, "Portrait replaced with content from a different source"),
    "field_edit": (field_edit, "Printed date of birth painted over and rewritten"),
    "stamp_overlay": (stamp_overlay, "Fake official seal composited onto the document"),
    "recompression": (recompression, "Document edited then re-saved at JPEG quality 45"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--samples", type=Path, default=root / "data" / "samples")
    parser.add_argument("--out", type=Path, default=root / "data" / "synthetic")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()

    source_path = args.samples / args.source
    image = cv2.imread(str(source_path))
    if image is None:
        raise SystemExit(
            f"Could not read {source_path}. Run generate_specimen_documents.py first."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    # Every sample passes through the same simulated capture, so the genuine
    # control and the forgeries share one noise floor and one compression
    # history. Tampering is then applied *after* capture -- modelling a forger
    # who edits a photographed document, which is what leaves the local
    # inconsistency the detectors look for.
    captured = simulate_capture(image)

    # The untouched control. Without a negative case, a detector that flags
    # everything scores perfectly on recall and is useless.
    control_name = "control_genuine.png"
    cv2.imwrite(str(args.out / control_name), captured)
    records = [
        Forgery(
            filename=control_name,
            tamper_type="genuine",
            source=args.source,
            description="Simulated capture of an untouched document",
            region=None,
        )
    ]

    for name, (operation, description) in OPERATIONS.items():
        forged, region = operation(captured)
        filename = f"forged_{name}.png"
        cv2.imwrite(str(args.out / filename), forged)
        records.append(
            Forgery(
                filename=filename,
                tamper_type=name,
                source=args.source,
                description=description,
                region=tuple(region) if region else None,
            )
        )
        print(f"wrote {args.out / filename}  [{name}]")

    manifest = args.out / "forgeries.json"
    manifest.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
