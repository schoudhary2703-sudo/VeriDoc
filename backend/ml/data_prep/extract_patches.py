"""Extract native-resolution patches from FantasyID for tamper classification.

Why patches, stated plainly: the first attempt resized whole 2784x1757 cards to
384px and scored macro-F1 0.306 on the held-out split -- *below* the 0.439 a
trivial "always say manipulated" classifier gets. Measured cause: manipulated
regions occupy a median 1.15% of image area, so at that resize a typical altered
field shrank from ~56,000 px to ~1,067 px, around 33x33. Nothing forensic
survives that, so the network memorized template identity instead (training loss
0.0055, validation F1 flat from epoch 3).

Cropping at native resolution keeps the artifacts intact.

The important design choice is where negatives come from. Every attack image
contributes **both** positive patches (from its `altered` regions) and negative
patches (from its own `original` regions). Positives and negatives therefore
share a template, a printer, a capture device and a lighting condition -- so the
model cannot separate the classes by recognising the card design, which is
exactly the shortcut that broke the first attempt. It has to look at the pixels
that changed.

Usage:
    python -m ml.data_prep.extract_patches
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

PATCH_SIZE = 256

# Per attack image: 2 patches on its altered regions.
POS_PER_ATTACK = 2

# Negatives taken from the untouched regions of an ATTACK image. Zero by default,
# and the reason matters.
#
# Taking negatives from the same image is appealing -- positives and negatives
# would share a template, printer, camera and lighting, so the model could not
# separate the classes by recognising the card design. But manipulating an image
# and re-saving it re-encodes the WHOLE image, so an "original" region of an
# attack image carries the same global re-compression signature as the altered
# one. Labelling those patches genuine teaches the model that the manipulation
# signature means genuine, cancelling the signal.
#
# Measured: with NEG_PER_ATTACK = 1, validation macro-F1 sat at chance (0.478 to
# 0.519 across 7 epochs) while training loss fell from 1.33 to 0.062.
NEG_PER_ATTACK = 0

# Per bonafide image: enough untouched patches to balance the positives.
NEG_PER_BONAFIDE = 4

# A patch must overlap its target region by at least this fraction of the
# region, otherwise a crop centred near a region edge can contain none of it.
MIN_REGION_COVERAGE = 0.25


def _regions_by_provenance(
    json_path: Path, *, is_attack: bool
) -> tuple[list[dict], list[dict]]:
    """Return (altered, original) region shape dicts for one image.

    Bonafide images carry region boxes but no `region_provenance` key at all --
    only attack images annotate which regions were touched. On a genuine card
    every region is by definition original, so unlabelled regions count as
    original there and are skipped on attack images, where "unlabelled" really
    does mean "provenance unknown".
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []

    altered: list[dict] = []
    original: list[dict] = []
    for region in data.get("regions", []):
        shape = region.get("shape_attributes") or {}
        if not {"x", "y", "width", "height"} <= shape.keys():
            continue
        provenance = (region.get("region_attributes") or {}).get("region_provenance")
        if provenance == "altered":
            altered.append(shape)
        elif provenance == "original" or (provenance is None and not is_attack):
            original.append(shape)
        # On an attack image an unlabelled region is not evidence of either class.
    return altered, original


def _crop_window(
    shape: dict, image_size: tuple[int, int], patch: int, rng: random.Random
) -> tuple[int, int, int, int] | None:
    """A patch-sized window at native resolution overlapping `shape`."""
    width, height = image_size
    if width < patch or height < patch:
        return None

    rx, ry = int(shape["x"]), int(shape["y"])
    rw, rh = int(shape["width"]), int(shape["height"])
    if rw <= 0 or rh <= 0:
        return None

    # Anchor on a random point inside the region, then centre the patch there.
    cx = rng.randint(rx, rx + rw - 1)
    cy = rng.randint(ry, ry + rh - 1)

    x1 = max(0, min(cx - patch // 2, width - patch))
    y1 = max(0, min(cy - patch // 2, height - patch))
    x2, y2 = x1 + patch, y1 + patch

    # Reject windows that barely touch the region.
    overlap_w = max(0, min(x2, rx + rw) - max(x1, rx))
    overlap_h = max(0, min(y2, ry + rh) - max(y1, ry))
    covered = (overlap_w * overlap_h) / float(rw * rh)
    visible = (overlap_w * overlap_h) / float(patch * patch)
    if covered < MIN_REGION_COVERAGE and visible < 0.02:
        return None

    return x1, y1, x2, y2


def extract_split(
    root: Path, split: str, out_dir: Path, patch: int, seed: int
) -> Counter:
    rng = random.Random(seed)
    counts: Counter = Counter()

    for label in ("genuine", "manipulated"):
        (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    with open(root / f"{split}.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        image_path = root / row["path"]
        if not image_path.exists():
            continue

        is_attack = row["is_attack"] == "True"
        altered, original = _regions_by_provenance(
            image_path.with_suffix(".json"), is_attack=is_attack
        )

        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            continue

        wanted: list[tuple[list[dict], str, int]] = []
        if is_attack:
            wanted.append((altered, "manipulated", POS_PER_ATTACK))
            wanted.append((original, "genuine", NEG_PER_ATTACK))
        else:
            wanted.append((original, "genuine", NEG_PER_BONAFIDE))

        stem = image_path.relative_to(root).as_posix().replace("/", "_")[:-4]

        for regions, label, quota in wanted:
            if not regions:
                continue
            for index in range(quota):
                shape = rng.choice(regions)
                window = _crop_window(shape, image.size, patch, rng)
                if window is None:
                    continue
                out_path = out_dir / split / label / f"{stem}_{label[:3]}{index}.jpg"
                image.crop(window).save(out_path, quality=95)
                counts[f"{split}/{label}"] += 1

    return counts


def main() -> None:
    root_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root_dir / "data" / "raw" / "FantasyID")
    parser.add_argument("--out", type=Path, default=root_dir / "data" / "processed" / "patches")
    parser.add_argument("--patch", type=int, default=PATCH_SIZE)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"FantasyID not found at {args.data}")

    totals: Counter = Counter()
    for split in ("train", "test"):
        print(f"extracting {split}...")
        totals.update(extract_split(args.data, split, args.out, args.patch, args.seed))

    print()
    for key in sorted(totals):
        print(f"  {key:24} {totals[key]:6d} patches")
    print(f"\nwrote {sum(totals.values())} patches to {args.out}")


if __name__ == "__main__":
    main()
