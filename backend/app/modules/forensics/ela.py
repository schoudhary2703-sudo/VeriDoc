"""Error Level Analysis.

The idea: JPEG compression is lossy but *idempotent at a given quality* -- an
untouched region that has already been compressed at quality Q barely changes
when recompressed at Q, while a region that was pasted in, painted over, or
re-rendered has different compression history and changes more. Recompress the
image and look at where the error is large.

What ELA is good for: pointing at a region whose compression history differs
from its surroundings. What it is not: proof of forgery. Resampling, sharpening
and even a screenshot will light ELA up. It is one signal among several, which
is exactly why `ForensicsResult` carries findings rather than a verdict.

Reference: Krawetz, "A Picture's Worth: Digital Image Analysis and Forensics".
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.core.schemas import ForensicsFinding, Region, TamperType
from app.modules.forensics.common import (
    block_reduce,
    blocks_to_boxes,
    merge_boxes,
    robust_z,
    structure_density,
    to_gray,
)

# Recompression quality. 90 is the conventional choice: high enough that genuine
# regions barely move, low enough that mismatched history shows up.
DEFAULT_QUALITY = 90

# A block is suspicious when its mean error exceeds the image's mean error by
# this many standard deviations. Tuned to be deliberately conservative -- a false
# "tampered" on a genuine passport costs an innocent traveller time.
# Measured on the synthetic forgery set: the genuine control peaks at 8.6 SD
# above the median block error, while the four forgeries peak at 15.3, 27.0,
# 48.1 and 83.9. 12.0 sits in that gap with margin on both sides.
#
# Caveat worth stating plainly: this is tuned against ONE genuine control. It is
# a starting point, not a validated threshold, and must be re-fitted on IDNet
# before any accuracy number derived from it is quoted.
DEFAULT_Z_THRESHOLD = 12.0

BLOCK_SIZE = 32

# Text strokes carry high compression error on any image, genuine or not, so
# text-dense blocks are excluded from the statistics for the same reason the
# noise test excludes them: they describe the typography, not the edit history.
MAX_STRUCTURE_DENSITY = 0.06

# A deviation covering most of the page is capture variation, not a local edit.
MAX_OUTLIER_FRACTION = 0.45


def compute_ela_map(image: np.ndarray, quality: int = DEFAULT_QUALITY) -> np.ndarray:
    """Return a single-channel float32 error map, one value per pixel.

    `image` is BGR as read by OpenCV.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = np.array(Image.open(buffer).convert("RGB"))

    # Per-pixel maximum absolute difference across channels.
    diff = np.abs(rgb.astype(np.int16) - recompressed.astype(np.int16))
    return diff.max(axis=2).astype(np.float32)


def _block_statistics(
    error_map: np.ndarray, block: int = BLOCK_SIZE
) -> tuple[np.ndarray, int, int]:
    """Mean error per block. Returns (block_means, rows, cols)."""
    height, width = error_map.shape
    rows, cols = height // block, width // block
    if rows == 0 or cols == 0:
        return np.zeros((0, 0), dtype=np.float32), 0, 0

    trimmed = error_map[: rows * block, : cols * block]
    reshaped = trimmed.reshape(rows, block, cols, block)
    return reshaped.mean(axis=(1, 3)), rows, cols


def _merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge overlapping or touching boxes so one edit yields one region.

    Without this a single spliced photo produces a scatter of adjacent blocks,
    and the evidence panel shows twenty findings where there is one.
    """
    if not boxes:
        return []

    merged = True
    current = list(boxes)
    while merged:
        merged = False
        output: list[tuple[int, int, int, int]] = []
        while current:
            x1, y1, x2, y2 = current.pop()
            keep: list[tuple[int, int, int, int]] = []
            for other in current:
                ox1, oy1, ox2, oy2 = other
                overlaps = not (ox1 > x2 or ox2 < x1 or oy1 > y2 or oy2 < y1)
                if overlaps:
                    x1, y1 = min(x1, ox1), min(y1, oy1)
                    x2, y2 = max(x2, ox2), max(y2, oy2)
                    merged = True
                else:
                    keep.append(other)
            current = keep
            output.append((x1, y1, x2, y2))
        current = output
    return current


def analyze(
    image: np.ndarray,
    *,
    quality: int = DEFAULT_QUALITY,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    min_blocks: int = 2,
) -> ForensicsFinding:
    """Run ELA and report regions whose compression history looks inconsistent.

    `min_blocks` requires more than one suspicious block before flagging, which
    suppresses the isolated hot blocks that sharp text edges naturally produce.
    """
    error_map = compute_ela_map(image, quality=quality)
    block_means = block_reduce(error_map, BLOCK_SIZE, how="mean")

    if block_means.size == 0:
        return ForensicsFinding(
            check="error_level_analysis",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail="Image too small for block-level error analysis",
        )

    # Structure is measured on a denoised copy so grain is not read as text.
    density = structure_density(to_gray(image), BLOCK_SIZE)
    smooth = density <= MAX_STRUCTURE_DENSITY

    if smooth.sum() < 8:
        return ForensicsFinding(
            check="error_level_analysis",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail="Document is too text-dense for reliable error-level analysis",
        )

    z_scores = robust_z(block_means, smooth)
    z_scores[~smooth] = 0.0
    suspicious = np.argwhere(z_scores > z_threshold)
    fraction = len(suspicious) / max(int(smooth.sum()), 1)

    if len(suspicious) < min_blocks:
        return ForensicsFinding(
            check="error_level_analysis",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail=(
                f"Compression error consistent across the document "
                f"(peak {z_scores.max():.1f} SD above median, threshold {z_threshold:.1f})"
            ),
        )

    if fraction > MAX_OUTLIER_FRACTION:
        return ForensicsFinding(
            check="error_level_analysis",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail=(
                f"Elevated compression error across {fraction:.0%} of the document, too "
                f"widespread to attribute to a local edit; treated as a low-quality capture"
            ),
        )

    regions = [
        Region(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0)
        for x1, y1, x2, y2 in merge_boxes(blocks_to_boxes(suspicious, BLOCK_SIZE))
    ]

    peak_z = float(z_scores.max())
    # Map the peak z-score onto 0-1 with a soft ceiling; 8 SD is already extreme.
    confidence = float(np.clip((peak_z - z_threshold) / 8.0, 0.0, 1.0))

    return ForensicsFinding(
        check="error_level_analysis",
        tamper_type=TamperType.RECOMPRESSION,
        flagged=True,
        confidence=confidence,
        detail=(
            f"{len(regions)} region(s) show compression artifacts inconsistent with "
            f"the rest of the document (peak {peak_z:.1f} SD above the median error). "
            f"Consistent with content pasted or re-rendered after the original scan."
        ),
        regions=regions,
    )


def render_ela_visualization(image: np.ndarray, quality: int = DEFAULT_QUALITY) -> np.ndarray:
    """Return a contrast-stretched ELA map as a BGR image, for the evidence panel."""
    error_map = compute_ela_map(image, quality=quality)
    peak = float(error_map.max())
    if peak <= 0:
        return np.zeros_like(image)
    scaled = np.clip(error_map * (255.0 / peak), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
