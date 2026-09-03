"""Shared statistics helpers for the forensic detectors.

Two lessons are encoded here, both learned by measuring against a genuine
control rather than by assumption:

**Structure masks must be computed on a denoised image.** Canny run on a noisy
region reports dense "edges" that are really grain. An early version masked out
text-heavy blocks to isolate the sensor-noise floor -- and then silently masked
out a spliced patch too, because its heavy grain looked like text. The detector
duly reported the forgery as consistent. Denoise first, then find structure.

**Use robust statistics.** Mean and standard deviation are computed *from* the
data being tested for outliers, so a large tampered region drags the mean toward
itself and shrinks its own z-score -- the bigger the forgery, the less it stands
out. Median and MAD do not have that failure mode.
"""

from __future__ import annotations

import cv2
import numpy as np

# 1.4826 scales the median absolute deviation to be a consistent estimator of
# the standard deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def block_reduce(values: np.ndarray, block: int, how: str = "mean") -> np.ndarray:
    """Reduce a 2-D array to one value per block x block tile."""
    height, width = values.shape
    rows, cols = height // block, width // block
    if rows == 0 or cols == 0:
        return np.zeros((0, 0), dtype=np.float32)

    tiles = values[: rows * block, : cols * block].reshape(rows, block, cols, block)
    if how == "std":
        return tiles.std(axis=(1, 3)).astype(np.float32)
    return tiles.mean(axis=(1, 3)).astype(np.float32)


def structure_density(gray: np.ndarray, block: int) -> np.ndarray:
    """Per-block density of real structure (text strokes, borders, printed rules).

    Two properties this needs, both learned by measurement:

    **Grain must not read as text.** The median blur removes salt-and-pepper
    noise while leaving genuine edges intact, so a heavily grained region is not
    mistaken for typography and excluded from its own analysis.

    **The result must not depend on exposure.** Fixed Canny thresholds collapse
    on a dark capture: at 25% brightness the detected text fraction fell from
    0.16 to 0.04, the machine-readable zone stopped being masked, and its
    repeated '<' fillers were reported as a cloned block on a genuine document --
    at an offset of (+104, 0), the MRZ's own character pitch. Where text *is*
    cannot depend on how brightly it was photographed, so contrast is normalised
    first and the Canny thresholds are derived from the image rather than fixed.
    """
    denoised = cv2.medianBlur(gray, 5)

    # Stretch to full range so a dim capture presents the same contrast as a
    # well-lit one. CLAHE rather than a global stretch: uneven lighting across a
    # document is common, and a global stretch leaves the shadowed half flat.
    normalised = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(denoised)

    # Thresholds from the image's own intensity distribution (the standard
    # median-based heuristic) rather than constants that only suit one exposure.
    median = float(np.median(normalised))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    if upper <= lower:
        lower, upper = 80, 200

    edges = cv2.Canny(normalised, lower, upper)
    return block_reduce(edges.astype(np.float32), block, how="mean") / 255.0


def robust_z(values: np.ndarray, reference_mask: np.ndarray | None = None) -> np.ndarray:
    """Signed robust z-scores using median and MAD.

    `reference_mask` selects which entries define "normal". Everything is scored,
    but only the reference entries set the baseline.
    """
    reference = values if reference_mask is None else values[reference_mask]
    if reference.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))

    scale = mad * MAD_TO_SIGMA
    if scale < 1e-6:
        # Degenerate spread: fall back to standard deviation, then give up.
        scale = float(reference.std())
        if scale < 1e-6:
            return np.zeros_like(values, dtype=np.float32)

    return ((values - median) / scale).astype(np.float32)


def noise_residual(gray: np.ndarray) -> np.ndarray:
    """High-frequency residual: the image minus its own denoised version."""
    denoised = cv2.medianBlur(gray, 3)
    return gray.astype(np.float32) - denoised.astype(np.float32)


def blocks_to_boxes(
    indices: np.ndarray, block: int
) -> list[tuple[int, int, int, int]]:
    """Convert (row, col) block indices into pixel boxes."""
    return [
        (int(c * block), int(r * block), int((c + 1) * block), int((r + 1) * block))
        for r, c in indices
    ]


def merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge overlapping or touching boxes so one edit yields one region."""
    if not boxes:
        return []

    current = list(boxes)
    merged = True
    while merged:
        merged = False
        output: list[tuple[int, int, int, int]] = []
        while current:
            x1, y1, x2, y2 = current.pop()
            keep: list[tuple[int, int, int, int]] = []
            for ox1, oy1, ox2, oy2 in current:
                if not (ox1 > x2 or ox2 < x1 or oy1 > y2 or oy2 < y1):
                    x1, y1 = min(x1, ox1), min(y1, oy1)
                    x2, y2 = max(x2, ox2), max(y2, oy2)
                    merged = True
                else:
                    keep.append((ox1, oy1, ox2, oy2))
            current = keep
            output.append((x1, y1, x2, y2))
        current = output
    return current
