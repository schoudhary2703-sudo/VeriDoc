"""Copy-move and splice detection using classical CV.

Two distinct attacks, two distinct methods.

**Splicing** pastes content from elsewhere -- the substituted photograph being the
attack this system exists to catch. The pasted region rarely matches the host's
noise floor, because sensor noise and JPEG history are capture-specific. Detected
by finding blocks whose high-frequency residual energy departs from the
document's own distribution.

**Copy-move** duplicates a region from within the same document -- cloning clean
background over a field, or repeating a security feature. Detected by matching
ORB keypoints against themselves and requiring a geometrically consistent block.

A warning learned the hard way, preserved here because it is not obvious: on
identity documents, *periodic typography defeats naive copy-move detection*. The
MRZ is monospace and full of repeated '<' fillers, so its glyphs match each other
at a constant horizontal pitch -- 73 keypoints sharing an offset of exactly
(+104, 0) on a genuine specimen. Geometric consistency alone therefore proves
nothing. Text regions are excluded outright, and duplicated regions are required
to be two-dimensional, because cloned content is a block while repeated text is a
line.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.schemas import ForensicsFinding, Region, TamperType
from app.modules.forensics.common import (
    block_reduce,
    blocks_to_boxes,
    merge_boxes,
    noise_residual,
    robust_z,
    structure_density,
    to_gray,
)

MIN_MATCH_DISTANCE_PX = 40
MAX_DESCRIPTOR_DISTANCE = 30
MIN_CLUSTERED_MATCHES = 12
OFFSET_BIN_PX = 8

# A cloned region is a block; repeated text is a line. Require the matched
# cluster to have real extent in both axes.
MIN_CLUSTER_HEIGHT_PX = 24
MIN_CLUSTER_WIDTH_PX = 24

NOISE_BLOCK_SIZE = 32
NOISE_Z_THRESHOLD = 4.0

# The noise-consistency check is ADVISORY until validated on real captured
# documents. Measured on synthetic specimens it does not discriminate at all:
# the genuine control peaked at 47.2 SD while the forgeries reached 19.6-46.1,
# i.e. the clean document looked *more* anomalous than every forgery. Normalizing
# for local image activity did not help (51.1 vs 37.3-50.1).
#
# The reason is that synthetic renders have no real sensor-noise field. JPEG
# residual energy tracks local content instead, so a flat cream background and a
# grey photo box differ legitimately. The method is sound on genuine camera
# captures, which is what IDNet, FantasyID and SIDTD provide -- so it stays in
# the code, reports its measurement, and does not flag until re-validated there.
NOISE_DETECTOR_VALIDATED = False

# Blocks with more structure than this are typography or printed rules, not
# surfaces on which a noise floor can be measured.
MAX_STRUCTURE_DENSITY = 0.06

# A tamper covering more than this fraction of the analysable area is not an
# "outlier" any more -- the statistics would be describing the tamper itself.
MAX_OUTLIER_FRACTION = 0.45


def _text_mask(gray: np.ndarray, block: int = 16) -> np.ndarray:
    """Full-resolution boolean mask, True where the pixel is in a text-dense block."""
    density = structure_density(gray, block)
    dense = (density > MAX_STRUCTURE_DENSITY).astype(np.uint8)
    if dense.size == 0:
        return np.zeros(gray.shape, dtype=bool)

    # Grow slightly so glyph edges near a block boundary are covered too.
    dense = cv2.dilate(dense, np.ones((3, 3), np.uint8), iterations=1)
    upscaled = cv2.resize(
        dense, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return upscaled.astype(bool)


def detect_copy_move(
    image: np.ndarray,
    *,
    max_descriptor_distance: int = MAX_DESCRIPTOR_DISTANCE,
    min_match_distance: int = MIN_MATCH_DISTANCE_PX,
    min_matches: int = MIN_CLUSTERED_MATCHES,
    offset_tolerance: int = OFFSET_BIN_PX,
) -> ForensicsFinding:
    """Find regions duplicated from elsewhere in the same document."""
    gray = to_gray(image)
    text = _text_mask(gray)

    orb = cv2.ORB_create(nfeatures=5000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 2:
        return ForensicsFinding(
            check="copy_move_detection",
            tamper_type=None,
            flagged=False,
            applicable=False,
            confidence=0.0,
            detail="Too few distinctive features to test for duplicated regions",
        )

    # Drop keypoints sitting on printed text before matching.
    height, width = gray.shape
    keep = [
        i
        for i, kp in enumerate(keypoints)
        if not text[min(int(kp.pt[1]), height - 1), min(int(kp.pt[0]), width - 1)]
    ]
    if len(keep) < 2:
        return ForensicsFinding(
            check="copy_move_detection",
            tamper_type=None,
            flagged=False,
            applicable=False,
            confidence=0.0,
            detail="Document is almost entirely printed text; copy-move test not applicable",
        )

    keypoints = [keypoints[i] for i in keep]
    descriptors = descriptors[keep]

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(descriptors, descriptors, k=2)

    offset_bins: dict[tuple[int, int], list[tuple[float, float]]] = {}
    candidate_count = 0

    for match_group in matches:
        if len(match_group) < 2:
            continue
        candidate = match_group[1]  # index 0 is the self-match
        if candidate.distance > max_descriptor_distance:
            continue

        source = keypoints[candidate.queryIdx].pt
        target = keypoints[candidate.trainIdx].pt
        dx, dy = target[0] - source[0], target[1] - source[1]
        if np.hypot(dx, dy) < min_match_distance:
            continue

        candidate_count += 1
        if (dx, dy) < (0.0, 0.0):  # normalize so A->B and B->A share a bin
            dx, dy = -dx, -dy
        key = (int(round(dx / offset_tolerance)), int(round(dy / offset_tolerance)))
        offset_bins.setdefault(key, []).extend([source, target])

    def no_finding(detail: str) -> ForensicsFinding:
        return ForensicsFinding(
            check="copy_move_detection",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail=detail,
        )

    if not offset_bins:
        return no_finding("No duplicated regions found outside printed text")

    dominant_key = max(offset_bins, key=lambda k: len(offset_bins[k]))
    points = np.array(offset_bins[dominant_key], dtype=np.float32)
    pair_count = len(points) // 2

    if pair_count < min_matches:
        return no_finding(
            f"No cloned region found: {candidate_count} similar features spread across "
            f"{len(offset_bins)} unrelated offsets, consistent with repeated printing "
            f"(largest consistent group {pair_count}, {min_matches} required)"
        )

    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    cluster_w, cluster_h = x2 - x1, y2 - y1

    if cluster_h < MIN_CLUSTER_HEIGHT_PX or cluster_w < MIN_CLUSTER_WIDTH_PX:
        return no_finding(
            f"Matched features form a {int(cluster_w)}x{int(cluster_h)} px line rather "
            f"than a block, consistent with repeated characters rather than cloning"
        )

    confidence = float(np.clip(pair_count / (min_matches * 4), 0.0, 1.0))
    shift_x = dominant_key[0] * offset_tolerance
    shift_y = dominant_key[1] * offset_tolerance

    return ForensicsFinding(
        check="copy_move_detection",
        tamper_type=TamperType.COPY_MOVE,
        flagged=True,
        confidence=confidence,
        detail=(
            f"{pair_count} keypoints outside printed text share a single translation of "
            f"about ({shift_x:+d}, {shift_y:+d}) px, indicating a block of content copied "
            f"and pasted elsewhere on the same document."
        ),
        regions=[Region(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), score=confidence)],
    )


def detect_splice(
    image: np.ndarray,
    *,
    block: int = NOISE_BLOCK_SIZE,
    z_threshold: float = NOISE_Z_THRESHOLD,
) -> ForensicsFinding:
    """Find regions whose sensor-noise floor differs from the rest of the document."""
    gray = to_gray(image)
    energy = block_reduce(noise_residual(gray), block, how="std")

    if energy.size == 0 or min(energy.shape) < 2:
        return ForensicsFinding(
            check="noise_consistency",
            tamper_type=None,
            flagged=False,
            applicable=False,
            confidence=0.0,
            detail="Image too small for block-level noise analysis",
        )

    # Structure density is measured on a denoised image, so a heavily grained
    # spliced patch is NOT mistaken for text and excluded from its own test.
    density = structure_density(gray, block)
    smooth = density <= MAX_STRUCTURE_DENSITY

    if smooth.sum() < 8:
        return ForensicsFinding(
            check="noise_consistency",
            tamper_type=None,
            flagged=False,
            applicable=False,
            confidence=0.0,
            detail="Document is too text-dense to measure a reliable noise floor",
        )

    z_scores = np.abs(robust_z(energy, smooth))
    z_scores[~smooth] = 0.0
    outliers = np.argwhere(z_scores > z_threshold)

    fraction = len(outliers) / max(int(smooth.sum()), 1)
    if len(outliers) == 0 or fraction > MAX_OUTLIER_FRACTION:
        reason = (
            f"Noise floor consistent across the document "
            f"(peak deviation {z_scores.max():.1f} SD, threshold {z_threshold:.1f})"
            if len(outliers) == 0
            else (
                f"Noise varies across {fraction:.0%} of the document, too widespread to "
                f"attribute to a local edit; treated as capture variation"
            )
        )
        return ForensicsFinding(
            check="noise_consistency",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail=reason,
        )

    boxes = merge_boxes(blocks_to_boxes(outliers, block))
    regions = [
        Region(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0) for x1, y1, x2, y2 in boxes
    ]
    peak = float(z_scores.max())
    confidence = float(np.clip((peak - z_threshold) / 8.0, 0.0, 1.0))

    if not NOISE_DETECTOR_VALIDATED:
        # applicable=False, not flagged=False. An uncalibrated check has not
        # examined this document and passed it -- it has not meaningfully run at
        # all, and the officer console renders that as "not applicable" rather
        # than as a green tick it has not earned.
        return ForensicsFinding(
            check="noise_consistency",
            tamper_type=None,
            flagged=False,
            applicable=False,
            confidence=0.0,
            detail=(
                f"Advisory only: {len(regions)} region(s) deviate from the document noise "
                f"floor (peak {peak:.1f} SD), but this check is not yet calibrated on real "
                f"captured documents and does not contribute to the verdict."
            ),
            regions=regions,
        )

    return ForensicsFinding(
        check="noise_consistency",
        tamper_type=TamperType.PHOTO_SPLICE,
        flagged=True,
        confidence=confidence,
        detail=(
            f"{len(regions)} region(s) have a noise signature inconsistent with the rest "
            f"of the document (peak {peak:.1f} SD). Consistent with content originating "
            f"from a different source image, such as a substituted photograph."
        ),
        regions=regions,
    )
