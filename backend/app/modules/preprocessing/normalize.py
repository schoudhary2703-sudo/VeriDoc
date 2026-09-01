"""Capture pre-processing: deskew, crop to the document, reduce glare.

Stage 1 of the pipeline. Everything downstream -- OCR, MRZ, forensics -- reads
the output of this module, so it is deliberately conservative: each step returns
the input unchanged when it cannot find what it is looking for, rather than
guessing and corrupting the image for every later stage.

One constraint worth stating: glare reduction and denoising alter pixel
statistics, which is exactly what the Phase 2 forensics engine measures. The
forensics stage must therefore run on the *cropped but otherwise unmodified*
image. `preprocess()` returns both, and that is why.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# A document occupying less than this fraction of the frame is assumed to be a
# spurious contour rather than the document itself.
MIN_DOCUMENT_AREA_RATIO = 0.20


@dataclass
class PreprocessResult:
    """Output of stage 1.

    `for_ocr` has had contrast and glare handling applied. `for_forensics` is the
    same crop with pixel statistics untouched.
    """

    for_ocr: np.ndarray
    for_forensics: np.ndarray
    deskew_angle: float
    document_found: bool
    notes: list[str]


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def find_document_corners(image: np.ndarray) -> np.ndarray | None:
    """Locate the document boundary as four corner points, or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Close small gaps so the document border forms one continuous contour.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = image.shape[0] * image.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < frame_area * MIN_DOCUMENT_AREA_RATIO:
            break
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    return None


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype="float32")
    coord_sum = points.sum(axis=1)
    coord_diff = np.diff(points, axis=1).ravel()

    ordered[0] = points[np.argmin(coord_sum)]   # top-left
    ordered[2] = points[np.argmax(coord_sum)]   # bottom-right
    ordered[1] = points[np.argmin(coord_diff)]  # top-right
    ordered[3] = points[np.argmax(coord_diff)]  # bottom-left
    return ordered


def crop_to_document(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Perspective-correct the image onto the document boundary.

    Returns the original image untouched when no plausible document is found --
    a full-frame scan is a perfectly valid input.
    """
    corners = find_document_corners(image)
    if corners is None:
        return image, False

    tl, tr, br, bl = _order_corners(corners)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return image, False

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(_order_corners(corners), destination)
    return cv2.warpPerspective(image, matrix, (width, height)), True


def estimate_skew_angle(image: np.ndarray) -> float:
    """Estimate the page rotation in degrees using the dominant text baseline."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    lines = cv2.HoughLinesP(
        thresh, 1, np.pi / 180, threshold=100, minLineLength=image.shape[1] // 3, maxLineGap=20
    )
    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only near-horizontal lines describe text baselines.
        if -45 < angle < 45:
            angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate the image so text baselines are horizontal."""
    angle = estimate_skew_angle(image)
    if abs(angle) < 0.1:
        return image, 0.0

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def reduce_glare(image: np.ndarray) -> np.ndarray:
    """Flatten specular highlights and even out illumination.

    CLAHE on the L channel of LAB space lifts text out of both over- and
    under-exposed regions without the halos a plain histogram equalization
    produces on laminated documents.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)

    merged = cv2.merge((lightness, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray) -> np.ndarray:
    """Suppress sensor noise while keeping character edges sharp."""
    return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)


def preprocess(image: np.ndarray, *, enhance: bool = True) -> PreprocessResult:
    """Run the full stage-1 chain.

    Set `enhance=False` to crop and deskew only -- the geometry is corrected but
    pixel statistics are left alone.
    """
    notes: list[str] = []

    cropped, found = crop_to_document(image)
    notes.append("Document boundary detected and perspective-corrected" if found
                 else "No document boundary found; using full frame")

    straightened, angle = deskew(cropped)
    if angle:
        notes.append(f"Deskewed by {angle:.2f} degrees")

    forensics_view = straightened

    if enhance:
        ocr_view = denoise(reduce_glare(straightened))
        notes.append("Glare reduction and denoising applied to the OCR view only")
    else:
        ocr_view = straightened

    return PreprocessResult(
        for_ocr=ocr_view,
        for_forensics=forensics_view,
        deskew_angle=angle,
        document_found=found,
        notes=notes,
    )
