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
    if lines is None or len(lines) == 0:
        return 0.0

    # OpenCV returns (N, 1, 4) on 4.x and (N, 4) on 5.x. Normalise both to (N, 4)
    # rather than indexing one shape and crashing on the other -- installing
    # InsightFace pulled in a second OpenCV build and silently changed this.
    segments = np.asarray(lines).reshape(-1, 4)

    angles = []
    for x1, y1, x2, y2 in segments:
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


def find_mrz_band(
    image: np.ndarray,
    *,
    search_fraction: float = 0.45,
    min_width_ratio: float = 0.55,
    padding: int = 12,
) -> tuple[int, int, int, int] | None:
    """Locate the machine-readable zone as a (x1, y1, x2, y2) box.

    The MRZ is the densest run of wide, uniform-height text on the document and
    always sits in the lower portion, so we look there for rows whose horizontal
    text density spans most of the page width.

    Worth doing for two reasons. Speed: OCR over the whole document takes 82-93 s
    on CPU, and the band is roughly a fifth of the page. Accuracy: the MRZ is the
    one field with a checksum, so isolating it removes every competing text
    region that OCR might otherwise mis-segment against it.

    Returns None when no band is confidently found, so callers fall back to
    whole-document OCR rather than silently reading the wrong strip.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape

    # Blackhat lifts dark text off a light background.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Close along x so the characters of one MRZ line merge into a single bar.
    closed = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 3))
    )

    search_top = int(height * (1.0 - search_fraction))
    row_coverage = (closed[search_top:] > 0).sum(axis=1) / width

    wide_rows = np.where(row_coverage >= min_width_ratio)[0]
    if wide_rows.size == 0:
        return None

    # Group qualifying rows into runs, tolerating the blank gap *between* MRZ
    # lines. Without this the band collapses onto a single line -- TD3 has two
    # lines and TD1 has three, so returning one of them silently loses half the
    # zone and every check digit on it.
    # Measured on a 660 px specimen: the blank gap between two TD3 lines is 37
    # rows, so a tolerance of 5% (33 px) split the zone in half. Line spacing
    # scales with document height, hence the proportional term.
    max_gap = max(int(height * 0.09), 40)
    runs: list[list[int]] = [[int(wide_rows[0])]]
    for row in wide_rows[1:]:
        if int(row) - runs[-1][-1] <= max_gap:
            runs[-1].append(int(row))
        else:
            runs.append([int(row)])

    # The MRZ is the bottom-most such block on every ICAO layout.
    band = runs[-1]

    y1 = max(0, search_top + band[0] - padding)
    y2 = min(height, search_top + band[-1] + padding)

    if y2 - y1 < 20:
        return None

    return 0, y1, width, y2


def extract_mrz_band(image: np.ndarray, **kwargs) -> np.ndarray | None:
    """Return the cropped MRZ strip, or None when it cannot be located."""
    box = find_mrz_band(image, **kwargs)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


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
