"""Face matching and intra-document face consistency.

Two capabilities, both built on InsightFace's pretrained `buffalo_l` ArcFace
embeddings. No training here -- face recognition is a solved problem and
re-solving it with 1,899 documents would be worse than using weights fitted on
millions of identities.

**1. Document photo vs live capture** (stage 4 of the pipeline). Cosine
similarity between embeddings, thresholded. This is the classic use.

**2. Intra-document face consistency** -- a tamper detector that needs no live
capture at all, and the most useful thing found in Phase 3.

Identity documents carry two portraits: the main photograph and a smaller ghost
image. On a genuine card those are physically different renderings of one
photograph, so their embeddings agree well but not perfectly. When a generative
face swap rewrites the card, it re-renders *both* portraits from the same model,
and the two become unnaturally alike.

Measured on 100 genuine and 100 face-swap documents from FantasyID's held-out
test split:

    genuine    intra-document cosine  median 0.729
    face swap  intra-document cosine  median 0.844
    AUC 0.839

Note the direction: **too much agreement is the tamper signal**, which is the
opposite of the intuition that a swap would break consistency. Operating points
measured on that sample:

    threshold 0.884   recall 18.2%   false positives 0.0%   <- default
    threshold 0.857   recall 39.4%   false positives 5.2%

The default sits at the zero-false-positive point deliberately, in line with the
project's stance that a border system must not accuse genuine travellers.

Crucially this signal is **orthogonal to the classical forensic checks**. On 80
genuine and 80 face-swap documents:

    intra-document only   11.2% recall
    classical only        28.7% recall
    union                 40.0% recall, with 0.0% overlap and 0.0% false positives

Zero overlap: the two catch disjoint sets of forgeries, so the union is worth
more than either alone.
"""

from __future__ import annotations

import functools
import time

import cv2
import numpy as np

from app.core.schemas import (
    FaceMatchResult,
    ForensicsFinding,
    Region,
    TamperType,
)

MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)

# Cosine similarity above which a document photo and a live capture are taken to
# be the same person. This is InsightFace's published guidance for w600k_r50, NOT
# a value calibrated on our own data -- we have no document+live pairs yet. It
# must be tuned against real capture hardware before any EER is quoted.
DEFAULT_MATCH_THRESHOLD = 0.40

# Intra-document similarity at or above which the two on-card portraits agree so
# closely that a shared generative origin is the better explanation. Measured at
# the zero-false-positive point; see the module docstring.
INTRA_DOC_THRESHOLD = 0.884


class FaceEngineError(RuntimeError):
    """Raised when InsightFace is unavailable."""


def is_available() -> bool:
    from importlib.util import find_spec

    return find_spec("insightface") is not None and find_spec("onnxruntime") is not None


@functools.lru_cache(maxsize=1)
def _analyzer():
    """Load buffalo_l once. First call downloads ~281 MB of model weights."""
    if not is_available():
        raise FaceEngineError("insightface and onnxruntime are required for face matching")

    from insightface.app import FaceAnalysis

    # Only detection and recognition are loaded. buffalo_l also ships two
    # landmark models and a gender/age estimator, which FaceAnalysis runs on
    # every detected face by default -- and whose output this project never
    # reads. Beyond the wasted time, running an age and gender classifier on
    # travellers is not something a border system should be doing incidentally.
    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=DETECTION_SIZE)
    return app


def _face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return float((x2 - x1) * (y2 - y1))


# Border added around an image when a first detection pass finds nothing, as a
# fraction of the longest side.
RETRY_PAD_RATIO = 0.35


def detect_faces(image: np.ndarray, *, allow_padded_retry: bool = True) -> list:
    """Return detected faces, largest first.

    Retries once with a padded border when nothing is found. RetinaFace's anchors
    expect a face to occupy a modest fraction of the frame, so a tightly cropped
    head-shot is missed entirely: a 470x622 crop containing a 390x542 face
    returned zero detections, while the same face with more margin around it
    returned one.

    That is a real capture condition, not a laboratory curiosity -- a traveller
    leaning towards the camera produces exactly this framing, and without the
    retry the face match would silently report "not performed" rather than
    comparing the faces it was given.
    """
    faces = sorted(_analyzer().get(image), key=_face_area, reverse=True)
    if faces or not allow_padded_retry:
        return faces

    pad = int(max(image.shape[:2]) * RETRY_PAD_RATIO)
    if pad <= 0:
        return faces

    # A neutral constant border, not BORDER_REPLICATE. Replicating the edge
    # pixels smears them into streaks that read as structure and suppress
    # detection: at this pad ratio REPLICATE still found nothing, while a flat
    # grey border found the face.
    padded = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    retried = sorted(_analyzer().get(padded), key=_face_area, reverse=True)

    # Map bounding boxes back onto the original frame so callers' coordinates
    # stay meaningful. Embeddings are unaffected -- the pixels are identical.
    for face in retried:
        face.bbox = face.bbox - pad
        if getattr(face, "kps", None) is not None:
            face.kps = face.kps - pad

    return retried


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def match_document_to_capture(
    document_image: np.ndarray,
    capture_image: np.ndarray,
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> FaceMatchResult:
    """Compare the portrait on a document against a live capture.

    Returns `performed=False` rather than raising when either side has no
    detectable face: a missing face is a finding for the risk scorer, not an
    error, and a document held at the wrong angle should not fail the request.
    """
    document_faces = detect_faces(document_image)
    capture_faces = detect_faces(capture_image)

    if not document_faces or not capture_faces:
        which = "document" if not document_faces else "live capture"
        return FaceMatchResult(
            performed=False,
            faces_in_document=len(document_faces),
            faces_in_capture=len(capture_faces),
            threshold=threshold,
            detail=f"No face detected in the {which}; face match could not be performed",
        )

    score = cosine_similarity(
        document_faces[0].embedding, capture_faces[0].embedding
    )
    matched = score >= threshold

    return FaceMatchResult(
        performed=True,
        match_score=round(score, 4),
        matched=matched,
        threshold=threshold,
        faces_in_document=len(document_faces),
        faces_in_capture=len(capture_faces),
        detail=(
            f"Document portrait and live capture match at cosine {score:.3f} "
            f"(threshold {threshold:.2f})"
            if matched
            else f"Document portrait does not match the live capture: cosine "
            f"{score:.3f} is below the {threshold:.2f} threshold"
        ),
    )


def check_intra_document_consistency(
    image: np.ndarray,
    *,
    threshold: float = INTRA_DOC_THRESHOLD,
) -> ForensicsFinding:
    """Compare the two portraits printed on one document.

    Requires two detectable faces (main portrait plus ghost image). Documents
    with a single portrait are reported as not applicable rather than as passing,
    since an absent test is not evidence of authenticity.
    """
    faces = detect_faces(image)

    if len(faces) < 2:
        return ForensicsFinding(
            check="intra_document_face_consistency",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail=(
                f"Only {len(faces)} face region(s) found; this document layout has no "
                f"second portrait to cross-check against"
            ),
        )

    primary, secondary = faces[0], faces[1]
    similarity = cosine_similarity(primary.embedding, secondary.embedding)

    boxes = [
        Region(
            x1=int(face.bbox[0]), y1=int(face.bbox[1]),
            x2=int(face.bbox[2]), y2=int(face.bbox[3]),
            score=round(similarity, 4),
        )
        for face in (primary, secondary)
    ]

    if similarity < threshold:
        return ForensicsFinding(
            check="intra_document_face_consistency",
            tamper_type=None,
            flagged=False,
            confidence=0.0,
            detail=(
                f"The two portraits on this document agree at cosine {similarity:.3f}, "
                f"within the range expected for a genuine card (threshold {threshold:.3f})"
            ),
            regions=boxes,
        )

    # Confidence is floored, not scaled from zero. The threshold was chosen at
    # the zero-false-positive operating point measured over 96 genuine documents,
    # so merely crossing it is already strong evidence -- treating a crossing as
    # near-zero confidence made the check unable to affect any verdict (a
    # similarity of 0.90 contributed 0.055 against a 0.20 decision threshold, so
    # it flagged and changed nothing). Above the floor, confidence still rises
    # towards the degenerate "identical render" case at 1.0.
    excess = (similarity - threshold) / (1.0 - threshold)
    confidence = float(np.clip(0.60 + 0.40 * excess, 0.0, 1.0))

    return ForensicsFinding(
        check="intra_document_face_consistency",
        tamper_type=TamperType.PHOTO_SPLICE,
        flagged=True,
        confidence=confidence,
        detail=(
            f"The main portrait and the ghost image agree at cosine {similarity:.3f}, "
            f"above the {threshold:.3f} expected for two separate renderings of one "
            f"photograph. Consistent with both portraits having been re-generated by a "
            f"single face-swapping model rather than printed from one original."
        ),
        regions=boxes,
    )


def analyze_document_faces(image: np.ndarray) -> tuple[ForensicsFinding, int]:
    """Convenience wrapper: the consistency finding plus the face count."""
    started = time.perf_counter()
    finding = check_intra_document_consistency(image)
    del started  # timing is reported by the caller that owns the pipeline stage
    return finding, len(detect_faces(image))
