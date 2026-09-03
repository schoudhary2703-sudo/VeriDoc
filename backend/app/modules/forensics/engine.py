"""Forensics engine: runs every detector and combines their findings.

Stage 3 of the pipeline. Classical detectors run first and always; the CNN
classifier joins them once trained and is treated as one more finding, never as
an override. That ordering is deliberate -- BUILD_PLAN calls for explainable
checks before the learned model, so a verdict never rests on a number nobody can
account for.

Important: this must be handed the *unenhanced* image. Glare reduction and
denoising in stage 1 rewrite precisely the pixel statistics measured here, which
is why `PreprocessResult` exposes `for_forensics` separately from `for_ocr`.
"""

from __future__ import annotations

import time

import numpy as np

from app.core.schemas import ForensicsFinding, ForensicsResult
from app.modules.forensics import copy_move, ela

# Per-check weights for the combined score. Noise inconsistency is the strongest
# single indicator of a substituted photograph -- the attack this system exists
# to catch -- while ELA is the noisiest signal and is weighted accordingly.
CHECK_WEIGHTS: dict[str, float] = {
    "error_level_analysis": 0.60,
    "copy_move_detection": 0.40,
    # Advisory only -- see NOISE_DETECTOR_VALIDATED in copy_move.py. It reports a
    # measurement but must not move the verdict until calibrated on real captures.
    "noise_consistency": 0.0,
    # Intra-document face consistency. Measured on FantasyID's held-out split it
    # detects 11.2% of face swaps at 0.0% false positives, and -- the reason it
    # earns real weight -- its detections have ZERO overlap with the classical
    # checks, lifting face-swap recall from 28.7% to 40.0% at no cost in false
    # positives. See app/modules/face/face_match.py.
    "intra_document_face_consistency": 0.40,
    # Stays at zero: three training attempts all scored at or below chance.
    # See docs/FORENSICS_CNN_ATTEMPTS.md.
    "cnn_classifier": 0.0,
}

# Combined score at or above which the document is called tampered.
#
# Set empirically on the synthetic forgery set: the genuine control scores 0.000
# and the weakest forgery (a painted-over date of birth) scores 0.250, so 0.20
# separates them. Conservative by design -- this is decision support, and a false
# positive costs a real traveller real time at a border.
#
# Stated plainly: one genuine control is not a calibration set. This threshold
# must be re-fitted against IDNet before any accuracy figure is published, and
# the gap above is a sanity check, not a measured false-positive rate.
TAMPER_THRESHOLD = 0.20


def _combine(findings: list[ForensicsFinding]) -> float:
    """Combine flagged findings into a 0-1 score using a weighted noisy-OR.

    Deliberately NOT a weighted mean. Evidence of tampering from one check must
    not be diluted by other checks finding nothing: a forger only has to leave a
    single trace, and absence of evidence in the ELA is not evidence of absence
    for the face cross-check. Independent detectors accumulate.

    This was learned by measuring. With a weighted mean, adding the
    intra-document face check dropped held-out attack detection from 18% to 15%
    and face-swap detection from 29% to 25% -- purely because a face check that
    ran and found nothing enlarged the denominator and pushed genuine ELA
    detections below the threshold. The checks were never in conflict; the
    arithmetic was.

    Each flagged finding contributes `weight * confidence` as an independent
    probability of tampering, so the score rises as evidence accumulates and
    never falls when an unrelated check stays quiet.
    """
    survival = 1.0
    for finding in findings:
        if not finding.flagged or not finding.applicable:
            continue
        weight = CHECK_WEIGHTS.get(finding.check, 0.0)
        if weight <= 0.0:
            continue
        survival *= 1.0 - min(weight * finding.confidence, 1.0)

    return float(min(1.0 - survival, 1.0))


def analyze(
    image: np.ndarray,
    *,
    include_cnn: bool = True,
    include_face: bool = True,
) -> ForensicsResult:
    """Run all available forensic checks over a document image.

    `include_cnn` is honoured only when a trained checkpoint is present; the
    classical checks always run, so the engine degrades to explainable-only
    rather than failing when no model has been trained yet.
    """
    started = time.perf_counter()

    findings: list[ForensicsFinding] = [
        copy_move.detect_splice(image),
        copy_move.detect_copy_move(image),
        ela.analyze(image),
    ]

    if include_face:
        face_finding = _try_face_consistency(image)
        if face_finding is not None:
            findings.append(face_finding)

    if include_cnn:
        cnn_finding = _try_cnn(image)
        if cnn_finding is not None:
            findings.append(cnn_finding)

    score = _combine(findings)

    return ForensicsResult(
        tampered=score >= TAMPER_THRESHOLD,
        score=score,
        findings=findings,
        processing_time_ms=int((time.perf_counter() - started) * 1000),
    )


def _try_face_consistency(image: np.ndarray) -> ForensicsFinding | None:
    """Cross-check the two printed portraits, when InsightFace is installed.

    Costs roughly 1.5 s per document on CPU. That is the price of taking
    face-swap recall from 28.7% to 40.0% without introducing a single false
    positive, which is a trade worth making at a checkpoint.
    """
    try:
        from app.modules.face import face_match
    except ImportError:
        return None

    if not face_match.is_available():
        return None

    try:
        finding = face_match.check_intra_document_consistency(image)
    except Exception:  # noqa: BLE001 - a face-model failure must not fail the pipeline
        return None

    if "no second portrait" in finding.detail:
        finding.applicable = False
    return finding


def _try_cnn(image: np.ndarray) -> ForensicsFinding | None:
    """Run the CNN classifier if one is trained and loadable, else return None."""
    try:
        from app.modules.forensics.cnn_classifier import classify, is_model_available
    except ImportError:
        return None

    if not is_model_available():
        return None

    try:
        return classify(image)
    except Exception:  # noqa: BLE001 - a broken checkpoint must not fail the pipeline
        return None
