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
    "cnn_classifier": 0.0,  # raised once a trained model is wired in
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
    """Weighted combination of flagged findings, normalized to 0-1."""
    total_weight = sum(
        CHECK_WEIGHTS.get(f.check, 0.0) for f in findings if f.check in CHECK_WEIGHTS
    )
    if total_weight <= 0:
        return 0.0

    weighted = sum(
        CHECK_WEIGHTS.get(f.check, 0.0) * f.confidence
        for f in findings
        if f.flagged
    )
    return float(min(weighted / total_weight, 1.0))


def analyze(
    image: np.ndarray,
    *,
    include_cnn: bool = True,
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
