"""Passive liveness cues for presentation-attack detection.

**This module ships UNCALIBRATED and does not affect any verdict.** It measures
and reports; it does not decide. `LIVENESS_CALIBRATED` is `False`, and until it
is `True` every result carries `passed=None` rather than a pass or a fail.

The reason is that we have no labelled data. CelebA-Spoof, OULU-NPU and
Replay-Attack are all research-licence only and were not obtained; FantasyID
contains only printed-and-recaptured cards, which is a single class with nothing
to discriminate against. A detector fitted to no data is a guess wearing a
number, and this project has already measured three of those: the tamper CNN
scored at or below chance in three configurations, and the noise-consistency
check rated genuine documents as more anomalous than forgeries. Shipping a
fourth would be a choice, not an accident.

What this module *is* good for: the cues below are real, individually
explainable, and the thresholds can be fitted in about twenty minutes from
photographs the team can take themselves — no licensed dataset required. See
`ml/calibrate_liveness.py`.

The four cues, and the attack each addresses:

**Moire interference** catches a *replay* attack -- a face shown on a screen. The
display's pixel grid beats against the camera's sensor grid and produces regular
off-axis peaks in the frequency domain that no real face generates. The cue
isolates those peaks by first subtracting the spectrum's isotropic radial
falloff -- the only thing blur changes -- so a soft print no longer reads as a
screen. See the reworked implementation and the confound it fixes below.

**Specular concentration** catches glossy prints and screens. Real skin scatters
light diffusely across many small highlights; a flat glossy surface reflects it
back as a few large, concentrated ones.

**Micro-texture energy** catches *print* attacks. Skin pores and fine hair carry
high-frequency detail that a print's halftone dots and a screen's subpixels
cannot reproduce, so a re-captured face is measurably smoother.

**Chroma spread** catches both. Printer inks and display phosphors cover a
narrower gamut than skin under real illumination, so the colour distribution of a
re-captured face is compressed.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.schemas import LivenessCue, LivenessResult

# Flip to True only when thresholds below have been fitted on real live-versus-
# spoof captures, and record what they were fitted on in docs/DATASETS.md.
LIVENESS_CALIBRATED = False

# Placeholder thresholds. These are plausible starting points, NOT measurements:
# nothing in this project has been evaluated against them, and they exist so the
# calibration script has something to overwrite.
PLACEHOLDER_THRESHOLDS = {
    "moire_interference": 0.18,
    "specular_concentration": 0.35,
    "micro_texture_energy": 12.0,
    "chroma_spread": 14.0,
}

FACE_CROP_SIZE = 224


def _prepare(face_image: np.ndarray) -> np.ndarray:
    """Square, resized face crop so cue values are comparable across captures."""
    return cv2.resize(
        face_image, (FACE_CROP_SIZE, FACE_CROP_SIZE), interpolation=cv2.INTER_AREA
    )


def moire_interference(face: np.ndarray) -> float:
    """Prominence of off-axis periodic frequency peaks. Higher suggests a screen.

    A display re-photographed by a camera produces a beat pattern between the two
    pixel grids. That shows up as isolated peaks away from the frequency origin,
    which natural images -- whose spectra fall off smoothly -- do not have.

    The earlier implementation measured global peakiness as ``top / median`` over
    an annular band, and it was confounded by blur: a soft print has a spectrum
    that falls off steeply, which collapses the band median and inflates that
    ratio, so a print scored *higher* than an actual screen replay (measured
    0.234 on a blurred print versus 0.044 on a synthetic screen grid). It was
    reading spectral falloff, not periodicity.

    This version removes the confound by subtracting the spectrum's isotropic
    radial background -- the mean magnitude at each radius -- before looking for
    peaks. Blur changes only that isotropic falloff, so subtracting it makes the
    cue blur-invariant; what survives is *anisotropic, localized* structure,
    which is exactly what a screen's beat frequencies are. The score is then how
    far the strongest surviving peak stands above the band's own noise floor, in
    robust MAD units, so it does not depend on absolute spectral scale either.
    """
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = gray - gray.mean()

    # 2-D Hann window so image edges do not leak energy into false off-axis peaks.
    window = np.hanning(gray.shape[0])[:, None] * np.hanning(gray.shape[1])[None, :]
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray * window))))

    centre = np.array(spectrum.shape) // 2
    y, x = np.ogrid[: spectrum.shape[0], : spectrum.shape[1]]
    radius = np.hypot(y - centre[0], x - centre[1])

    # Subtract the isotropic radial background: the mean magnitude at each radius.
    # Overall spectral falloff (the only thing blur alters) is isotropic, so this
    # cancels it and leaves the directional peaks a screen grid produces.
    radial_bin = radius.astype(np.int32)
    totals = np.bincount(radial_bin.ravel(), spectrum.ravel())
    counts = np.bincount(radial_bin.ravel())
    radial_mean = totals / np.maximum(counts, 1)
    residual = spectrum - radial_mean[radial_bin]

    # Ignore the DC region and the very top of the band, where sensor noise lives.
    band = (radius > spectrum.shape[0] * 0.10) & (radius < spectrum.shape[0] * 0.45)
    values = residual[band]
    if values.size == 0:
        return 0.0

    # How far the strongest peak sits above the band's own noise floor, measured
    # in MAD units so a few sharp peaks cannot inflate the scale they are judged
    # against. A smooth (natural or blurred) spectrum leaves only small residuals.
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-6
    top = float(np.quantile(values, 0.9995))
    prominence = (top - median) / mad
    return float(np.clip(prominence / 30.0, 0.0, 1.0))


def specular_concentration(face: np.ndarray) -> float:
    """How concentrated the bright highlights are. Higher suggests a flat surface.

    Real skin produces many small diffuse highlights; a glossy print or a screen
    reflects the light source back as a few large ones.
    """
    lightness = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)[:, :, 0]
    threshold = max(int(np.quantile(lightness, 0.98)), 200)
    bright = (lightness >= threshold).astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if count <= 1:
        return 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    total = float(areas.sum())
    if total <= 0:
        return 0.0

    # Fraction of highlight area held by the single largest blob.
    return float(np.clip(areas.max() / total, 0.0, 1.0))


def micro_texture_energy(face: np.ndarray) -> float:
    """High-frequency detail in the skin. Lower suggests a print or screen."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Median-blur removes noise without erasing pores, so what remains is texture.
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    return float(np.std(residual) * 10.0)


def chroma_spread(face: np.ndarray) -> float:
    """Spread of the a/b colour channels. Lower suggests a compressed gamut."""
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    return float(np.sqrt(lab[:, :, 1].std() ** 2 + lab[:, :, 2].std() ** 2))


CUE_FUNCTIONS = {
    "moire_interference": (moire_interference, "above", "screen replay"),
    "specular_concentration": (specular_concentration, "above", "glossy print or screen"),
    "micro_texture_energy": (micro_texture_energy, "below", "print or screen"),
    "chroma_spread": (chroma_spread, "below", "print or screen"),
}


def measure(face_image: np.ndarray) -> LivenessResult:
    """Measure every passive cue on a face crop.

    Returns `passed=None` while `LIVENESS_CALIBRATED` is False: an uncalibrated
    check has not examined this capture and cleared it, so reporting a pass would
    be a claim the module cannot support.
    """
    face = _prepare(face_image)
    cues: list[LivenessCue] = []

    for name, (function, direction, attack) in CUE_FUNCTIONS.items():
        value = round(function(face), 4)
        threshold = PLACEHOLDER_THRESHOLDS[name]
        suspicious = value > threshold if direction == "above" else value < threshold

        cues.append(
            LivenessCue(
                name=name,
                value=value,
                threshold=threshold,
                direction=direction,
                suspicious=suspicious if LIVENESS_CALIBRATED else None,
                detail=(
                    f"{name.replace('_', ' ')} measured {value:.3f}; a value "
                    f"{direction} {threshold} would indicate {attack}."
                ),
            )
        )

    if not LIVENESS_CALIBRATED:
        return LivenessResult(
            performed=True,
            calibrated=False,
            passed=None,
            cues=cues,
            detail=(
                "Liveness cues measured but NOT evaluated: the thresholds have "
                "never been fitted against real live-versus-spoof captures, so no "
                "pass or fail can be claimed. This check does not contribute to "
                "the verdict."
            ),
        )

    flagged = [c for c in cues if c.suspicious]
    return LivenessResult(
        performed=True,
        calibrated=True,
        passed=not flagged,
        cues=cues,
        detail=(
            "All passive liveness cues fall within the range expected for a live "
            "capture."
            if not flagged
            else "Presentation-attack indicators: "
            + ", ".join(c.name.replace("_", " ") for c in flagged)
        ),
    )


def measure_largest_face(image: np.ndarray) -> LivenessResult:
    """Locate the largest face in `image` and measure liveness cues on it."""
    from app.modules.face import face_match

    if not face_match.is_available():
        return LivenessResult(
            performed=False,
            calibrated=LIVENESS_CALIBRATED,
            detail="InsightFace is not installed; liveness cues not measured.",
        )

    faces = face_match.detect_faces(image)
    if not faces:
        return LivenessResult(
            performed=False,
            calibrated=LIVENESS_CALIBRATED,
            detail="No face detected in the capture; liveness cues not measured.",
        )

    x1, y1, x2, y2 = (int(v) for v in faces[0].bbox)
    height, width = image.shape[:2]
    crop = image[max(y1, 0) : min(y2, height), max(x1, 0) : min(x2, width)]

    if crop.size == 0 or min(crop.shape[:2]) < 32:
        return LivenessResult(
            performed=False,
            calibrated=LIVENESS_CALIBRATED,
            detail="Detected face is too small to measure liveness cues reliably.",
        )

    return measure(crop)
