"""Phase 3 tests: passive liveness cues.

The cues are measured, not trusted. The load-bearing assertion in this file is
`test_uncalibrated_module_never_reports_a_pass`: with no labelled live-versus-
spoof data, the module must decline to judge rather than guess. Three components
in this project have already been measured at or below chance after being
shipped on the assumption they worked, and this is the guard against a fourth.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.modules.face import liveness


def _synthetic_face(seed: int = 0) -> np.ndarray:
    """A textured, skin-toned patch. Not a face, but exercises every cue."""
    rng = np.random.default_rng(seed)
    base = np.zeros((200, 200, 3), dtype=np.float32)
    base[:, :] = (120, 150, 190)                       # BGR skin-ish
    base += rng.normal(0, 9, base.shape)               # micro-texture
    cv2.circle(base, (150, 60), 6, (250, 250, 250), -1)  # a small highlight
    return np.clip(base, 0, 255).astype(np.uint8)


def _blur_and_flatten(face: np.ndarray) -> np.ndarray:
    """Stands in for a print: micro-texture lost, colour gamut compressed."""
    out = cv2.GaussianBlur(face, (5, 5), 1.4)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] = 128 + (lab[:, :, 1] - 128) * 0.45
    lab[:, :, 2] = 128 + (lab[:, :, 2] - 128) * 0.45
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _screen_grid(face: np.ndarray, period: float = 4.0) -> np.ndarray:
    """Stands in for a replay: a periodic display grid beating against the sensor.

    A real face has no such regular structure; a re-photographed screen does.
    """
    h, w = face.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    grid = 12 * np.sin(2 * np.pi * xx / period) + 12 * np.sin(2 * np.pi * yy / period)
    return np.clip(face.astype(np.float32) + grid[:, :, None], 0, 255).astype(np.uint8)


class TestUncalibratedByDefault:
    def test_module_declares_itself_uncalibrated(self) -> None:
        assert liveness.LIVENESS_CALIBRATED is False

    def test_uncalibrated_module_never_reports_a_pass(self) -> None:
        """No labelled data means no verdict — not a default of 'live'."""
        result = liveness.measure(_synthetic_face())
        assert result.performed is True
        assert result.calibrated is False
        assert result.passed is None, "an uncalibrated check must not claim a pass"

    def test_cues_do_not_claim_suspicion_either(self) -> None:
        result = liveness.measure(_synthetic_face())
        assert result.cues
        assert all(c.suspicious is None for c in result.cues)

    def test_detail_says_it_does_not_contribute(self) -> None:
        detail = liveness.measure(_synthetic_face()).detail.lower()
        assert "not" in detail and "verdict" in detail


class TestCueMechanics:
    """The cues must respond in the documented direction to be worth calibrating."""

    def test_every_cue_is_measured(self) -> None:
        result = liveness.measure(_synthetic_face())
        assert {c.name for c in result.cues} == set(liveness.CUE_FUNCTIONS)

    def test_micro_texture_falls_when_detail_is_lost(self) -> None:
        face = _synthetic_face()
        assert liveness.micro_texture_energy(_blur_and_flatten(face)) < liveness.micro_texture_energy(face)

    def test_chroma_spread_falls_when_gamut_is_compressed(self) -> None:
        face = _synthetic_face()
        assert liveness.chroma_spread(_blur_and_flatten(face)) < liveness.chroma_spread(face)

    def test_cues_are_deterministic(self) -> None:
        face = _synthetic_face(7)
        assert liveness.micro_texture_energy(face) == liveness.micro_texture_energy(face)

    def test_moire_responds_to_a_screen_grid(self) -> None:
        """The cue must fire on the periodic pattern it exists to catch."""
        face = _synthetic_face()
        assert liveness.moire_interference(_screen_grid(face)) > liveness.moire_interference(face)

    def test_moire_is_not_confounded_by_blur(self) -> None:
        """The confound this rework fixes: a soft print must NOT out-score a screen.

        The previous implementation measured spectral falloff, so a blurred print
        scored higher than an actual screen replay. After subtracting the isotropic
        radial background the cue is blur-invariant, so a screen grid must win and
        blur must not lift the cue above the live baseline.
        """
        face = _synthetic_face()
        live = liveness.moire_interference(face)
        blurred = liveness.moire_interference(_blur_and_flatten(face))
        screen = liveness.moire_interference(_screen_grid(face))
        assert screen > blurred, "screen replay must out-score a blurred print"
        assert blurred <= live + 0.05, "blur must not inflate the moire cue"

    def test_cue_values_are_finite(self) -> None:
        for cue in liveness.measure(_synthetic_face()).cues:
            assert np.isfinite(cue.value)


class TestDegradation:
    def test_flat_image_does_not_crash_any_cue(self) -> None:
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = liveness.measure(flat)
        assert result.performed
        assert all(np.isfinite(c.value) for c in result.cues)

    def test_missing_face_is_reported_not_raised(self) -> None:
        from app.modules.face import face_match

        if not face_match.is_available():
            pytest.skip("insightface not installed")

        blank = np.full((400, 400, 3), 200, dtype=np.uint8)
        result = liveness.measure_largest_face(blank)
        assert result.performed is False
        assert result.passed is None
        assert "no face" in result.detail.lower()
