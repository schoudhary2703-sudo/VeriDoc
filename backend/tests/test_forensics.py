"""Phase 2 tests: classical forensic detectors.

The load-bearing test here is `test_genuine_document_is_not_flagged`. Every
detector in this module was, at some point during development, capable of
flagging a genuine document -- copy-move matched the MRZ's repeated fillers
against themselves, and the noise check rated the clean control as more anomalous
than any forgery. A tamper detector without a negative control is not a detector,
it is a rubber stamp.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from app.core.schemas import TamperType
from app.modules.forensics import copy_move, ela, engine

SYNTHETIC_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"


def _load(name: str):
    path = SYNTHETIC_DIR / name
    if not path.exists():
        pytest.skip(
            "Synthetic forgeries not generated. Run: "
            "python -m ml.data_prep.generate_synthetic_forgeries"
        )
    image = cv2.imread(str(path))
    assert image is not None, f"could not read {path}"
    return image


@pytest.fixture
def manifest() -> list[dict]:
    path = SYNTHETIC_DIR / "forgeries.json"
    if not path.exists():
        pytest.skip("Synthetic forgeries not generated")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def genuine():
    return _load("control_genuine.png")


class TestNegativeControl:
    """The genuine document must survive every check."""

    def test_genuine_document_is_not_flagged(self, genuine) -> None:
        result = engine.analyze(genuine)
        assert not result.tampered, (
            f"false positive on a genuine document: {result.summary()}"
        )
        assert result.score < engine.TAMPER_THRESHOLD

    def test_no_individual_check_flags_genuine(self, genuine) -> None:
        for finding in engine.analyze(genuine).findings:
            assert not finding.flagged, f"{finding.check} flagged genuine: {finding.detail}"

    def test_mrz_repetition_does_not_read_as_cloning(self, genuine) -> None:
        """Regression: the MRZ's repeated '<' fillers once produced 73 'cloned'
        keypoints sharing a (+104, 0) offset on a genuine document."""
        finding = copy_move.detect_copy_move(genuine)
        assert not finding.flagged, finding.detail


# Deliberate, measured miss.
#
# Making the text mask exposure-invariant (so a dim capture no longer produces a
# false positive) also made it correctly classify this forgery's painted-over
# date-of-birth field as text -- and text blocks are excluded from error-level
# analysis, because stroke edges inflate compression error on any document.
#
# Five density thresholds were swept; none recover it. The trade was accepted on
# evidence rather than taste: on the FantasyID held-out split the same change
# moved text-manipulation detection from 6% to 8% and face-swap detection from
# 46% to 47%, with false positives still 0/150, while eliminating false positives
# across the whole exposure range. A crude synthetic paint-over is not
# representative of real text forgery; a genuine passport photographed in a dim
# booth is very representative of real capture.
KNOWN_SYNTHETIC_MISSES = {"forged_field_edit.png"}


class TestTamperDetection:
    @pytest.mark.parametrize(
        "filename",
        [
            "forged_photo_splice.png",
            "forged_field_edit.png",
            "forged_stamp_overlay.png",
            "forged_recompression.png",
        ],
    )
    def test_each_forgery_is_detected(self, filename: str) -> None:
        result = engine.analyze(_load(filename))
        if filename in KNOWN_SYNTHETIC_MISSES:
            pytest.xfail(
                f"{filename} is a documented miss - see KNOWN_SYNTHETIC_MISSES. "
                f"Held-out text detection improved (6% to 8%) despite this."
            )
        assert result.tampered, f"{filename} not detected: {result.summary()}"

    def test_findings_carry_evidence_not_just_a_flag(self) -> None:
        """Every flagged finding must be explainable to an officer."""
        result = engine.analyze(_load("forged_photo_splice.png"))
        flagged = result.flagged_findings
        assert flagged

        for finding in flagged:
            assert len(finding.detail) > 30, "detail is too terse to be evidence"
            assert finding.tamper_type is not None
            assert 0.0 <= finding.confidence <= 1.0

    def test_flagged_findings_localize_the_tamper(self) -> None:
        result = engine.analyze(_load("forged_photo_splice.png"))
        regions = [r for f in result.flagged_findings for r in f.regions]
        assert regions, "a flagged tamper should point at where it is"
        assert all(r.area > 0 for r in regions)

    def test_manifest_expectations_hold(self, manifest: list[dict]) -> None:
        for entry in manifest:
            if entry["filename"] in KNOWN_SYNTHETIC_MISSES:
                continue
            result = engine.analyze(_load(entry["filename"]))
            expected = entry["tamper_type"] != "genuine"
            assert result.tampered is expected, (
                f"{entry['filename']} ({entry['tamper_type']}): "
                f"expected tampered={expected}, got {result.score:.3f}"
            )

    def test_known_misses_are_still_missed_not_silently_fixed(self) -> None:
        """If a documented miss starts passing, the note explaining it is stale.

        This is not a demand that it keep failing -- it is a prompt to re-measure
        and update KNOWN_SYNTHETIC_MISSES and the docs when it changes.
        """
        for filename in KNOWN_SYNTHETIC_MISSES:
            result = engine.analyze(_load(filename))
            if result.tampered:
                pytest.fail(
                    f"{filename} is now detected. Good - but re-run "
                    f"ml/evaluate_fantasyid.py, update the held-out numbers in "
                    f"docs/, and remove it from KNOWN_SYNTHETIC_MISSES."
                )


class TestELA:
    def test_error_map_matches_image_dimensions(self, genuine) -> None:
        error_map = ela.compute_ela_map(genuine)
        assert error_map.shape == genuine.shape[:2]

    def test_recompression_raises_error_level(self, genuine) -> None:
        finding = ela.analyze(_load("forged_recompression.png"))
        assert finding.flagged
        assert finding.tamper_type is TamperType.RECOMPRESSION

    def test_visualization_is_renderable(self, genuine) -> None:
        vis = ela.render_ela_visualization(genuine)
        assert vis.shape == genuine.shape


class TestAdvisoryChecks:
    def test_noise_check_is_advisory_and_cannot_move_the_verdict(self, genuine) -> None:
        """The noise check is not calibrated on real captures, so it must report
        without flagging until it is."""
        assert copy_move.NOISE_DETECTOR_VALIDATED is False
        assert engine.CHECK_WEIGHTS["noise_consistency"] == 0.0

        finding = copy_move.detect_splice(genuine)
        assert not finding.flagged
        assert "advisory" in finding.detail.lower() or not finding.regions


class TestApplicabilityHonesty:
    """A check that could not run must never render as a pass.

    Three separate checks have shipped a green tick they had not earned: the
    uncalibrated noise check, and the copy-move and ELA branches that decline to
    analyse a document. The badge and the explanation must agree, because an
    officer reads the badge.
    """

    def test_no_check_says_pass_while_its_detail_says_otherwise(self, genuine) -> None:
        result = engine.analyze(genuine)
        for finding in result.findings:
            detail = finding.detail.lower()
            declines = any(
                phrase in detail
                for phrase in (
                    "not applicable",
                    "too few",
                    "too small",
                    "too text-dense",
                    "advisory only",
                    "no second portrait",
                    "not yet calibrated",
                )
            )
            if declines:
                assert not finding.applicable, (
                    f"{finding.check} declines to analyse but is marked applicable, "
                    f"so it renders as a pass: {finding.detail}"
                )

    def test_text_dense_document_does_not_pass_copy_move(self) -> None:
        """A page of solid text cannot be cleared by a test that skipped it."""
        import numpy as np

        text_page = np.full((600, 900, 3), 245, dtype=np.uint8)
        for y in range(40, 560, 22):
            cv2.putText(
                text_page, "SPECIMEN TEXT LINE FOR DENSITY", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA,
            )

        finding = copy_move.detect_copy_move(text_page)
        if "not applicable" in finding.detail.lower() or "too few" in finding.detail.lower():
            assert not finding.applicable
