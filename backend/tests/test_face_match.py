"""Phase 3 tests: face matching and intra-document face consistency.

Tests requiring InsightFace skip when it is not installed, so the suite stays
runnable on a machine without the 281 MB model weights.

The substantive test here is `test_genuine_documents_are_not_flagged`. The
intra-document check exists to catch face swaps, and its default threshold was
chosen at the zero-false-positive operating point precisely because a border
system must not accuse genuine travellers. If that property ever breaks, the
threshold is wrong regardless of what it does to recall.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pytest

FANTASYID = Path(__file__).resolve().parents[2] / "data" / "raw" / "FantasyID"


def _require_insightface():
    from app.modules.face import face_match

    if not face_match.is_available():
        pytest.skip("insightface/onnxruntime not installed")
    return face_match


def _sample(kind: str, n: int, seed: int = 21) -> list[Path]:
    """A reproducible sample of FantasyID test-split images."""
    csv_path = FANTASYID / "test.csv"
    if not csv_path.exists():
        pytest.skip("FantasyID not downloaded")

    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if kind == "genuine":
        pool = [r for r in rows if r["is_attack"] == "False"]
    else:
        pool = [r for r in rows if r.get("attack_type") == kind]

    random.Random(seed).shuffle(pool)
    paths = [FANTASYID / r["path"] for r in pool]
    return [p for p in paths if p.exists()][:n]


class TestCosineSimilarity:
    """Pure maths, no model needed."""

    def test_identical_vectors_score_one(self) -> None:
        from app.modules.face.face_match import cosine_similarity

        v = np.array([0.3, -0.7, 1.2, 0.5])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        from app.modules.face.face_match import cosine_similarity

        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_does_not_divide_by_zero(self) -> None:
        from app.modules.face.face_match import cosine_similarity

        assert cosine_similarity(np.zeros(4), np.ones(4)) == 0.0


class TestIntraDocumentConsistency:
    @pytest.fixture(scope="module")
    def face_match(self):
        return _require_insightface()

    def test_genuine_documents_are_not_flagged(self, face_match) -> None:
        """The default threshold is the zero-false-positive operating point."""
        import cv2

        paths = _sample("genuine", 8)
        if not paths:
            pytest.skip("no genuine samples available")

        flagged = []
        for path in paths:
            finding = face_match.check_intra_document_consistency(cv2.imread(str(path)))
            if finding.flagged:
                flagged.append((path.name, finding.detail))

        assert not flagged, f"false positives on genuine documents: {flagged}"

    def test_single_portrait_document_is_not_applicable(self, face_match) -> None:
        """A test that could not run is not evidence of authenticity."""
        import numpy as np

        blank = np.full((600, 900, 3), 240, dtype=np.uint8)
        finding = face_match.check_intra_document_consistency(blank)

        assert not finding.flagged
        assert finding.tamper_type is None
        assert "second portrait" in finding.detail

        # The assertion this test was named for, and originally missing: without
        # it the finding rendered as a green PASS in the evidence panel while its
        # own detail text said the check had nothing to compare.
        assert finding.applicable is False

    def test_a_check_that_did_not_run_is_not_shown_as_a_pass(self, face_match) -> None:
        """Guard the mapping, not just the flag -- the defect lived downstream."""
        import numpy as np

        from app.core.risk_scoring import _forensics_evidence
        from app.core.schemas import EvidenceStatus, ForensicsResult

        blank = np.full((600, 900, 3), 240, dtype=np.uint8)
        finding = face_match.check_intra_document_consistency(blank)

        items, _ = _forensics_evidence(
            ForensicsResult(findings=[finding], score=0.0, tampered=False)
        )
        assert items[0].status is EvidenceStatus.NOT_APPLICABLE

    def test_finding_reports_both_face_regions(self, face_match) -> None:
        import cv2

        for path in _sample("genuine", 6):
            finding = face_match.check_intra_document_consistency(cv2.imread(str(path)))
            if "second portrait" in finding.detail:
                continue
            assert len(finding.regions) == 2
            assert all(r.area > 0 for r in finding.regions)
            return
        pytest.skip("no two-portrait document found in the sample")


class TestEngineIntegration:
    def test_face_check_is_weighted_and_counted(self) -> None:
        from app.modules.forensics import engine

        assert engine.CHECK_WEIGHTS["intra_document_face_consistency"] > 0.0

    def test_inapplicable_checks_do_not_dilute_the_score(self) -> None:
        """Regression: adding a fourth check must not suppress the other three.

        Before `applicable` existed, a check that could not run still counted in
        the score denominator, which pushed a genuine detection below threshold.
        """
        from app.core.schemas import ForensicsFinding
        from app.modules.forensics.engine import _combine

        strong = ForensicsFinding(
            check="error_level_analysis", flagged=True, confidence=1.0, detail="x"
        )
        not_applicable = ForensicsFinding(
            check="intra_document_face_consistency",
            flagged=False,
            confidence=0.0,
            detail="no second portrait",
            applicable=False,
        )

        assert _combine([strong]) == pytest.approx(_combine([strong, not_applicable]))


class TestTightCropDetection:
    """Regression: a closely-framed face must still be detected.

    RetinaFace's anchors expect a face to occupy a modest fraction of the frame,
    so a tight head-shot returned zero detections and the face match silently
    reported "not performed". A traveller leaning towards the camera produces
    exactly that framing, so this is a real capture condition rather than a
    laboratory curiosity.
    """

    @pytest.fixture(scope="module")
    def face_match(self):
        return _require_insightface()

    def _tight_crop(self, face_match, image):
        import cv2  # noqa: F401  (imported for parity with callers)

        face = face_match.detect_faces(image)[0]
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        pad = 40
        return image[max(y1 - pad, 0) : y2 + pad, max(x1 - pad, 0) : x2 + pad]

    def test_tightly_cropped_face_is_still_detected(self, face_match) -> None:
        import cv2

        paths = _sample("genuine", 3)
        for path in paths:
            image = cv2.imread(str(path))
            if not face_match.detect_faces(image):
                continue
            crop = self._tight_crop(face_match, image)
            assert face_match.detect_faces(crop), (
                f"tight crop of {path.name} lost the face; the padded retry failed"
            )
            return
        pytest.skip("no sample with a detectable face available")

    def test_same_person_matches_across_a_tight_crop(self, face_match) -> None:
        import cv2

        for path in _sample("genuine", 3):
            image = cv2.imread(str(path))
            if not face_match.detect_faces(image):
                continue
            result = face_match.match_document_to_capture(
                image, self._tight_crop(face_match, image)
            )
            assert result.performed, result.detail
            assert result.matched is True
            assert result.match_score is not None and result.match_score > 0.7
            return
        pytest.skip("no sample with a detectable face available")

    def test_padded_retry_keeps_bboxes_in_the_original_frame(self, face_match) -> None:
        """Coordinates must stay meaningful after the retry, or the evidence
        panel would draw regions outside the image."""
        import cv2

        for path in _sample("genuine", 3):
            image = cv2.imread(str(path))
            if not face_match.detect_faces(image):
                continue
            crop = self._tight_crop(face_match, image)
            faces = face_match.detect_faces(crop)
            assert faces
            height, width = crop.shape[:2]
            x1, y1, x2, y2 = faces[0].bbox
            # Allow a small overhang: the true face may extend past a tight crop.
            assert -width < x1 < width * 1.5
            assert -height < y1 < height * 1.5
            return
        pytest.skip("no sample with a detectable face available")
