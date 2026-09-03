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
