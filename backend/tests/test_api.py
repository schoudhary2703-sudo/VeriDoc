"""Phase 0 + Phase 4 API tests.

The Phase 4 Definition of Done is asserted directly: a genuine sample document
returns a clear verdict, and a tampered one returns a referral carrying the
specific evidence that caused it. Asserting only the band would let a verdict
that is right for the wrong reason pass, so the evidence is checked too.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.main import app
from app.modules.audit import logger as audit_logger

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# The pipeline takes several seconds per document, so each sample is verified
# once and the response reused. Function-scoped fixtures re-ran it for every
# assertion, turning a twelve-test file into a twelve-verification file.
_RESPONSE_CACHE: dict[str, dict] = {}


async def _verify_once(client: httpx.AsyncClient, filename: str) -> dict:
    if filename not in _RESPONSE_CACHE:
        resp = await client.post(
            "/api/verify",
            files={"document_image": (filename, _sample(filename), "image/png")},
            params={"fast": "true"},
        )
        assert resp.status_code == 200, resp.text
        _RESPONSE_CACHE[filename] = resp.json()
    return _RESPONSE_CACHE[filename]


def _sample(name: str) -> bytes:
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(
            "Specimen images not generated. Run: "
            "python -m ml.data_prep.generate_specimen_documents"
        )
    return path.read_bytes()


class TestHealth:
    async def test_root(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert resp.json()["service"] == "VeriDoc"

    async def test_health_shape(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert set(body["dependencies"]) == {"database", "redis"}


class TestVerifyValidation:
    async def test_rejects_non_image(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/verify",
            files={"document_image": ("notes.txt", b"this is not an image", "text/plain")},
        )
        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()

    async def test_rejects_undecodable_image(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/verify",
            files={"document_image": ("broken.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "image/png")},
        )
        assert resp.status_code == 400
        assert "decode" in resp.json()["detail"].lower()

    async def test_rejects_empty_upload(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/verify",
            files={"document_image": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    async def test_document_image_is_required(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/api/verify")
        assert resp.status_code == 422


class TestVerifyContract:
    """The Phase 4 Definition of Done."""

    @pytest.fixture
    async def genuine(self, client: httpx.AsyncClient):
        return await _verify_once(client, "specimen_passport_genuine.png")

    @pytest.fixture
    async def tampered(self, client: httpx.AsyncClient):
        return await _verify_once(client, "specimen_passport_tampered_dob.png")

    async def test_response_matches_the_contract(self, genuine) -> None:
        for key in (
            "verification_id", "verdict", "extracted_fields", "mrz_check",
            "forensics", "face_match", "db_crosscheck", "processing_time_ms",
        ):
            assert key in genuine, f"missing {key} from the response"

        verdict = genuine["verdict"]
        assert set(verdict) >= {"band", "score", "evidence", "recommendation"}
        assert verdict["band"] in {"clear", "review", "high_risk"}

    async def test_genuine_document_is_clear(self, genuine) -> None:
        assert genuine["mrz_check"]["valid"] is True
        assert genuine["verdict"]["band"] == "clear", genuine["verdict"]["recommendation"]

    async def test_tampered_document_is_referred(self, tampered) -> None:
        assert tampered["mrz_check"]["valid"] is False
        assert tampered["verdict"]["band"] in {"review", "high_risk"}

    async def test_tampered_evidence_names_the_failed_check(self, tampered) -> None:
        """A band alone is not decision support; the reason must be present."""
        failures = [
            e for e in tampered["verdict"]["evidence"] if e["status"] == "fail"
        ]
        assert failures, "a referred document must carry at least one failed check"

        checks = {e["check"] for e in failures}
        assert "mrz_checksum" in checks

        mrz_failure = next(e for e in failures if e["check"] == "mrz_checksum")
        assert "date of birth" in mrz_failure["detail"].lower()

    async def test_every_evidence_item_explains_itself(self, genuine) -> None:
        for item in genuine["verdict"]["evidence"]:
            assert item["status"] in {"pass", "weak", "fail", "not_applicable"}
            assert len(item["detail"]) > 20, f"{item['check']} detail is too terse"

    async def test_absent_face_match_is_not_a_pass(self, genuine) -> None:
        """No live capture means no evidence, never a successful match."""
        face = next(
            e for e in genuine["verdict"]["evidence"] if e["check"] == "face_match"
        )
        assert face["status"] == "not_applicable"
        assert genuine["face_match"]["performed"] is False

    async def test_db_crosscheck_declares_it_is_simulated(self, genuine) -> None:
        """The UI must never imply a live watchlist connection."""
        assert "simulated" in genuine["db_crosscheck"]["source"].lower()


class TestAuditLog:
    async def test_verification_is_recorded(self, client: httpx.AsyncClient) -> None:
        audit_logger.clear()
        resp = await client.post(
            "/api/verify",
            files={
                "document_image": (
                    "genuine.png", _sample("specimen_passport_genuine.png"), "image/png"
                )
            },
            params={"fast": "true"},
        )
        verification_id = resp.json()["verification_id"]

        log = await client.get("/api/audit-log")
        assert log.status_code == 200
        assert any(e["verification_id"] == verification_id for e in log.json())

    async def test_officer_decision_is_attached(self, client: httpx.AsyncClient) -> None:
        audit_logger.clear()
        resp = await client.post(
            "/api/verify",
            files={
                "document_image": (
                    "genuine.png", _sample("specimen_passport_genuine.png"), "image/png"
                )
            },
            params={"fast": "true"},
        )
        verification_id = resp.json()["verification_id"]

        decision = await client.post(
            f"/api/audit-log/{verification_id}/decision",
            json={"action": "cleared", "officer_id": "BSF-2291", "note": "Laminate intact."},
        )
        assert decision.status_code == 200
        body = decision.json()
        assert body["officer_action"] == "cleared"
        assert body["officer_id"] == "BSF-2291"
        assert body["decided_at"] is not None

    async def test_unknown_verification_returns_404(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/audit-log/does-not-exist/decision",
            json={"action": "cleared", "officer_id": "BSF-2291"},
        )
        assert resp.status_code == 404

    async def test_audit_log_stores_no_images(self, client: httpx.AsyncClient) -> None:
        """An audit trail that accumulates travellers' photographs is a liability."""
        audit_logger.clear()
        await client.post(
            "/api/verify",
            files={
                "document_image": (
                    "genuine.png", _sample("specimen_passport_genuine.png"), "image/png"
                )
            },
            params={"fast": "true"},
        )
        entry = (await client.get("/api/audit-log")).json()[0]
        assert not any(
            key in entry for key in ("image", "document_image", "photo", "face")
        )
