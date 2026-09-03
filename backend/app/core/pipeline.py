"""Pipeline orchestrator: stages 1-6 in sequence.

    1. pre-process        deskew, crop, glare (OCR view and forensics view)
    2. OCR + MRZ          field extraction and ICAO 9303 checksum validation
    3. forensics          ELA, copy-move, noise, intra-document face consistency
    4. face match         document portrait vs live capture (when supplied)
    5. record cross-check simulated issuing-authority lookup
    6. risk scoring       one band, one score, and the evidence behind them

Every stage is wrapped so a failure degrades that stage rather than the request.
A crashed OCR engine should cost the officer one line of evidence, not the whole
verification -- and the risk scorer already treats a missing stage as
`NOT_APPLICABLE` rather than as a pass, so degrading is safe by construction.

Stage 1 returns two images and the distinction matters: glare reduction and
denoising rewrite exactly the pixel statistics the forensics engine measures, so
forensics reads `for_forensics` while OCR reads `for_ocr`.
"""

from __future__ import annotations

import time
import uuid
from datetime import date

import numpy as np

from app.core.risk_scoring import score_verification
from app.core.schemas import (
    DBCrosscheckResult,
    ExtractedFields,
    FaceMatchResult,
    ForensicsResult,
    MRZCheckResult,
    VerifyResponse,
)
from app.modules.db_crosscheck.lookup import cross_check
from app.modules.forensics import engine as forensics_engine
from app.modules.ocr_mrz.mrz_parser import parse_mrz
from app.modules.ocr_mrz.ocr_engine import OCREngineError, get_engine
from app.modules.preprocessing.normalize import preprocess


class _Timer:
    """Records how long each stage took, for the dashboard's timing panel."""

    def __init__(self) -> None:
        self.timings: dict[str, int] = {}

    def measure(self, stage: str):
        timer = self

        class _Scope:
            def __enter__(self):
                self.started = time.perf_counter()
                return self

            def __exit__(self, *exc):
                timer.timings[stage] = int((time.perf_counter() - self.started) * 1000)
                return False

        return _Scope()


def _run_ocr_mrz(image: np.ndarray) -> tuple[ExtractedFields, MRZCheckResult]:
    """OCR the document and validate its MRZ, degrading rather than raising."""
    try:
        ocr = get_engine()
    except OCREngineError as exc:
        return ExtractedFields(), MRZCheckResult(present=False, errors=[str(exc)])

    try:
        result = ocr.extract_text(image)
    except Exception as exc:  # noqa: BLE001 - one bad stage must not fail the request
        return ExtractedFields(), MRZCheckResult(
            present=False, errors=[f"OCR failed: {type(exc).__name__}"]
        )

    return parse_mrz(result.full_text)


def _run_face_match(
    document_image: np.ndarray, capture_image: np.ndarray | None
) -> FaceMatchResult:
    if capture_image is None:
        return FaceMatchResult(
            performed=False,
            detail="No live capture supplied; face match not performed.",
        )

    try:
        from app.modules.face import face_match
    except ImportError:
        return FaceMatchResult(performed=False, detail="Face matching is unavailable.")

    if not face_match.is_available():
        return FaceMatchResult(
            performed=False,
            detail="InsightFace is not installed; face match not performed.",
        )

    try:
        return face_match.match_document_to_capture(document_image, capture_image)
    except Exception as exc:  # noqa: BLE001
        return FaceMatchResult(
            performed=False, detail=f"Face match failed: {type(exc).__name__}"
        )


def verify_document(
    document_image: np.ndarray,
    live_face_image: np.ndarray | None = None,
    *,
    fast_ocr: bool = False,
) -> VerifyResponse:
    """Run the full verification pipeline over one document.

    `fast_ocr` restricts OCR to the machine-readable zone. That is roughly ten
    times faster and still validates every check digit, but it does not read the
    printed fields, so the printed-vs-MRZ comparison is unavailable.
    """
    started = time.perf_counter()
    timer = _Timer()

    with timer.measure("preprocessing"):
        prepared = preprocess(document_image)

    ocr_source = prepared.for_ocr
    if fast_ocr:
        import cv2

        from app.modules.preprocessing.normalize import extract_mrz_band

        band = extract_mrz_band(prepared.for_ocr)
        if band is not None:
            # Upscale the strip before recognition. MRZ glyphs are only ~22 px
            # wide at native scan resolution, and doubling them lifted the
            # recogniser's confidence from 0.88 to 0.99 for well under a second.
            ocr_source = cv2.resize(
                band, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
            )

    with timer.measure("ocr_mrz"):
        fields, mrz = _run_ocr_mrz(ocr_source)

    with timer.measure("forensics"):
        try:
            forensics = forensics_engine.analyze(prepared.for_forensics)
        except Exception:  # noqa: BLE001
            forensics = ForensicsResult()

    with timer.measure("face"):
        face = _run_face_match(prepared.for_forensics, live_face_image)

    with timer.measure("db_crosscheck"):
        try:
            db = cross_check(fields.document_number)
        except Exception as exc:  # noqa: BLE001
            db = DBCrosscheckResult(
                performed=False, detail=f"Record lookup failed: {type(exc).__name__}"
            )

    expired = bool(fields.expiry_date and fields.expiry_date < date.today())

    with timer.measure("risk_scoring"):
        verdict = score_verification(
            mrz=mrz, forensics=forensics, face=face, db=db, document_expired=expired
        )

    return VerifyResponse(
        verification_id=str(uuid.uuid4()),
        verdict=verdict,
        extracted_fields=fields,
        mrz_check=mrz,
        forensics=forensics,
        face_match=face,
        db_crosscheck=db,
        processing_time_ms=int((time.perf_counter() - started) * 1000),
        stage_timings_ms=timer.timings,
    )
