"""Standalone Phase 1 pipeline: image path in, fields + MRZ validity out.

Deliberately not wired into the API. BUILD_PLAN Phase 1 asks for a callable
pipeline function that can be exercised on its own; `core/pipeline.py` composes
this with forensics, face match, and the DB check in Phase 4.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.core.schemas import ExtractedFields, MRZCheckResult, OCRMRZResult
from app.modules.ocr_mrz.mrz_parser import parse_mrz
from app.modules.ocr_mrz.ocr_engine import OCREngine, OCREngineError, get_engine
from app.modules.preprocessing.normalize import load_image, preprocess


def run_ocr_mrz(
    image: np.ndarray,
    *,
    engine: OCREngine | None = None,
    skip_preprocessing: bool = False,
) -> OCRMRZResult:
    """Extract fields from a document image and validate its MRZ."""
    started = time.perf_counter()

    if skip_preprocessing:
        ocr_view = image
    else:
        ocr_view = preprocess(image).for_ocr

    if engine is None:
        engine = get_engine()

    ocr_result = engine.extract_text(ocr_view)
    fields, mrz_check = parse_mrz(ocr_result.full_text)

    return OCRMRZResult(
        extracted_fields=fields,
        mrz_check=mrz_check,
        ocr=ocr_result,
        processing_time_ms=int((time.perf_counter() - started) * 1000),
    )


def run_ocr_mrz_on_path(path: str | Path, **kwargs) -> OCRMRZResult:
    """Convenience wrapper: read an image from disk and run the pipeline."""
    return run_ocr_mrz(load_image(path), **kwargs)


def run_mrz_only(mrz_text: str) -> tuple[ExtractedFields, MRZCheckResult]:
    """Parse and validate MRZ text directly, with no OCR involved.

    Useful for testing the checksum logic against known-good and deliberately
    corrupted MRZ strings without depending on an OCR backend being installed.
    """
    return parse_mrz(mrz_text)


__all__ = [
    "OCREngineError",
    "run_mrz_only",
    "run_ocr_mrz",
    "run_ocr_mrz_on_path",
]
