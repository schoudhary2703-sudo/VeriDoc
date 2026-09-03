"""POST /api/verify — the endpoint the officer dashboard calls.

Input validation is deliberately strict and stated in the error message. A
border-checkpoint service that accepts a 400 MB upload or a file whose bytes are
not an image is a denial-of-service surface, and "unsupported file" tells an
officer nothing about what to do next.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.pipeline import verify_document
from app.core.schemas import VerifyResponse
from app.db.session import get_db
from app.modules.audit.logger import record_verification

router = APIRouter(tags=["verification"])
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB: a 600 dpi document scan fits easily
ACCEPTED_MIME_PREFIXES = ("image/",)


async def _decode_upload(upload: UploadFile, label: str) -> np.ndarray:
    """Read an upload into a BGR array, or raise a 400 explaining why not."""
    if upload.content_type and not upload.content_type.startswith(ACCEPTED_MIME_PREFIXES):
        raise HTTPException(
            status_code=400,
            detail=(
                f"The {label} must be an image. Received content type "
                f"{upload.content_type!r}."
            ),
        )

    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail=f"The {label} upload was empty.")

    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"The {label} is {len(payload) / 1_048_576:.1f} MB, over the "
                f"{MAX_UPLOAD_BYTES // 1_048_576} MB limit."
            ),
        )

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The {label} could not be decoded as an image. Supported formats "
                f"are JPEG, PNG, BMP, TIFF and WebP."
            ),
        )
    return image


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    document_image: UploadFile = File(..., description="Scan or photo of the document"),
    live_face_image: UploadFile | None = File(
        None, description="Optional live capture; enables the face-match stage"
    ),
    db: AsyncSession = Depends(get_db),
    fast: bool = Query(
        False,
        description=(
            "Restrict OCR to the machine-readable zone. Roughly ten times faster "
            "and still validates every check digit, but does not read the printed "
            "fields."
        ),
    ),
) -> VerifyResponse:
    """Verify one document, optionally against a live face capture."""
    document = await _decode_upload(document_image, "document image")
    capture = (
        await _decode_upload(live_face_image, "live face image")
        if live_face_image is not None
        else None
    )

    try:
        result = verify_document(document, capture, fast_ocr=fast)
    except Exception:  # noqa: BLE001 - surface a clean error, log the detail
        logger.exception("verification pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Verification failed while processing this document.",
        ) from None

    await record_verification(db, result)
    return result
