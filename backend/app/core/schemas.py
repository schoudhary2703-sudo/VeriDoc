"""Shared Pydantic models.

These are the contract described in Section 5 of BUILD_PLAN.md. Phase 1 fills in
the OCR/MRZ half; the forensics, face, and DB-crosscheck halves arrive in later
phases and slot into the same `VerifyResponse` shape.

Design note: every check carries its own outcome plus a human-readable detail
string, not a bare boolean. `evidence[]` in the final response is assembled from
these, and the officer dashboard renders that array directly.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class MRZFormat(str, Enum):
    """ICAO 9303 machine-readable-zone layouts."""

    TD1 = "TD1"  # ID cards - 3 lines x 30 chars
    TD2 = "TD2"  # Older travel documents - 2 lines x 36 chars
    TD3 = "TD3"  # Passports - 2 lines x 44 chars


class Sex(str, Enum):
    MALE = "M"
    FEMALE = "F"
    UNSPECIFIED = "X"


class OCRField(BaseModel):
    """One text region returned by an OCR engine."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    # (x1, y1, x2, y2) in pixels, axis-aligned.
    bbox: tuple[int, int, int, int] | None = None


class OCRResult(BaseModel):
    """Everything an OCR engine produces for one image."""

    engine: str
    fields: list[OCRField] = Field(default_factory=list)
    processing_time_ms: int = 0

    @property
    def full_text(self) -> str:
        return "\n".join(f.text for f in self.fields)

    @property
    def mean_confidence(self) -> float:
        if not self.fields:
            return 0.0
        return sum(f.confidence for f in self.fields) / len(self.fields)


class CheckDigitResult(BaseModel):
    """Outcome of a single ICAO 9303 check-digit validation.

    Kept per-field rather than collapsed into one boolean: knowing *which* digit
    failed is the difference between "MRZ invalid" and "the date of birth was
    altered", and the second is what an officer can act on.
    """

    field: str
    raw_value: str
    expected: str
    actual: str
    passed: bool

    @property
    def detail(self) -> str:
        if self.passed:
            return f"{self.field} check digit valid"
        return (
            f"{self.field} check digit mismatch: "
            f"expected {self.expected!r}, found {self.actual!r}"
        )


class ExtractedFields(BaseModel):
    """Identity fields recovered from the document."""

    name: str | None = None
    surname: str | None = None
    given_names: str | None = None
    dob: date | None = None
    document_number: str | None = None
    expiry_date: date | None = None
    nationality: str | None = None
    issuing_state: str | None = None
    sex: Sex | None = None
    personal_number: str | None = None


class MRZCheckResult(BaseModel):
    """Result of parsing and validating a machine-readable zone."""

    present: bool = False
    mrz_format: MRZFormat | None = None
    raw_lines: list[str] = Field(default_factory=list)

    valid: bool = False
    checksum_match: bool = False
    checks: list[CheckDigitResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def failed_checks(self) -> list[CheckDigitResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        if not self.present:
            return "No MRZ detected"
        if not self.checks:
            return "MRZ found but no check digits could be validated"
        failed = self.failed_checks
        fmt = self.mrz_format.value if self.mrz_format else "MRZ"
        if not failed:
            return f"{fmt} MRZ valid - all {len(self.checks)} check digits match"
        names = ", ".join(c.field for c in failed)
        return f"{fmt} MRZ check digit failure: {names}"


class OCRMRZResult(BaseModel):
    """Combined output of the Phase 1 pipeline."""

    extracted_fields: ExtractedFields = Field(default_factory=ExtractedFields)
    mrz_check: MRZCheckResult = Field(default_factory=MRZCheckResult)
    ocr: OCRResult | None = None
    processing_time_ms: int = 0
