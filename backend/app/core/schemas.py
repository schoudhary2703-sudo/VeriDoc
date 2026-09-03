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


class TamperType(str, Enum):
    """The tamper classes the forensics engine reports on.

    Accuracy is reported per type, never blended into one number -- a detector
    that catches photo splices but misses date edits is a different (and more
    dangerous) tool than one that is uniformly mediocre.
    """

    PHOTO_SPLICE = "photo_splice"
    FIELD_EDIT = "field_edit"        # altered DOB / name / document number
    STAMP_OVERLAY = "stamp_overlay"
    RECOMPRESSION = "recompression"
    COPY_MOVE = "copy_move"
    # Reported when a detector establishes that a document was digitally
    # manipulated but cannot attribute which kind. The learned classifier is
    # binary (see ml/data_prep/fantasyid_dataset.py for why), so this is the
    # honest label for its positives rather than guessing a specific type.
    DIGITAL_MANIPULATION = "digital_manipulation"


class Region(BaseModel):
    """A coarse region of interest, in pixels.

    Deliberately coarse. Tampered areas on ID documents occupy 0.27-4.17% of the
    image and state-of-the-art detectors score near-zero on pixel-level
    localization (DocForge-Bench 2026), so this is a bounding box for an officer
    to look at -- never a claim of exact-pixel segmentation. See
    docs/DATA_STRATEGY.md section 2.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    score: float = Field(ge=0.0, le=1.0, default=0.0)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class ForensicsFinding(BaseModel):
    """One forensic observation, with the reasoning attached.

    `detail` is written to be read aloud to a human. A finding that cannot be
    explained in a sentence does not belong in the evidence panel.
    """

    check: str
    tamper_type: TamperType | None = None
    flagged: bool
    confidence: float = Field(ge=0.0, le=1.0)
    detail: str
    regions: list[Region] = Field(default_factory=list)

    # False when the check could not run on this document at all -- a single-
    # portrait card for the face cross-check, an image too small for block
    # analysis. A test that could not run is not evidence of authenticity, and
    # must not dilute the checks that did run, so the score normalises over
    # applicable checks only.
    applicable: bool = True


class ForensicsResult(BaseModel):
    """Combined output of the forensics engine (stage 3)."""

    tampered: bool = False
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    findings: list[ForensicsFinding] = Field(default_factory=list)
    processing_time_ms: int = 0

    @property
    def flagged_findings(self) -> list[ForensicsFinding]:
        return [f for f in self.findings if f.flagged]

    def summary(self) -> str:
        flagged = self.flagged_findings
        if not flagged:
            return f"No tamper indicators found across {len(self.findings)} forensic checks"
        names = ", ".join(
            f.tamper_type.value if f.tamper_type else f.check for f in flagged
        )
        return f"Tamper indicators: {names}"


class FaceMatchResult(BaseModel):
    """Stage 4 output: document photo compared against a live capture."""

    performed: bool = False
    match_score: float | None = None
    matched: bool | None = None
    threshold: float | None = None
    faces_in_document: int = 0
    faces_in_capture: int = 0
    liveness_passed: bool | None = None
    detail: str = ""


class LivenessCue(BaseModel):
    """One passive liveness measurement.

    `suspicious` is None while the check is uncalibrated: an unfitted threshold
    cannot say whether a value is normal, and reporting False would read as a
    pass the module has not earned.
    """

    name: str
    value: float
    threshold: float
    direction: str          # "above" or "below" the threshold is suspicious
    suspicious: bool | None = None
    detail: str = ""


class LivenessResult(BaseModel):
    """Stage 4b output: passive presentation-attack detection."""

    performed: bool = False
    calibrated: bool = False
    # None means "not established", never "passed".
    passed: bool | None = None
    cues: list[LivenessCue] = Field(default_factory=list)
    detail: str = ""


class OCRMRZResult(BaseModel):
    """Combined output of the Phase 1 pipeline."""

    extracted_fields: ExtractedFields = Field(default_factory=ExtractedFields)
    mrz_check: MRZCheckResult = Field(default_factory=MRZCheckResult)
    ocr: OCRResult | None = None
    processing_time_ms: int = 0


# ---------------------------------------------------------------------------
# Verification response contract (BUILD_PLAN Section 5)
#
# Extended beyond the original spec in one respect: evidence carries a
# four-state status rather than a boolean `passed`. The officer dashboard needs
# to distinguish "this check failed" from "this check is borderline" from "this
# check could not run on this document", and collapsing those three into
# passed=false would either overstate a weak signal or hide a missing one.
# ---------------------------------------------------------------------------


class EvidenceStatus(str, Enum):
    PASS = "pass"
    WEAK = "weak"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class RiskBand(str, Enum):
    CLEAR = "clear"
    REVIEW = "review"
    HIGH_RISK = "high_risk"


class EvidenceItem(BaseModel):
    """One line in the officer's evidence panel."""

    stage: str          # preprocessing | ocr_mrz | forensics | face | db_crosscheck
    check: str
    status: EvidenceStatus
    detail: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    regions: list[Region] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Back-compatibility with the original boolean contract."""
        return self.status is EvidenceStatus.PASS


class DBCrosscheckResult(BaseModel):
    """Stage 5 output.

    `source` is deliberately explicit. This prototype queries a seeded local
    table, never a live watchlist, and the UI must say so rather than implying a
    connection to Interpol SLTD or a national lookout system.
    """

    performed: bool = False
    found: bool = False
    blacklisted: bool = False
    status: str | None = None
    source: str = "simulated local record set (no live watchlist access)"
    detail: str = ""


class Verdict(BaseModel):
    band: RiskBand
    score: float = Field(ge=0.0, le=1.0)
    recommendation: str
    evidence: list[EvidenceItem] = Field(default_factory=list)

    @property
    def failed(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.status is EvidenceStatus.FAIL]

    @property
    def weak(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.status is EvidenceStatus.WEAK]


class VerifyResponse(BaseModel):
    """The complete `POST /api/verify` response."""

    verification_id: str
    verdict: Verdict
    extracted_fields: ExtractedFields = Field(default_factory=ExtractedFields)
    mrz_check: MRZCheckResult = Field(default_factory=MRZCheckResult)
    forensics: ForensicsResult = Field(default_factory=ForensicsResult)
    face_match: FaceMatchResult = Field(default_factory=FaceMatchResult)
    db_crosscheck: DBCrosscheckResult = Field(default_factory=DBCrosscheckResult)
    processing_time_ms: int = 0
    stage_timings_ms: dict[str, int] = Field(default_factory=dict)
