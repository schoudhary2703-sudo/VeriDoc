"""Rules-weighted risk scoring.

Turns the outputs of stages 1-5 into one band, one score, and the evidence that
justifies them. Deliberately rules-based rather than learned: a judge, an
officer, or an auditor can read this file and know exactly why a document was
referred. BUILD_PLAN calls for exactly this ordering -- explainable weights
first, a learned ensemble only once labelled evidence has accumulated.

Two design rules carried over from the forensics engine, both learned by
measurement:

**Signals combine by noisy-OR, not by averaging.** A clean MRZ is not evidence
that the portrait was not substituted. Averaging lets one passing check cancel
another's finding, which measurably suppressed detection when it was tried in
`forensics.engine`. Independent evidence accumulates instead.

**A check that could not run is never scored as a pass.** No live capture means
no face-match evidence, not a face-match success. Those become
`NOT_APPLICABLE`, and they neither raise nor lower the score.

The bands are decision support, not decisions. `high_risk` means "an officer
should look closely", never "reject".
"""

from __future__ import annotations

from app.core.schemas import (
    DBCrosscheckResult,
    EvidenceItem,
    EvidenceStatus,
    FaceMatchResult,
    ForensicsResult,
    MRZCheckResult,
    RiskBand,
    Verdict,
)

# Weight of each signal in the combined risk score. These express relative
# importance and are hand-set, not fitted -- there is no labelled corpus of
# officer decisions to fit them against, and pretending otherwise would be worse
# than stating the judgement openly.
SIGNAL_WEIGHTS: dict[str, float] = {
    # A failed check digit is arithmetic, not inference: the strongest and
    # cheapest signal available, and the hardest for a forger to satisfy.
    "mrz_checksum": 0.95,
    # Watchlist association is decisive when present.
    "watchlist": 0.95,
    # Tamper evidence from the forensics engine, already noisy-OR combined.
    "forensics": 0.70,
    # A face mismatch against a live capture is strong but sensitive to capture
    # quality -- glare and angle cause genuine mismatches, so it is weighted
    # below the arithmetic checks.
    "face_mismatch": 0.60,
    # An expired document is a compliance matter, not fraud.
    "expired_document": 0.30,
    # Not being found in the record set is mildly suspicious but the mock record
    # set is tiny, so this must not drive a verdict on its own.
    "record_not_found": 0.15,
}

# Band thresholds on the combined 0-1 score.
REVIEW_THRESHOLD = 0.20
HIGH_RISK_THRESHOLD = 0.55


def _noisy_or(contributions: list[float]) -> float:
    """Accumulate independent evidence without letting silence cancel it."""
    survival = 1.0
    for value in contributions:
        survival *= 1.0 - min(max(value, 0.0), 1.0)
    return float(min(1.0 - survival, 1.0))


def _mrz_evidence(mrz: MRZCheckResult) -> tuple[EvidenceItem, float]:
    if not mrz.present:
        return (
            EvidenceItem(
                stage="ocr_mrz",
                check="mrz_checksum",
                status=EvidenceStatus.NOT_APPLICABLE,
                detail=(
                    "No machine-readable zone was located on this document. "
                    "Checksum validation could not be performed."
                ),
            ),
            0.0,
        )

    if mrz.valid:
        return (
            EvidenceItem(
                stage="ocr_mrz",
                check="mrz_checksum",
                status=EvidenceStatus.PASS,
                confidence=0.97,
                detail=(
                    f"All {len(mrz.checks)} {mrz.mrz_format.value if mrz.mrz_format else ''} "
                    f"check digits recompute correctly against the printed MRZ."
                ),
            ),
            0.0,
        )

    failed = ", ".join(c.field.replace("_", " ") for c in mrz.failed_checks)
    details = "; ".join(c.detail for c in mrz.failed_checks[:3])
    return (
        EvidenceItem(
            stage="ocr_mrz",
            check="mrz_checksum",
            status=EvidenceStatus.FAIL,
            confidence=0.97,
            detail=(
                f"Check digit failure in: {failed}. {details}. A check digit is "
                f"arithmetic on the field it protects, so a mismatch means the field "
                f"was altered after the document was issued."
            ),
        ),
        SIGNAL_WEIGHTS["mrz_checksum"],
    )


def _forensics_evidence(forensics: ForensicsResult) -> tuple[list[EvidenceItem], float]:
    items: list[EvidenceItem] = []
    for finding in forensics.findings:
        if not finding.applicable:
            status = EvidenceStatus.NOT_APPLICABLE
        elif finding.flagged:
            status = EvidenceStatus.FAIL if finding.confidence >= 0.5 else EvidenceStatus.WEAK
        else:
            status = EvidenceStatus.PASS

        items.append(
            EvidenceItem(
                stage="forensics",
                check=finding.check,
                status=status,
                confidence=round(finding.confidence, 3),
                detail=finding.detail,
                regions=finding.regions,
            )
        )

    contribution = SIGNAL_WEIGHTS["forensics"] * forensics.score
    return items, contribution


def _face_evidence(face: FaceMatchResult) -> tuple[EvidenceItem, float]:
    if not face.performed:
        return (
            EvidenceItem(
                stage="face",
                check="face_match",
                status=EvidenceStatus.NOT_APPLICABLE,
                detail=face.detail or "No live capture supplied; face match not performed.",
            ),
            0.0,
        )

    score = face.match_score or 0.0
    threshold = face.threshold or 0.0

    if face.matched:
        return (
            EvidenceItem(
                stage="face",
                check="face_match",
                status=EvidenceStatus.PASS,
                confidence=round(min(score, 1.0), 3),
                detail=face.detail,
            ),
            0.0,
        )

    # How far below threshold, normalised. A near-miss is weak evidence; a large
    # gap is strong. Capture quality causes near-misses on genuine travellers.
    shortfall = (threshold - score) / max(threshold, 1e-6)
    status = EvidenceStatus.FAIL if shortfall > 0.35 else EvidenceStatus.WEAK
    return (
        EvidenceItem(
            stage="face",
            check="face_match",
            status=status,
            confidence=round(min(max(shortfall, 0.0), 1.0), 3),
            detail=(
                f"{face.detail} A shortfall this size can also be caused by glare or "
                f"capture angle, so a recapture under diffuse light may resolve it."
                if status is EvidenceStatus.WEAK
                else face.detail
            ),
        ),
        SIGNAL_WEIGHTS["face_mismatch"] * min(max(shortfall, 0.0), 1.0),
    )


def _db_evidence(db: DBCrosscheckResult) -> tuple[EvidenceItem, float]:
    if not db.performed:
        return (
            EvidenceItem(
                stage="db_crosscheck",
                check="watchlist_and_record_check",
                status=EvidenceStatus.NOT_APPLICABLE,
                detail="Record cross-check was not performed.",
            ),
            0.0,
        )

    if db.blacklisted:
        return (
            EvidenceItem(
                stage="db_crosscheck",
                check="watchlist_and_record_check",
                status=EvidenceStatus.FAIL,
                confidence=0.99,
                detail=f"{db.detail} Source: {db.source}.",
            ),
            SIGNAL_WEIGHTS["watchlist"],
        )

    if not db.found:
        return (
            EvidenceItem(
                stage="db_crosscheck",
                check="watchlist_and_record_check",
                status=EvidenceStatus.WEAK,
                confidence=0.30,
                detail=(
                    f"{db.detail} Source: {db.source}. Absence from a simulated record "
                    f"set is weak evidence and should not drive a referral on its own."
                ),
            ),
            SIGNAL_WEIGHTS["record_not_found"],
        )

    return (
        EvidenceItem(
            stage="db_crosscheck",
            check="watchlist_and_record_check",
            status=EvidenceStatus.PASS,
            confidence=0.90,
            detail=f"{db.detail} Source: {db.source}.",
        ),
        0.0,
    )


def _band_for(score: float) -> RiskBand:
    if score >= HIGH_RISK_THRESHOLD:
        return RiskBand.HIGH_RISK
    if score >= REVIEW_THRESHOLD:
        return RiskBand.REVIEW
    return RiskBand.CLEAR


def _recommendation(band: RiskBand, evidence: list[EvidenceItem]) -> str:
    failed = [e for e in evidence if e.status is EvidenceStatus.FAIL]
    weak = [e for e in evidence if e.status is EvidenceStatus.WEAK]

    def phrase(items: list[EvidenceItem]) -> str:
        return ", ".join(e.check.replace("_", " ") for e in items)

    if band is RiskBand.HIGH_RISK:
        return (
            f"Recommend secondary inspection before the traveller is cleared. "
            f"Failed checks: {phrase(failed)}."
        )
    if band is RiskBand.REVIEW:
        parts = []
        if failed:
            parts.append(f"failed checks: {phrase(failed)}")
        if weak:
            parts.append(f"borderline checks: {phrase(weak)}")
        return (
            "Officer review recommended before clearing — "
            + "; ".join(parts)
            + ". The system does not accept or reject on its own."
        )
    return (
        "No tamper indicators found. Every applicable check passed; the officer "
        "retains the final decision."
    )


def score_verification(
    *,
    mrz: MRZCheckResult,
    forensics: ForensicsResult,
    face: FaceMatchResult,
    db: DBCrosscheckResult,
    document_expired: bool = False,
) -> Verdict:
    """Combine every stage's output into one explainable verdict."""
    evidence: list[EvidenceItem] = []
    contributions: list[float] = []

    mrz_item, mrz_contribution = _mrz_evidence(mrz)
    evidence.append(mrz_item)
    contributions.append(mrz_contribution)

    forensic_items, forensic_contribution = _forensics_evidence(forensics)
    evidence.extend(forensic_items)
    contributions.append(forensic_contribution)

    face_item, face_contribution = _face_evidence(face)
    evidence.append(face_item)
    contributions.append(face_contribution)

    db_item, db_contribution = _db_evidence(db)
    evidence.append(db_item)
    contributions.append(db_contribution)

    if document_expired:
        evidence.append(
            EvidenceItem(
                stage="ocr_mrz",
                check="document_validity",
                status=EvidenceStatus.FAIL,
                confidence=0.99,
                detail="The document's expiry date has passed.",
            )
        )
        contributions.append(SIGNAL_WEIGHTS["expired_document"])

    score = _noisy_or(contributions)
    band = _band_for(score)

    return Verdict(
        band=band,
        score=round(score, 3),
        recommendation=_recommendation(band, evidence),
        evidence=evidence,
    )
