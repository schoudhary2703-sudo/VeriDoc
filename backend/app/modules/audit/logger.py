"""Durable, append-only audit log of verifications and officer decisions.

Every verification and every officer action is recorded and survives a restart.
That is the point of a decision-support system: the machine's recommendation and
the human's decision are both accountable, and an audit must be able to show
that the officer -- not the model -- cleared or referred a traveller.

Nothing here updates a row. A decision is inserted into its own table alongside
the verification it refers to; see `app/models/audit_log.py` for why.

Writes are best-effort by design: `record_verification` logs and swallows a
database failure rather than propagating it. An officer who has just verified a
document should still get their result if the audit database is briefly
unreachable, and the alternative -- failing the request -- would push them
towards working around the system entirely. Officer *decisions* do propagate
their errors, because a decision the officer believes was recorded and was not
is far worse than an error they can see and retry.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import EvidenceStatus, RiskBand, VerifyResponse
from app.models.audit_log import OfficerDecision, Verification

logger = logging.getLogger(__name__)

OfficerAction = Literal["cleared", "referred", "escalated"]


class DecisionEntry(BaseModel):
    action: OfficerAction
    officer_id: str
    note: str | None = None
    decided_at: datetime


class AuditEntry(BaseModel):
    """One verification and every decision recorded against it."""

    verification_id: str
    recorded_at: datetime
    band: RiskBand
    score: float
    document_number: str | None = None
    failed_checks: list[str] = Field(default_factory=list)
    weak_checks: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0

    decisions: list[DecisionEntry] = Field(default_factory=list)

    # The most recent decision, flattened for convenience. The dashboard shows
    # this; the full history stays available in `decisions`.
    @property
    def officer_action(self) -> OfficerAction | None:
        return self.decisions[-1].action if self.decisions else None

    @property
    def officer_id(self) -> str | None:
        return self.decisions[-1].officer_id if self.decisions else None

    @property
    def officer_note(self) -> str | None:
        return self.decisions[-1].note if self.decisions else None

    @property
    def decided_at(self) -> datetime | None:
        return self.decisions[-1].decided_at if self.decisions else None

    @classmethod
    def from_row(cls, row: Verification) -> "AuditEntry":
        return cls(
            verification_id=row.verification_id,
            recorded_at=row.recorded_at,
            band=RiskBand(row.band),
            score=row.score,
            document_number=row.document_number,
            failed_checks=list(row.failed_checks or []),
            weak_checks=list(row.weak_checks or []),
            processing_time_ms=row.processing_time_ms,
            decisions=[
                DecisionEntry(
                    action=d.action,
                    officer_id=d.officer_id,
                    note=d.note,
                    decided_at=d.decided_at,
                )
                for d in row.decisions
            ],
        )

    def model_dump_api(self) -> dict:
        """Serialise with the flattened decision fields the dashboard reads."""
        data = self.model_dump()
        data.update(
            officer_action=self.officer_action,
            officer_id=self.officer_id,
            officer_note=self.officer_note,
            decided_at=self.decided_at,
        )
        return data


async def record_verification(session: AsyncSession, result: VerifyResponse) -> None:
    """Append the outcome of one verification. Never raises."""
    row = Verification(
        verification_id=result.verification_id,
        band=result.verdict.band.value,
        score=result.verdict.score,
        document_number=result.extracted_fields.document_number,
        failed_checks=[
            e.check for e in result.verdict.evidence if e.status is EvidenceStatus.FAIL
        ],
        weak_checks=[
            e.check for e in result.verdict.evidence if e.status is EvidenceStatus.WEAK
        ],
        processing_time_ms=result.processing_time_ms,
    )
    try:
        session.add(row)
        await session.commit()
    except Exception:  # noqa: BLE001 - the officer's result must not be lost to a log write
        await session.rollback()
        logger.exception(
            "could not write audit entry for verification %s", result.verification_id
        )


async def record_decision(
    session: AsyncSession,
    verification_id: str,
    action: OfficerAction,
    officer_id: str,
    note: str | None = None,
) -> AuditEntry | None:
    """Append an officer's decision. Returns None if the verification is unknown.

    Errors here propagate: a decision the officer believes was recorded, and was
    not, is worse than a visible failure they can retry.
    """
    exists = await session.scalar(
        select(Verification).where(Verification.verification_id == verification_id)
    )
    if exists is None:
        return None

    session.add(
        OfficerDecision(
            verification_id=verification_id,
            action=action,
            officer_id=officer_id,
            note=note,
        )
    )
    await session.commit()

    # The session factory uses expire_on_commit=False, so the Verification
    # loaded above is still in the identity map with its `decisions` collection
    # as it was *before* this insert. Without expiring, the read below returns
    # the stale object and the caller sees a decision that was written but does
    # not appear.
    session.expire_all()
    return await get(session, verification_id)


async def get(session: AsyncSession, verification_id: str) -> AuditEntry | None:
    row = await session.scalar(
        select(Verification).where(Verification.verification_id == verification_id)
    )
    return AuditEntry.from_row(row) if row else None


async def recent(session: AsyncSession, limit: int = 50) -> list[AuditEntry]:
    """Most recent entries, newest first."""
    rows = (
        await session.scalars(
            select(Verification).order_by(desc(Verification.recorded_at)).limit(limit)
        )
    ).all()
    return [AuditEntry.from_row(r) for r in rows]


async def clear(session: AsyncSession) -> None:
    """Test-only. Never call this from application code."""
    await session.execute(delete(OfficerDecision))
    await session.execute(delete(Verification))
    await session.commit()
