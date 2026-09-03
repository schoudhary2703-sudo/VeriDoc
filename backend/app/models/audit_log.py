"""ORM models for the audit trail.

Two tables, and the split is the point.

`verifications` records what the system produced. `officer_decisions` records
what a human then decided. A decision is **inserted**, never written back over
the verification row, so nothing in this schema is ever updated after it is
written. An audit trail whose rows get edited is not an audit trail -- the
question it exists to answer is "what did the system recommend, and what did the
officer do about it", and that needs both facts preserved independently.

It also allows more than one decision against a verification (an escalation
following a referral, say), with the ordering intact.

What is deliberately NOT stored: no images, and no personal fields beyond the
document number. A trail that quietly accumulates travellers' photographs is a
liability, not a feature.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Verification(Base):
    """One run of the pipeline. Written once, never modified."""

    __tablename__ = "verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    verification_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    band: Mapped[str] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)
    document_number: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Check names only. The full evidence text is regenerable from the pipeline
    # and would bloat every row.
    failed_checks: Mapped[list] = mapped_column(JSON, default=list)
    weak_checks: Mapped[list] = mapped_column(JSON, default=list)

    processing_time_ms: Mapped[int] = mapped_column(Integer, default=0)

    decisions: Mapped[list["OfficerDecision"]] = relationship(
        back_populates="verification",
        order_by="OfficerDecision.decided_at",
        lazy="selectin",
    )

    __table_args__ = (Index("ix_verifications_recorded_at_desc", recorded_at.desc()),)


class OfficerDecision(Base):
    """A human's decision on a verification. Appended, never overwritten."""

    __tablename__ = "officer_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    verification_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("verifications.verification_id"), index=True
    )
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    action: Mapped[str] = mapped_column(String(16))
    officer_id: Mapped[str] = mapped_column(String(64), index=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    verification: Mapped[Verification] = relationship(back_populates="decisions")
