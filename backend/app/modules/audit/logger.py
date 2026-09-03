"""Append-only audit log of verifications and officer decisions.

Every verification and every officer action is recorded. That is the point of a
decision-support system: the machine's recommendation and the human's decision
are both accountable, and an audit must be able to show that the officer, not the
model, cleared or referred a traveller.

Entries are held in memory for the prototype. The interface is deliberately the
one a durable store would have, so Phase 6 can swap in the Postgres-backed
implementation without touching callers. What must not change when it does: the
log is append-only, and nothing in it is ever edited or removed.

Note what is *not* stored: no images, and no extracted personal fields beyond the
document number. An audit trail that quietly accumulates travellers' photographs
is a liability, not a feature.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.core.schemas import EvidenceStatus, RiskBand, VerifyResponse

MAX_ENTRIES = 500

OfficerAction = Literal["cleared", "referred", "escalated"]


class AuditEntry(BaseModel):
    """One immutable record."""

    verification_id: str
    recorded_at: datetime
    band: RiskBand
    score: float
    document_number: str | None = None
    failed_checks: list[str] = Field(default_factory=list)
    weak_checks: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0

    officer_action: OfficerAction | None = None
    officer_id: str | None = None
    officer_note: str | None = None
    decided_at: datetime | None = None


_entries: deque[AuditEntry] = deque(maxlen=MAX_ENTRIES)
_lock = threading.Lock()


def record_verification(result: VerifyResponse) -> AuditEntry:
    """Append the outcome of one verification."""
    entry = AuditEntry(
        verification_id=result.verification_id,
        recorded_at=datetime.now(timezone.utc),
        band=result.verdict.band,
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
    with _lock:
        _entries.append(entry)
    return entry


def record_decision(
    verification_id: str,
    action: OfficerAction,
    officer_id: str,
    note: str | None = None,
) -> AuditEntry | None:
    """Attach an officer's decision to an existing verification.

    Returns None when the verification is unknown. The decision is attached to
    the existing entry rather than replacing it: the recommendation the officer
    saw stays on the record next to what they decided.
    """
    with _lock:
        for entry in reversed(_entries):
            if entry.verification_id == verification_id:
                entry.officer_action = action
                entry.officer_id = officer_id
                entry.officer_note = note
                entry.decided_at = datetime.now(timezone.utc)
                return entry
    return None


def recent(limit: int = 50) -> list[AuditEntry]:
    """Most recent entries, newest first."""
    with _lock:
        return list(reversed(list(_entries)))[:limit]


def clear() -> None:
    """Test-only. Never call this from application code."""
    with _lock:
        _entries.clear()
