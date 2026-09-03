"""GET /api/audit-log and the officer-decision endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.modules.audit.logger import AuditEntry, OfficerAction, recent, record_decision

router = APIRouter(tags=["audit"])


class DecisionRequest(BaseModel):
    action: OfficerAction
    officer_id: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=2000)


@router.get("/audit-log", response_model=list[AuditEntry])
async def audit_log(limit: int = Query(50, ge=1, le=200)) -> list[AuditEntry]:
    """Recent verifications and the officer decisions attached to them."""
    return recent(limit)


@router.post("/audit-log/{verification_id}/decision", response_model=AuditEntry)
async def submit_decision(verification_id: str, request: DecisionRequest) -> AuditEntry:
    """Record an officer's decision against a verification.

    The system never decides on its own; this is where the human's call is
    written to the record, alongside the recommendation they were shown.
    """
    entry = record_decision(
        verification_id, request.action, request.officer_id, request.note
    )
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"No verification found with id {verification_id}"
        )
    return entry
