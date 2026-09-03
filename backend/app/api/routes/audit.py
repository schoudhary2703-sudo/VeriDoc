"""GET /api/audit-log and the officer-decision endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.modules.audit.logger import OfficerAction, recent, record_decision

router = APIRouter(tags=["audit"])


class DecisionRequest(BaseModel):
    action: OfficerAction
    officer_id: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=2000)


@router.get("/audit-log")
async def audit_log(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Recent verifications and the officer decisions recorded against them."""
    return [entry.model_dump_api() for entry in await recent(db, limit)]


@router.post("/audit-log/{verification_id}/decision")
async def submit_decision(
    verification_id: str,
    request: DecisionRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Record an officer's decision against a verification.

    The system never decides on its own; this is where the human's call is
    written to the record, alongside the recommendation they were shown. The
    decision is appended -- the verification row itself is never modified.
    """
    entry = await record_decision(
        db, verification_id, request.action, request.officer_id, request.note
    )
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"No verification found with id {verification_id}"
        )
    return entry.model_dump_api()
