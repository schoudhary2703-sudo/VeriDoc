"""Audit trail: verifications and officer decisions.

Two tables rather than one. A decision is inserted alongside the verification it
refers to, never written back over it, so no row in this schema is ever updated
after insert. See app/models/audit_log.py.

Revision ID: 0002_audit_tables
Revises: 0001_initial
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_audit_tables"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "verifications",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("verification_id", sa.String(length=64), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("band", sa.String(length=16), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("document_number", sa.String(length=64), nullable=True),
        sa.Column("failed_checks", sa.JSON(), nullable=False),
        sa.Column("weak_checks", sa.JSON(), nullable=False),
        sa.Column("processing_time_ms", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("verification_id"),
    )
    op.create_index("ix_verifications_recorded_at", "verifications", ["recorded_at"])
    op.create_index("ix_verifications_document_number", "verifications", ["document_number"])

    op.create_table(
        "officer_decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("verification_id", sa.String(length=64), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("officer_id", sa.String(length=64), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["verification_id"], ["verifications.verification_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_officer_decisions_verification_id", "officer_decisions", ["verification_id"]
    )
    op.create_index("ix_officer_decisions_officer_id", "officer_decisions", ["officer_id"])


def downgrade() -> None:
    op.drop_table("officer_decisions")
    op.drop_index("ix_verifications_document_number", table_name="verifications")
    op.drop_index("ix_verifications_recorded_at", table_name="verifications")
    op.drop_table("verifications")
