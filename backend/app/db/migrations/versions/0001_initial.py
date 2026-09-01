"""Initial empty baseline.

Phase 0 has no tables yet -- this revision exists so the migration chain has a
root and `alembic upgrade head` is a valid no-op. Models arrive in Phase 4.
"""

from collections.abc import Sequence

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
