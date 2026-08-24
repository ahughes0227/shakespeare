"""Add plans.fingerprint.

The idempotency receipt matches on the identity of a plan's decisions rather than on its
digest, because a digest covers run_id and two runs of the same request never share one.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_plan_fingerprint"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Append-only triggers reject UPDATE, so existing rows keep the empty default rather
    # than being backfilled. They simply never match a receipt, which is correct: they
    # were recorded before fingerprints existed.
    op.add_column(
        "plans", sa.Column("fingerprint", sa.String(), nullable=False, server_default="")
    )


def downgrade() -> None:
    raise RuntimeError(
        "the audit log is append-only and permanent; there is no supported downgrade"
    )
