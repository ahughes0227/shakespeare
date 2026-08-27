"""Add the measurements ledger.

What a run observed about its own cost and its own confidence, kept so a constant that is
currently declared can be replaced by one that is measured. Observations rather than
aggregates, for the same reason the rest of this log holds facts: an average cannot be
re-derived under a new weighting, and cannot be invalidated when the model changes.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from shakespeare.runtime.audit import schema

revision = "0003_measurements"
down_revision = "0002_plan_fingerprint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "measurements",
        sa.Column("measurement_id", sa.String(), nullable=False, primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.run_id"), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("resolved_model", sa.String(), nullable=False),
        sa.Column("prompt_version", sa.String(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.Boolean(), nullable=False),
        sa.Column("bound", sa.String()),
        sa.Column("recorded_at", sa.String(), nullable=False),
    )
    # Derived from the table list rather than the columns, so a new table gets its
    # triggers without this migration having to name them.
    schema.install_append_only_triggers(op.get_bind())


def downgrade() -> None:
    raise RuntimeError(
        "the audit log is append-only and permanent; there is no supported downgrade"
    )
