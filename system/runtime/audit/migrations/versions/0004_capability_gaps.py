"""Record requests nothing registered could serve.

An operator a capability lacks has had a backlog since admission was written: the request
is journalled, risk is computed, and a person sees what was asked for and why. A workflow
nobody has built had nowhere to be recorded at all — the router's analysis of what the
request would take was rendered to a terminal and the process exited.

The analysis is the valuable part. It is already being produced; this keeps it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from system.runtime.audit import schema

revision = "0004_capability_gaps"
down_revision = "0003_measurements"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "capability_gaps",
        sa.Column("gap_id", sa.String(), nullable=False, primary_key=True),
        sa.Column("run_id", sa.String(), nullable=False),
        # The prompt itself is the user's content and is never stored; its digest is what
        # lets the same unmet request be recognised across runs.
        sa.Column("prompt_digest", sa.String(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("requires", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.String(), nullable=False),
    )
    schema.install_append_only_triggers(op.get_bind())


def downgrade() -> None:
    raise RuntimeError(
        "the audit log is append-only and permanent; there is no supported downgrade"
    )
