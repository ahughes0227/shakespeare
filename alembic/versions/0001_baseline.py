"""Baseline audit schema.

Creates every table from `shakespeare.audit.schema` plus the append-only triggers that
make it an audit log rather than a working table.

Note there is no meaningful downgrade: dropping an append-only ledger would destroy the
provenance record it exists to keep.
"""

from __future__ import annotations

from alembic import op

from shakespeare.audit import schema

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    schema.metadata.create_all(bind)
    schema.install_append_only_triggers(bind)


def downgrade() -> None:
    raise RuntimeError(
        "the audit log is append-only and permanent; there is no supported downgrade"
    )
