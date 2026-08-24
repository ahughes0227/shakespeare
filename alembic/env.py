"""Alembic environment.

The audit log is append-only and permanent, so its schema is migrated rather than
recreated.  `target_metadata` points at the single source of truth in
`shakespeare.audit.schema`, which lets `alembic check` detect drift between the models
and the migrations.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from shakespeare.audit import schema

config = context.config
target_metadata = schema.metadata


def _url() -> str:
    return os.environ.get("SHAKESPEARE_AUDIT_URL") or config.get_main_option(
        "sqlalchemy.url", "sqlite:///audit.sqlite3"
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_url(), target_metadata=target_metadata, literal_binds=True, render_as_batch=True
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
