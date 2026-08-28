"""Schema migrations.

The audit log is permanent, so its schema is migrated rather than recreated. The test
that matters is drift: if a table is added to the models and not to the migrations, a
fresh install and an upgraded one diverge, and the second one breaks in production
rather than here.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from system.runtime.audit import AuditStore
from system.runtime.audit.schema import metadata

ROOT = Path(__file__).resolve().parents[1]


def _alembic_upgrade(database: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "SHAKESPEARE_AUDIT_URL": f"sqlite:///{database}"},
    )


def _schema_of(database: Path) -> dict[str, set[str]]:
    """Tables to column names, plus the trigger set, ignoring Alembic's own bookkeeping."""
    connection = sqlite3.connect(database)
    try:
        tables = {
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        } - {"alembic_version"}
        columns = {
            table: {row[1] for row in connection.execute(f"PRAGMA table_info('{table}')")}
            for table in tables
        }
        triggers = {
            name
            for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        }
    finally:
        connection.close()
    return {"tables": set(tables), "triggers": triggers, **columns}


@pytest.fixture
def migrated(tmp_path: Path) -> Path:
    database = tmp_path / "migrated.sqlite3"
    result = _alembic_upgrade(database)
    assert result.returncode == 0, result.stderr
    return database


class TestBaseline:
    def test_upgrade_creates_every_table(self, migrated: Path) -> None:
        assert _schema_of(migrated)["tables"] == {table.name for table in metadata.sorted_tables}

    def test_upgrade_installs_the_append_only_triggers(self, migrated: Path) -> None:
        triggers = _schema_of(migrated)["triggers"]
        for table in metadata.sorted_tables:
            assert f"{table.name}_no_update" in triggers
            assert f"{table.name}_no_delete" in triggers

    def test_the_migrated_log_is_actually_append_only(self, migrated: Path) -> None:
        connection = sqlite3.connect(migrated)
        connection.execute(
            "INSERT INTO runs VALUES ('r','now','w','1.0.0','a','b','c')"
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE runs SET workflow_id='x'")
        connection.close()


class TestNoDrift:
    def test_migrations_and_models_agree(self, migrated: Path, tmp_path: Path) -> None:
        """A table added to the models but not the migrations breaks upgraded installs."""
        fresh = tmp_path / "fresh.sqlite3"
        store = AuditStore(fresh)
        store.close()

        from_models = _schema_of(fresh)
        from_migrations = _schema_of(migrated)
        assert from_migrations["tables"] == from_models["tables"]
        for table in from_models["tables"]:
            assert from_migrations[table] == from_models[table], f"column drift in {table}"
        assert from_migrations["triggers"] == from_models["triggers"]


class TestDowngrade:
    def test_downgrade_is_refused(self, migrated: Path) -> None:
        """Dropping an append-only ledger would destroy the provenance it exists to keep.

        Run against a genuinely migrated database: an empty one is already at base, so
        Alembic would have nothing to downgrade and would exit successfully.
        """
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "downgrade", "base"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "SHAKESPEARE_AUDIT_URL": f"sqlite:///{migrated}"},
        )
        assert result.returncode != 0
        assert "append-only and permanent" in result.stderr
