from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.audit import AuditStore


@pytest.fixture
def store(tmp_path: Path) -> AuditStore:
    audit = AuditStore(tmp_path / "audit.sqlite3")
    yield audit
    audit.close()
