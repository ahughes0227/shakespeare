"""record.append — store one row per item in the run's record table, replacing any row already held
for that item.

Every operator module holds the same five things in this order: what it is, what it takes,
what it produces, and how it runs. The behaviour it calls lives in this family's own logic
modules; nothing here does the work itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...contracts import OperatorFamily, RiskLevel
from ..arguments import OperatorInput
from . import storage as records

NAME = "record.append"
FAMILY = OperatorFamily.RECORD_STORE
OPERATION = "record_append"
SUMMARY = (
    "Store one row per item in the run's record table, replacing any row already "
    "held for that item."
)
FEATURES = frozenset({"record_append"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    rows: list[dict[str, Any]] = Field(
        description="One row per item. Each must carry the key column."
    )
    table: str = Field(default="items", description="Which table to write.")
    key: str = Field(default="item_id", description="Column identifying an item.")


OUTPUTS = ("table", "stored", "added", "replaced", "path")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    rows = arguments.get("rows") or arguments.get("records") or ()
    return records.append(
        workspace=workspace,
        table=str(arguments.get("table") or "items"),
        rows=tuple(rows),
        key=str(arguments.get("key") or "item_id"),
    )