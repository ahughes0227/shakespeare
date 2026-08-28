"""record.read — read back every stored row, so later work runs from the table rather than from a
model response.

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

NAME = "record.read"
FAMILY = OperatorFamily.RECORD_STORE
OPERATION = "record_read"
SUMMARY = (
    "Read back every stored row, so later work runs from the table rather than "
    "from a model response."
)
FEATURES = frozenset({"record_read"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    table: str = Field(default="items", description="Which table to read.")


OUTPUTS = ("records", "stored", "table")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return records.read(workspace=workspace, table=str(arguments.get("table") or "items"))