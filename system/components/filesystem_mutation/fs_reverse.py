"""fs.reverse — reverse one journaled mutation.

Every operator module holds the same five things in this order: what it is, what it takes,
what it produces, and how it runs. The behaviour it calls lives in this family's own logic
modules; nothing here does the work itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts import (
    OperatorFamily,
    ReversalRecord,
    RiskLevel,
)
from . import mutation

NAME = "fs.reverse"
FAMILY = OperatorFamily.FILESYSTEM_MUTATION
OPERATION = "journal_reverse"
SUMMARY = "Reverse one journaled mutation."
FEATURES = frozenset({"journal_reverse"})
SIDE_EFFECTS = ("write:output_root",)
RISK = RiskLevel.HIGH
IDEMPOTENT = False
TIMEOUT_SECONDS = 300.0
COMPOSABLE = False

#: Reserved to the runtime, so no agent-facing argument model.
Input = None


OUTPUTS = ()


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    record = ReversalRecord.model_validate(arguments["reversal"])
    mutation.reverse(record)
    return {"reversed": record.mutation_id}
