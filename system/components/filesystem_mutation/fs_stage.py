"""fs.stage — materialise a plan into a staging tree by copying."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts import (
    ChangePlan,
    OperatorFamily,
    RiskLevel,
)
from ..arguments import config_value
from . import mutation

NAME = "fs.stage"
FAMILY = OperatorFamily.FILESYSTEM_MUTATION
OPERATION = "stage_write"
SUMMARY = "Materialise a plan into a staging tree by copying."
FEATURES = frozenset({"journal_reverse", "mirror_tree", "stage_write"})
SIDE_EFFECTS = ("write:staging_root",)
RISK = RiskLevel.HIGH
IDEMPOTENT = True
TIMEOUT_SECONDS = 3600.0
COMPOSABLE = False

#: Reserved to the runtime, so no agent-facing argument model.
Input = None


OUTPUTS = ()


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    plan = ChangePlan.model_validate(arguments["plan"])
    reversals = mutation.stage_plan(
        plan=plan,
        input_root=Path(arguments["input_root"]),
        staging_root=Path(arguments["staging_root"]),
        quarantine_dirname=config_value(arguments, "write", "quarantine_dirname", "_unresolved"),
    )
    return {"reversals": [item.model_dump(mode="json") for item in reversals]}
