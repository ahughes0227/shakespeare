"""fs.verify — re-scan a staged tree and compare it against a plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...contracts import (
    ChangePlan,
    OperatorFamily,
    RiskLevel,
)
from ..arguments import OperatorInput
from ..filesystem_mutation import mutation

NAME = "fs.verify"
FAMILY = OperatorFamily.READONLY_SCAN
OPERATION = "verify_staging"
SUMMARY = "Re-scan a staged tree and compare it against a plan."
FEATURES = frozenset({"digest", "verify_staging"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    plan: dict[str, Any] = Field(description="The ChangePlan to verify against.")
    staging_root: str


OUTPUTS = ("ok", "missing", "mismatched", "staged_files", "planned_entries")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return dict(
        mutation.verify_tree(
            plan=ChangePlan.model_validate(arguments["plan"]),
            staging_root=Path(arguments["staging_root"]),
        )
    )
