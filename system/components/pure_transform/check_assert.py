"""check.assert — run a named deterministic obligation check.

Every operator module holds the same five things in this order: what it is, what it takes,
what it produces, and how it runs. The behaviour it calls lives in this family's own logic
modules; nothing here does the work itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...contracts import OperatorFamily, RiskLevel
from ...runtime import checks
from ..arguments import OperatorInput

NAME = "check.assert"
FAMILY = OperatorFamily.PURE_TRANSFORM
OPERATION = "obligation_check"
SUMMARY = "Run a named deterministic obligation check."
FEATURES = frozenset({"obligation_check"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    obligation_id: str
    check: str
    payload: dict[str, Any] = Field(default_factory=dict)


OUTPUTS = ("obligation_id", "passed", "detail")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    result = checks.run_check(
        arguments["obligation_id"], arguments["check"], arguments.get("payload", {})
    )
    return result.model_dump(mode="json")
