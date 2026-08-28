"""text.normalize — normalise whitespace, case and aliases in extracted values.

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
from . import text

NAME = "text.normalize"
FAMILY = OperatorFamily.PURE_TRANSFORM
OPERATION = "normalize"
SUMMARY = "Normalise whitespace, case and aliases in extracted values."
FEATURES = frozenset({"normalize"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    values: dict[str, Any] = Field(description="Field name to raw value.")


OUTPUTS = ("values",)


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return {
        "values": text.normalize(
            arguments["values"],
            collapse_whitespace=bool(arguments.get("collapse_whitespace", True)),
            strip=bool(arguments.get("strip", True)),
            aliases=arguments.get("aliases"),
            case=str(arguments.get("case", "preserve")),
        )
    }
