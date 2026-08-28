"""fs.dirs — list the directory structure of a tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...contracts import OperatorFamily, RiskLevel
from ..arguments import OperatorInput
from . import inspection as filesystem

NAME = "fs.dirs"
FAMILY = OperatorFamily.READONLY_SCAN
OPERATION = "directories"
SUMMARY = "List the directory structure of a tree."
FEATURES = frozenset({"directories"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    root: str = Field(description="Directory whose structure to list. Bind from a stage input.")


OUTPUTS = ("directories",)


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return {"directories": list(filesystem.directories(Path(arguments["root"])))}
