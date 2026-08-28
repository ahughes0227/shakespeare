"""fs.scan — walk an input tree and inventory every file deterministically."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...contracts import OperatorFamily, RiskLevel
from ..arguments import OperatorInput
from . import inspection as filesystem

NAME = "fs.scan"
FAMILY = OperatorFamily.READONLY_SCAN
OPERATION = "walk"
SUMMARY = "Walk an input tree and inventory every file deterministically."
FEATURES = frozenset({"digest", "mime_detect", "stable_sort", "stat", "walk"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    root: str = Field(
        description="Directory to walk. Bind from a stage input; never a literal path."
    )
    depth_limit: int = Field(default=32, ge=1, le=64)
    include_hidden: bool = False


OUTPUTS = ("items", "skipped", "count")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    items, skipped = filesystem.scan(
        Path(arguments["root"]),
        depth_limit=int(arguments.get("depth_limit", 32)),
        include_hidden=bool(arguments.get("include_hidden", False)),
    )
    return {
        "items": [item.model_dump(mode="json") for item in items],
        "skipped": list(skipped),
        "count": len(items),
    }
