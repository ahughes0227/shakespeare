"""fs.discard — discard an uncommitted staging tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts import OperatorFamily, RiskLevel
from . import mutation

NAME = "fs.discard"
FAMILY = OperatorFamily.FILESYSTEM_MUTATION
OPERATION = "discard"
SUMMARY = "Discard an uncommitted staging tree."
FEATURES = frozenset({"discard"})
SIDE_EFFECTS = ("write:staging_root",)
RISK = RiskLevel.HIGH
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = False

#: Reserved to the runtime, so no agent-facing argument model.
Input = None


OUTPUTS = ()


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    mutation.discard(Path(arguments["staging_root"]))
    return {"discarded": arguments["staging_root"]}
