"""fs.commit — atomically move a verified staging tree into the output root.

Every operator module holds the same five things in this order: what it is, what it takes,
what it produces, and how it runs. The behaviour it calls lives in this family's own logic
modules; nothing here does the work itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts import OperatorFamily, RiskLevel
from . import mutation

NAME = "fs.commit"
FAMILY = OperatorFamily.FILESYSTEM_MUTATION
OPERATION = "atomic_move"
SUMMARY = "Atomically move a verified staging tree into the output root."
FEATURES = frozenset({"atomic_move", "idempotency_receipt"})
SIDE_EFFECTS = ("write:output_root",)
RISK = RiskLevel.HIGH
IDEMPOTENT = False
TIMEOUT_SECONDS = 300.0
COMPOSABLE = False

#: Reserved to the runtime, so no agent-facing argument model.
Input = None


OUTPUTS = ()


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    record = mutation.commit(
        staging_root=Path(arguments["staging_root"]),
        output_root=Path(arguments["output_root"]),
    )
    return {"reversal": record.model_dump(mode="json")}
