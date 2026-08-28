"""batch.window — take the next slice of work not yet done, so a large set can be processed in
windows.

Every operator module holds the same five things in this order: what it is, what it takes,
what it produces, and how it runs. The behaviour it calls lives in this family's own logic
modules; nothing here does the work itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...contracts import OperatorFamily, RiskLevel
from ..arguments import OperatorInput, config_value
from . import plans as planning

NAME = "batch.window"
FAMILY = OperatorFamily.PURE_TRANSFORM
OPERATION = "next_window"
SUMMARY = (
    "Take the next slice of work not yet done, so a large set can be processed in "
    "windows."
)
FEATURES = frozenset({"next_window"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    items: list[dict[str, Any]] | None = Field(
        default=None, description="The full set of work. Bind from fs.scan."
    )
    scanned: list[dict[str, Any]] | None = None
    completed: list[Any] | None = Field(
        default=None,
        description="Work already done in earlier windows. Bind from the accumulated "
        "results; ids or records both work.",
    )


OUTPUTS = ("window", "window_size", "remaining", "completed_count", "exhausted")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return planning.next_window(
        items=tuple(arguments.get("items") or arguments.get("scanned") or ()),
        # The agent may thread progress explicitly; if it does not, the runtime's own
        # record of what earlier rounds produced is used instead.
        completed=tuple(arguments.get("completed") or arguments.get("_completed") or ()),
        window_size=int(config_value(arguments, "schedule", "window_size", 20)),
    )