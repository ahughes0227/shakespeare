"""schedule.plan — size the next batch of work from what earlier batches actually cost, or report
that the whole set already fits.

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

NAME = "schedule.plan"
FAMILY = OperatorFamily.PURE_TRANSFORM
OPERATION = "plan_batch"
SUMMARY = (
    "Size the next batch of work from what earlier batches actually cost, or "
    "report that the whole set already fits."
)
FEATURES = frozenset({"plan_batch"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    remaining: list[dict[str, Any]] | None = Field(
        default=None, description="Work not yet handed over."
    )
    items: list[dict[str, Any]] | None = None
    capacity: int = Field(description="Output tokens one response may use.")
    cost_per_item: int = Field(description="Starting estimate of one item's cost.")
    observations: list[dict[str, Any]] | None = Field(
        default=None, description="What earlier batches actually cost."
    )
    weights: list[int] | None = Field(
        default=None, description="How much material each remaining item carries."
    )


OUTPUTS = ("needed", "batch", "batch_size", "batch_weight", "remaining_count", "estimate")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return planning.plan_batch(
        remaining=tuple(arguments.get("remaining") or arguments.get("items") or ()),
        capacity=int(arguments["capacity"]),
        cost_per_item=int(arguments["cost_per_item"]),
        observations=tuple(arguments.get("observations") or ()),
        weights=tuple(arguments.get("weights") or ()),
        reserve=float(config_value(arguments, "schedule", "reserve", 0.6)),
        growth=float(config_value(arguments, "schedule", "growth", 2.0)),
        backoff=float(config_value(arguments, "schedule", "backoff", 0.5)),
    )