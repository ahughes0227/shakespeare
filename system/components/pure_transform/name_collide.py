"""name.collide — resolve duplicate targets deterministically across a whole set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...contracts import OperatorFamily, RiskLevel
from ..arguments import OperatorInput, config_value
from . import naming

NAME = "name.collide"
FAMILY = OperatorFamily.PURE_TRANSFORM
OPERATION = "collision_resolve"
SUMMARY = "Resolve duplicate targets deterministically across a whole set."
FEATURES = frozenset({"collision_resolve"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    candidates: list[dict[str, Any]] = Field(
        description="Rendered names: item_id, directory, name. Bind from name.render."
    )
    unrendered: list[dict[str, Any]] | None = None


OUTPUTS = ("resolutions",)


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    candidates = tuple(naming.Candidate.model_validate(item) for item in arguments["candidates"])
    resolutions = naming.resolve_collisions(
        candidates,
        naming.CollisionPolicy(config_value(arguments, "collision", "policy", "suffix_n")),
    )
    carried = [
        {"item_id": item["item_id"], "directory": "", "name": None, "reason": item["reason"]}
        for item in arguments.get("unrendered") or ()
    ]
    return {
        "resolutions": [item.model_dump(mode="json") for item in resolutions] + carried
    }
