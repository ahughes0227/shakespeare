"""plan.assemble — assemble a ChangePlan and enforce balanced accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...contracts import (
    ChangeAction,
    OperatorFamily,
    RiskLevel,
)
from ..arguments import OperatorInput, RunnerError
from . import plans as planning

NAME = "plan.assemble"
FAMILY = OperatorFamily.PURE_TRANSFORM
OPERATION = "plan_assemble"
SUMMARY = "Assemble a ChangePlan and enforce balanced accounting."
FEATURES = frozenset({"plan_assemble"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    run_id: str
    workflow_id: str
    workflow_digest: str
    decision_digest: str
    scanned: list[dict[str, Any]] = Field(description="The inventory. Bind from fs.scan.")
    skipped: list[dict[str, Any]] | None = Field(
        default=None,
        description="Unreadable paths from fs.scan. Bind them so they appear in the plan.",
    )
    planned: list[dict[str, Any]] | None = Field(
        default=None,
        description="Resolved names. Must be bound from name.collide, never written by hand.",
    )


OUTPUTS = ("plan", "entries", "scanned")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    if "planned" in set(arguments.get("_agent_supplied") or ()):
        # Names must flow from name.render.  Allowing a hand-written `planned` here would
        # let an agent bypass the renderer and invent filenames directly, which is exactly
        # the inconsistency the frozen spec exists to prevent.
        raise RunnerError(
            "planned names must flow from name.render, not be supplied as parameters"
        )
    plan = planning.assemble_plan(
        run_id=arguments["run_id"],
        workflow_id=arguments["workflow_id"],
        workflow_digest=arguments["workflow_digest"],
        decision_digest=arguments["decision_digest"],
        scanned=tuple(planning.ScannedItem.model_validate(i) for i in arguments["scanned"]),
        planned=tuple(
            planning.PlannedName.model_validate(i) for i in arguments.get("planned") or ()
        ),
        skipped=tuple(arguments.get("skipped") or ()),
        operator_versions=arguments.get("operator_versions"),
        default_action=ChangeAction(arguments.get("default_action", "unresolved")),
    )
    payload = plan.model_dump(mode="json")
    return {
        "plan": payload,
        # Published as evidence so `balanced` and `resolved_or_quarantined` can be checked
        # without the runtime knowing anything about plans.
        "entries": payload["entries"],
        "scanned": len(arguments["scanned"]) + len(arguments.get("skipped") or ()),
    }
