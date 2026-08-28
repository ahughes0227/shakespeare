"""Goals and gates.

A goal states what must become true (§4). It never states how — that decomposition
belongs to whichever capability answers it.

A gate decides whether the artifacts now available sufficiently satisfy a goal (§5).
Deterministic where the question has an exact answer, semantic where it genuinely
requires judgment, hybrid where a deterministic floor must hold before judgment is worth
asking for.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from ..contracts import Contract
from .artifacts import Artifact, Quality


class GateKind(StrEnum):
    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"
    #: A deterministic floor that must hold, then a judgment about sufficiency. Use this
    #: rather than a bare semantic gate wherever part of the question has an exact answer:
    #: never ask a model something a check can settle.
    HYBRID = "hybrid"


class Gate(Contract):
    id: str = Field(min_length=1)
    kind: GateKind
    #: Artifact kinds that must exist before the goal can be considered at all.
    requires: tuple[str, ...] = ()
    #: Named deterministic checks, run against the artifacts.
    checks: tuple[str, ...] = ()
    #: What a model should weigh when the question needs judgment. A rubric structures the
    #: judgment without pretending it is deterministic (§5).
    rubric: str = ""
    #: Reject an artifact whose quality is below this. PARTIAL passes by default, because
    #: a capability that has done part of the work honestly should not read as failure.
    minimum_quality: Quality = Quality.PARTIAL

    @model_validator(mode="after")
    def _coherent(self) -> Gate:
        if self.kind is not GateKind.SEMANTIC and not self.checks and not self.requires:
            raise ValueError(f"gate {self.id} is deterministic but checks nothing")
        if self.kind is not GateKind.DETERMINISTIC and not self.rubric:
            raise ValueError(f"gate {self.id} asks for judgment without a rubric")
        return self


class Goal(Contract):
    id: str = Field(min_length=1)
    #: What must become true. An outcome, never a procedure.
    statement: str = Field(min_length=1)
    #: Other goals whose artifacts this one materially depends on. Only real causal
    #: dependencies: anything else forces sequence where none exists (§3).
    depends_on: tuple[str, ...] = ()
    gate: Gate
    #: Capabilities permitted to answer this goal. Containment, not procedure — several
    #: may be able to, and the planner chooses.
    capabilities: tuple[str, ...] = Field(min_length=1)
    #: Human-facing label for dashboards and audit. Not an execution primitive (§12).
    label: str = ""


class GateOutcome(StrEnum):
    SATISFIED = "satisfied"
    INSUFFICIENT = "insufficient"
    #: The deterministic floor failed, so judgment was never asked for.
    BLOCKED = "blocked"


class GateResult(Contract):
    gate_id: str
    goal_id: str
    outcome: GateOutcome
    missing_kinds: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def satisfied(self) -> bool:
        return self.outcome is GateOutcome.SATISFIED


def evidence_for(gate: Gate, artifacts: tuple[Artifact, ...]) -> tuple[str, ...]:
    """Artifact kinds the gate requires that are absent or below its quality floor."""
    order = [Quality.EMPTY, Quality.DEGRADED, Quality.PARTIAL, Quality.COMPLETE]
    floor = order.index(gate.minimum_quality)
    present = {
        item.kind
        for item in artifacts
        if item.quality in order and order.index(item.quality) >= floor
    }
    return tuple(kind for kind in gate.requires if kind not in present)


class GoalGraph(Contract):
    """A workflow: goals and the dependencies that actually matter."""

    goals: tuple[Goal, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _acyclic_and_resolvable(self) -> GoalGraph:
        known = {goal.id for goal in self.goals}
        if len(known) != len(self.goals):
            raise ValueError("goal ids must be unique")
        for goal in self.goals:
            unknown = set(goal.depends_on) - known
            if unknown:
                raise ValueError(f"goal {goal.id} depends on unknown goals: {sorted(unknown)}")

        # A cycle would leave every goal in it permanently unreachable, and the planner
        # would loop looking for something to do.
        colour: dict[str, int] = {}

        def visit(goal_id: str, stack: tuple[str, ...]) -> None:
            state = colour.get(goal_id, 0)
            if state == 1:
                raise ValueError(f"goal dependencies form a cycle: {' -> '.join(stack)}")
            if state == 2:
                return
            colour[goal_id] = 1
            for parent in self.goal(goal_id).depends_on:
                visit(parent, (*stack, parent))
            colour[goal_id] = 2

        for goal in self.goals:
            visit(goal.id, (goal.id,))
        return self

    def goal(self, goal_id: str) -> Goal:
        for item in self.goals:
            if item.id == goal_id:
                return item
        raise KeyError(f"unknown goal: {goal_id}")

    def open_goals(self, satisfied: frozenset[str]) -> tuple[Goal, ...]:
        """Goals not yet satisfied whose dependencies are.

        More than one may come back, and that is the point: independent work should not
        be forced into a sequence (§3).
        """
        return tuple(
            goal
            for goal in self.goals
            if goal.id not in satisfied and set(goal.depends_on) <= satisfied
        )

    def blocked_goals(self, satisfied: frozenset[str]) -> tuple[Goal, ...]:
        return tuple(
            goal
            for goal in self.goals
            if goal.id not in satisfied and not set(goal.depends_on) <= satisfied
        )

    def describe(self) -> list[dict[str, Any]]:
        """What the planner is shown about the graph."""
        return [
            {
                "id": goal.id,
                "statement": goal.statement,
                "depends_on": list(goal.depends_on),
                "capabilities": list(goal.capabilities),
                "requires_artifacts": list(goal.gate.requires),
            }
            for goal in self.goals
        ]
