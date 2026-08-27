"""Deterministic replay.

A run is replayed by swapping the only two components that talk to a model — the planner
and the domain agents — for journal-backed ones.  Everything else is the production path:
the same verifier authorizes, the same executor runs, the same obligations are checked.

That is the point.  If replay had its own driver it would prove nothing; because it reuses
the real one, a successful replay is evidence that the recorded compositions really do
determine the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import (
    Composition,
    ObligationResult,
    RequestContract,
    RouteDecision,
    StageDecision,
    StagePlan,
    StageSpec,
    StageVerdict,
)
from ..gateway import ModelUsage
from .audit import AuditStore


class ReplayError(RuntimeError):
    pass


@dataclass
class _Recorded:
    """The journaled attempts for one run, consumed in the order they happened."""

    workflow_id: str
    workflow_digest: str
    attempts: list[dict[str, Any]]
    cursor: dict[str, int] = field(default_factory=dict)

    def next_attempt(self, stage: str) -> dict[str, Any]:
        index = self.cursor.get(stage, 0)
        matching = [item for item in self.attempts if item["stage_name"] == stage]
        if index >= len(matching):
            raise ReplayError(
                f"the journal records {len(matching)} attempt(s) for stage {stage!r}, "
                f"but replay asked for another. The workflow has changed since this run."
            )
        self.cursor[stage] = index + 1
        return matching[index]

    def peek(self, stage: str) -> dict[str, Any]:
        index = max(0, self.cursor.get(stage, 1) - 1)
        matching = [item for item in self.attempts if item["stage_name"] == stage]
        if index >= len(matching):
            raise ReplayError(f"no recorded attempt to replay for stage {stage!r}")
        return matching[index]


@dataclass
class JournalPlanner:
    """Replays the planner's recorded decisions.  Makes no model call, ever."""

    recorded: _Recorded
    model_calls: int = 0

    def select_workflow(
        self, request: RequestContract, catalog: dict[str, dict[str, str]]
    ) -> tuple[RouteDecision, ModelUsage | None]:
        return (
            RouteDecision(
                workflow_id=self.recorded.workflow_id,
                supported=True,
                rationale="replayed from the audit log",
            ),
            None,
        )

    def select_goal(self, open_goals: Any, artifacts: list[dict[str, Any]]) -> str:
        """Replay the recorded order.

        The journal records which goal was pursued and when, so replay follows that
        rather than re-deciding — a re-decision would be a different run.
        """
        pursued = [item["stage_name"] for item in self.recorded.attempts]
        available = [goal.id for goal in open_goals]
        for goal_id in pursued:
            if goal_id in available:
                return str(goal_id)
        return str(available[0])

    def select_capability(
        self,
        goal: Any,
        candidates: list[dict[str, Any]],
        evidence: dict[str, Any] | None = None,
    ) -> Any:
        """Replay the recorded choice, ignoring the evidence the original was shown.

        A replay reproduces decisions rather than re-taking them; consulting the corpus
        again is exactly how a replay would stop being one.
        """
        from ..planner import CapabilityChoice

        names = [item["id"] if isinstance(item, dict) else item for item in candidates]
        for item in self.recorded.attempts:
            if item["stage_name"] == goal.id:
                for capability_id in item["compositions"]:
                    if capability_id in names:
                        return CapabilityChoice(capability_id=str(capability_id))
        return CapabilityChoice(capability_id=str(names[0]))

    def judge(
        self,
        *,
        goal: Any,
        rubric: str,
        artifacts: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> tuple[bool, str]:
        """Replay the recorded verdict, so a semantic gate does not re-decide."""
        for item in self.recorded.attempts:
            if item["stage_name"] == goal.id and item["verdict"] is not None:
                return bool(item["verdict"]["met"]), "replayed from the audit log"
        return True, "replayed: no recorded verdict"

    def plan_stage(
        self, stage: StageSpec, request: RequestContract, stage_inputs: dict[str, Any]
    ) -> tuple[StagePlan, ModelUsage | None]:
        attempt = self.recorded.next_attempt(stage.name)
        if attempt["plan"] is None:
            raise ReplayError(f"no stage plan was recorded for {stage.name}")
        return StagePlan.model_validate(attempt["plan"]), None

    def review_stage(
        self,
        stage: StageSpec,
        plan: StagePlan,
        obligations: tuple[ObligationResult, ...],
        summary: dict[str, Any],
        *,
        attempts_remaining: int,
    ) -> tuple[StageVerdict, ModelUsage | None]:
        recorded = self.recorded.peek(stage.name)["verdict"]
        if recorded is None:
            raise ReplayError(f"no verdict was recorded for {stage.name}")
        import json

        return (
            StageVerdict(
                met=bool(recorded["met"]),
                unmet=tuple(json.loads(recorded["unmet"])),
                decision=StageDecision(recorded["decision"]),
                rationale=recorded["rationale"],
            ),
            None,
        )


@dataclass
class JournalAgent:
    """Replays a capability's recorded organization. Makes no model call, ever."""

    recorded: _Recorded
    #: capability id -> the goal attempts it was recorded under, in order.
    _cursor: dict[str, int] = field(default_factory=dict)

    def organize(
        self,
        *,
        capability: Any,
        request: str,
        artifacts: list[dict[str, Any]],
        context: dict[str, Any],
        prior: list[dict[str, Any]],
        catalog_summary: dict[str, Any],
    ) -> tuple[Any, ModelUsage | None]:
        from ..capabilities.runner import Organization

        rounds = [
            payload
            for attempt in self.recorded.attempts
            for domain_id, payload in attempt["compositions"].items()
            if domain_id == capability.id
        ]
        index = self._cursor.get(capability.id, 0)
        if index >= len(rounds):
            raise ReplayError(
                f"the journal records {len(rounds)} round(s) for capability "
                f"{capability.id!r}, but replay asked for another. The workflow has "
                f"changed since this run."
            )
        self._cursor[capability.id] = index + 1
        composition = Composition.model_validate(rounds[index])
        final = index + 1 >= len(rounds)
        return (
            Organization(
                invocations=composition.invocations,
                intent=composition.rationale,
                # The recorded run stopped where it stopped; the last round finished it.
                sufficient=final,
                # The journal records what ran, not what was published. A capability
                # publishes what its package declares, so the final round republishes it.
                # A capability declaring more than one kind would need the journal
                # extended; none does today, and registration would have to allow it.
                publishes=capability.produces[0] if final and capability.produces else None,
            ),
            None,
        )


def journal_components(
    audit: AuditStore, run_id: str, *, stage_of: dict[str, str] | None = None
) -> tuple[JournalPlanner, dict[str, Any], str, str]:
    """Build the replay planner and agents for a recorded run."""
    source = audit.replay_source(run_id)
    recorded = _Recorded(
        workflow_id=source["workflow_id"],
        workflow_digest=source["workflow_digest"],
        attempts=list(source["attempts"]),
    )
    if not recorded.attempts:
        raise ReplayError(f"run {run_id} recorded no stage attempts to replay")

    planner = JournalPlanner(recorded)
    agent = JournalAgent(recorded)
    return planner, {"*": agent}, source["workflow_id"], source["workflow_digest"]


def assert_same_workflow(recorded_digest: str, current_digest: str) -> None:
    """Refuse to replay against a changed workflow.

    The digest covers pinned stage versions and pinned prompt versions, so a mismatch
    means the run being replayed is not the run this code would produce.  Replaying
    anyway would give a confident answer to the wrong question.
    """
    if recorded_digest != current_digest:
        raise ReplayError(
            "the workflow has changed since this run was recorded "
            f"(recorded {recorded_digest[:12]}, current {current_digest[:12]}). "
            "Replay is only meaningful against the workflow that produced the run."
        )
