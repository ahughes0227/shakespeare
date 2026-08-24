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

from .agent import DomainAgent
from .audit import AuditStore
from .contracts import (
    Composition,
    DomainGoal,
    DomainSpec,
    ObligationResult,
    RequestContract,
    RouteDecision,
    StageDecision,
    StagePlan,
    StageSpec,
    StageVerdict,
)
from .gateway import ModelUsage


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
    """Replays a domain's recorded composition.  Makes no model call, ever."""

    recorded: _Recorded
    stage_of: dict[str, str]

    def compose(
        self,
        *,
        domain: DomainSpec,
        goal: DomainGoal,
        stage_inputs: dict[str, Any],
        catalog_summary: dict[str, Any],
    ) -> tuple[Composition, ModelUsage | None]:
        stage = self.stage_of.get(domain.id)
        if stage is None:
            raise ReplayError(f"domain {domain.id} belongs to no stage in this workflow")
        payload = self.recorded.peek(stage)["compositions"].get(domain.id)
        if payload is None:
            raise ReplayError(
                f"no composition was recorded for {stage}.{domain.id}; "
                f"the stage package has changed since this run"
            )
        return Composition.model_validate(payload), None


def journal_components(
    audit: AuditStore, run_id: str, *, stage_of: dict[str, str]
) -> tuple[JournalPlanner, dict[str, DomainAgent], str, str]:
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
    agent = JournalAgent(recorded, stage_of)
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
