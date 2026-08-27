"""The control loop (§19).

    workflow goal
        -> planner asks what it needs next and who can answer it
        -> capability plans within its bounded domain
        -> components execute deterministically
        -> artifacts produced
        -> gate evaluates goal satisfaction
        -> satisfied ? advance : planner decides what is missing

This replaces walking a fixed spine. The loop does not know it is in a stage called
"extract"; it knows which goals remain open, what evidence exists, and which capabilities
could produce what is missing.

Everything transactional is unchanged: staging, balanced accounting, two-phase commit,
journalled reversal, replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifacts import ArtifactStore
from .audit import AuditStore
from .capabilities import CapabilityRegistry, CapabilityRunner, CapabilitySpec
from .capabilities.runner import CapabilityOutcome
from .contracts import (
    ChangePlan,
    ErrorCode,
    RequestContract,
    content_digest,
    utc_now,
)
from .domain import mutation
from .gating import GateEvaluator
from .goals import GateResult, Goal, GoalGraph
from .telemetry import Tracer
from .verifier import Denial


class _NoSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _no_span() -> _NoSpan:
    return _NoSpan()


class ControlError(RuntimeError):
    def __init__(self, message: str, code: ErrorCode) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GoalAttempt:
    goal_id: str
    capability: str
    outcome: CapabilityOutcome
    gate: GateResult

    @property
    def satisfied(self) -> bool:
        return self.gate.satisfied


@dataclass
class RunReport:
    run_id: str
    workflow_id: str
    outcome: str
    plan: ChangePlan | None = None
    committed_to: str | None = None
    attempts: tuple[GoalAttempt, ...] = ()
    satisfied: frozenset[str] = frozenset()
    error_code: ErrorCode | None = None
    detail: str = ""
    planned_output_root: str | None = None

    @property
    def committed(self) -> bool:
        return self.outcome == "committed"


@dataclass
class Controller:
    """The deterministic runtime (§10). It schedules; it does not reason."""

    capabilities: CapabilityRegistry
    runner: CapabilityRunner
    artifacts: ArtifactStore
    audit: AuditStore
    planner: Any
    workspace: Path
    tracer: Tracer | None = None
    #: The ceiling on attempts at one goal. Rarely what stops a run: an attempt that
    #: achieves nothing new stops it sooner, so this only bounds a goal that keeps
    #: making progress without ever arriving.
    max_goal_attempts: int = 6
    #: Called with the working context each time a goal is satisfied. The loop announces
    #: that something became true; what to do about it is the runtime's business, because
    #: acting on it may mean writing and the loop never writes.
    on_goal_satisfied: Any = None
    context: dict[str, Any] = field(default_factory=dict)

    def pursue(
        self,
        *,
        graph: GoalGraph,
        request: RequestContract,
        run_id: str,
        budget_for: Any,
    ) -> tuple[tuple[GoalAttempt, ...], frozenset[str], str | None]:
        """Work open goals until none remain or one cannot be satisfied."""
        evaluator = GateEvaluator(artifacts=self.artifacts, judge=self.planner)
        satisfied: set[str] = set()
        attempts: list[GoalAttempt] = []
        #: goal id -> why its last attempt was rejected.
        rejected: dict[str, dict[str, Any]] = {}
        tried: dict[str, int] = {}
        #: goal id -> what its last rejected attempt had achieved.
        standing: dict[str, tuple[Any, ...]] = {}

        while True:
            open_goals = graph.open_goals(frozenset(satisfied))
            if not open_goals:
                blocked = graph.blocked_goals(frozenset(satisfied))
                if blocked:
                    return (
                        tuple(attempts),
                        frozenset(satisfied),
                        f"goals remain blocked with nothing open: "
                        f"{sorted(item.id for item in blocked)}",
                    )
                return tuple(attempts), frozenset(satisfied), None

            goal = self._choose_goal(open_goals, graph)
            tried[goal.id] = tried.get(goal.id, 0) + 1
            if tried[goal.id] > self.max_goal_attempts:
                return (
                    tuple(attempts),
                    frozenset(satisfied),
                    f"goal {goal.id!r} could not be satisfied in "
                    f"{self.max_goal_attempts} attempts",
                )

            capability, impediment = self._choose_capability(goal)
            if capability is None:
                # Not a failure to be retried: a statement that no attempt of this shape
                # will work. It ends the run for a person to read.
                return (
                    tuple(attempts),
                    frozenset(satisfied),
                    f"impediment at goal {goal.id!r}: {impediment}",
                )
            goal_span = (
                self.tracer.span(
                    f"goal.{goal.id}",
                    stage=goal.id,
                    attempt=tried[goal.id],
                    domain=capability.id,
                )
                if self.tracer
                else _no_span()
            )
            with goal_span as span:
                outcome = self.runner.run(
                    capability=capability,
                    request=goal.statement,
                    context=self.context,
                    budget=budget_for(goal, self.context),
                    workspace=self.workspace,
                    goal_id=goal.id,
                    feedback=rejected.get(goal.id),
                )
                self.context.update(outcome.context)
                # Local to the attempt that was told it: a later goal has its own history
                # and should not inherit this one's.
                self.context.pop("previous_attempt", None)

                result = evaluator.evaluate(
                    goal, self.context, exhausted=outcome.exhausted
                )
                if span is not None:
                    span.add_count("rounds", len(outcome.rounds))
                    span.add_count(
                        "rounds_failed",
                        sum(1 for item in outcome.rounds if not item.succeeded),
                    )
                    span.record(
                        outcome=str(result.outcome),
                        sufficient=outcome.sufficient,
                        # Check and artifact-kind names are declared in packages, so
                        # neither can carry content.
                        failed_checks=result.failed_checks,
                        missing_kinds=result.missing_kinds,
                    )
                    if not result.satisfied:
                        span.fail(ErrorCode.OBLIGATION_FAILED)
            attempts.append(
                GoalAttempt(
                    goal_id=goal.id,
                    capability=capability.id,
                    outcome=outcome,
                    gate=result,
                )
            )
            if outcome.impediment:
                return (
                    tuple(attempts),
                    frozenset(satisfied),
                    f"impediment at goal {goal.id!r}: {outcome.impediment}",
                )
            if result.satisfied:
                satisfied.add(goal.id)
                if self.on_goal_satisfied is not None:
                    self.on_goal_satisfied(self.context)
            else:
                # What the gate actually objected to, carried into the next attempt.
                # Without it a retry can only repeat itself and hope for a better roll.
                rejected[goal.id] = {
                    "attempt": tried[goal.id],
                    "failed_checks": list(result.failed_checks),
                    "missing_evidence": list(result.missing_kinds),
                    "rationale": result.rationale,
                }
                # An attempt that changed nothing will not change anything next time
                # either. Counting attempts bounds the damage; noticing that one achieved
                # nothing is what actually stops the loop, and it stops it at the first
                # repetition rather than at an arbitrary count.
                reached = _achievement(self.context, result)
                if standing.get(goal.id) == reached:
                    return (
                        tuple(attempts),
                        frozenset(satisfied),
                        f"goal {goal.id!r} was attempted again and achieved nothing new: "
                        f"{result.rationale}",
                    )
                standing[goal.id] = reached

    # -- selection ---------------------------------------------------------------------



    def _choose_goal(self, open_goals: tuple[Goal, ...], graph: GoalGraph) -> Goal:
        """Which open goal to work next.

        Deterministic when there is no real choice, which is most of the time. Asking a
        model to pick between one option is the kind of semantic call §13 says to avoid.
        """
        if len(open_goals) == 1:
            return open_goals[0]
        chosen = self.planner.select_goal(open_goals, self.artifacts.describe())
        return graph.goal(chosen)

    def _choose_capability(self, goal: Goal) -> tuple[CapabilitySpec | None, str | None]:
        """Which capability answers this goal — decided from the corpus, not from names.

        The facts that decide it are the ones the scheduler was computing privately: how
        much there is to do, what one item costs, and what a single response can hold.
        Handing them over is the difference between a planner that picks a name and one
        that picks a shape.
        """
        candidates = [
            self.capabilities.get(name)
            for name in goal.capabilities
            if name in self.capabilities
        ]
        if not candidates:
            raise ControlError(
                f"goal {goal.id!r} names no registered capability",
                ErrorCode.COMPOSITION_INVALID,
            )
        if len(candidates) == 1:
            return candidates[0], None
        choice = self.planner.select_capability(
            goal,
            [
                {
                    "id": spec.id,
                    "standing_goal": spec.standing_goal,
                    "cost_per_item": spec.cost_per_item,
                    "durable": bool(spec.catalog & _DURABLE),
                }
                for spec in candidates
            ],
            self._corpus_evidence(candidates),
        )
        if getattr(choice, "impediment", None):
            return None, choice.impediment
        for spec in candidates:
            if spec.id == getattr(choice, "capability_id", choice):
                return spec, None
        return candidates[0], None

    def _corpus_evidence(self, candidates: list[CapabilitySpec]) -> dict[str, Any]:
        """What there is to do, and what one response can hold."""
        divides = {spec.divides for spec in candidates}
        counts = {
            key: len(value)
            for key in sorted(divides)
            if isinstance(value := self.context.get(key), list)
        }
        return {
            "items": counts,
            "response_ceiling_tokens": getattr(self.runner, "capacity", None),
        }


def commit_if_verified(
    *,
    plan: ChangePlan | None,
    staging: Path,
    output_root: Path,
    audit: AuditStore,
    run_id: str,
) -> tuple[str, str]:
    """Two-phase commit, unchanged by the convergence.

    Returns (outcome, detail).
    """
    if plan is None:
        return "aborted", "the workflow produced no change plan"

    existing = audit.find_commit(plan_digest=plan.fingerprint(), output_root=str(output_root))
    if existing is not None and output_root.exists():
        mutation.discard(staging)
        return "committed", f"already committed by run {existing['run_id']}; nothing to do"

    report = mutation.verify_tree(plan=plan, staging_root=staging)
    if not report["ok"]:
        mutation.discard(staging)
        return "aborted", f"staging does not match the plan: {report}"

    staging_digest = content_digest(report)
    record = mutation.commit(staging_root=staging, output_root=output_root)
    audit.record_commit(
        run_id=run_id, plan=plan, staging_digest=staging_digest, output_root=str(output_root)
    )
    audit.record_mutation(
        run_id=run_id,
        target_ref=str(output_root),
        operation="commit",
        reversal=record,
        after_digest=staging_digest,
    )
    return "committed", ""


def new_run_id() -> str:
    return uuid4().hex


def started_at() -> str:
    return utc_now().isoformat()


__all__ = [
    "Controller",
    "ControlError",
    "Denial",
    "GoalAttempt",
    "RunReport",
    "commit_if_verified",
    "new_run_id",
    "started_at",
]


#: Components that put a per-item result somewhere it survives the response reporting it.
#: A capability holding one can be told apart from one that carries its results in a
#: model's answer, which is the distinction the choice actually turns on.
_DURABLE: frozenset[str] = frozenset({"record.append"})


def _achievement(context: dict[str, Any], result: Any) -> tuple[Any, ...]:
    """What an attempt actually got to, in a form two attempts can be compared by.

    Sizes rather than contents: an attempt that resolved one more item has moved, and one
    that produced the same amount of the same evidence and failed the same checks has not.
    Contents would be stricter and worse — a model rephrasing one field would read as
    progress forever.
    """
    evidence = tuple(
        (key, len(value))
        for key, value in sorted(context.items())
        if isinstance(value, list) and not key.startswith("_")
    )
    return (evidence, tuple(result.failed_checks), tuple(result.missing_kinds))
