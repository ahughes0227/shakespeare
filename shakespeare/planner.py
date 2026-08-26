"""The supervisory loop.

The planner is the only adaptive element in the system, and it adapts only at stage
boundaries: it selects a prebuilt workflow, decides which domains to activate, issues
each a verifiable goal, then reviews the completed attempt and decides whether to accept,
rerun with revised goals, or abort.

It never builds a workflow, stage, domain or operator, and it cannot widen any surface —
a `DomainGoal` carries no catalog, no config groups and no budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import (
    Contract,
    ObligationResult,
    RequestContract,
    RouteDecision,
    StageDecision,
    StagePlan,
    StageSpec,
    StageVerdict,
)
from .gateway import Gateway, ModelProfile, ModelUsage, render_prompt
from .prompts import PromptStore

ROUTE_SIGNATURE = "planner.route"
PLAN_SIGNATURE = "planner.stage_plan"
REVIEW_SIGNATURE = "planner.stage_review"


class Planner(Protocol):
    def select_workflow(
        self, request: RequestContract, catalog: dict[str, dict[str, str]]
    ) -> tuple[RouteDecision, ModelUsage | None]: ...

    def plan_stage(
        self, stage: StageSpec, request: RequestContract, stage_inputs: dict[str, Any]
    ) -> tuple[StagePlan, ModelUsage | None]: ...

    def review_stage(
        self,
        stage: StageSpec,
        plan: StagePlan,
        obligations: tuple[ObligationResult, ...],
        summary: dict[str, Any],
        *,
        attempts_remaining: int,
    ) -> tuple[StageVerdict, ModelUsage | None]: ...


@dataclass
class ModelPlanner:
    gateway: Gateway
    profile: ModelProfile
    prompts: PromptStore = field(default_factory=PromptStore)
    #: Pinned per signature: these are independently versioned artifacts and promoting
    #: one must not silently move the others.
    route_version: str = "1.0.0"
    plan_version: str = "1.0.0"
    review_version: str = "1.1.0"

    def select_workflow(
        self, request: RequestContract, catalog: dict[str, dict[str, str]]
    ) -> tuple[RouteDecision, ModelUsage | None]:
        """Choose among registered workflows.

        `catalog` holds only the ten-field cards, so adding a workflow extends routing
        without touching this module.
        """
        artifact = self.prompts.load(ROUTE_SIGNATURE, self.route_version)
        messages = render_prompt(artifact, prompt=request.prompt, workflows=catalog)
        return self.gateway.complete(self.profile, messages, RouteDecision)

    def plan_stage(
        self, stage: StageSpec, request: RequestContract, stage_inputs: dict[str, Any]
    ) -> tuple[StagePlan, ModelUsage | None]:
        artifact = self.prompts.load(PLAN_SIGNATURE, self.plan_version)
        messages = render_prompt(
            artifact,
            request=request.prompt,
            stage=stage.name,
            stage_goal=stage.goal,
            domains=[
                {"id": domain.id, "scope": domain.scope, "skippable": domain.skippable}
                for domain in stage.domains
            ],
            obligations=list(stage.obligations),
            inputs=sorted(stage_inputs),
        )
        return self.gateway.complete(self.profile, messages, StagePlan)

    def review_stage(
        self,
        stage: StageSpec,
        plan: StagePlan,
        obligations: tuple[ObligationResult, ...],
        summary: dict[str, Any],
        *,
        attempts_remaining: int,
    ) -> tuple[StageVerdict, ModelUsage | None]:
        artifact = self.prompts.load(REVIEW_SIGNATURE, self.review_version)
        messages = render_prompt(
            artifact,
            stage=stage.name,
            stage_goal=stage.goal,
            activated=[goal.domain_id for goal in plan.activated],
            skipped=[skip.domain_id for skip in plan.skipped],
            obligations=[item.model_dump(mode="json") for item in obligations],
            summary=summary,
            attempts_remaining=attempts_remaining,
        )
        return self.gateway.complete(self.profile, messages, StageVerdict)


def obligations_met(results: tuple[ObligationResult, ...]) -> bool:
    return all(item.passed for item in results)


def default_verdict(
    results: tuple[ObligationResult, ...], *, attempts_remaining: int
) -> StageVerdict:
    """The verdict a planner should not be able to talk its way out of.

    Deterministic obligations are a hard gate: if one failed, the only choices are rerun
    or abort.  This is used to constrain a model verdict, not to replace it.
    """
    unmet = tuple(item.obligation_id for item in results if not item.passed)
    if not unmet:
        return StageVerdict(met=True, decision=StageDecision.ACCEPT)
    decision = StageDecision.RERUN if attempts_remaining > 0 else StageDecision.ABORT
    return StageVerdict(
        met=False,
        unmet=unmet,
        decision=decision,
        rationale=f"unmet obligations: {', '.join(unmet)}",
    )


def constrain(verdict: StageVerdict, results: tuple[ObligationResult, ...]) -> StageVerdict:
    """Refuse an `accept` that contradicts a failed obligation.

    The planner's judgment decides *whether the goal was met*; it does not get to
    overrule a deterministic check.
    """
    unmet = tuple(item.obligation_id for item in results if not item.passed)
    if unmet and verdict.decision is StageDecision.CONTINUE:
        # A continuation legitimately leaves obligations unmet: the work is not finished
        # yet. It is the one decision that may stand over a failing check.
        return verdict
    if unmet and verdict.decision is StageDecision.ACCEPT:
        return verdict.model_copy(
            update={
                "met": False,
                "unmet": unmet,
                "decision": StageDecision.RERUN,
                "rationale": (
                    f"planner accepted, but obligations are unmet: {', '.join(unmet)}"
                ),
            }
        )
    return verdict


@dataclass
class FakePlanner:
    """Scriptable planner so the supervisory loop is testable offline.

    It can be driven to skip domains, force reruns, and abort, which is what the loop's
    tests need.
    """

    route: RouteDecision | None = None
    stage_plans: dict[str, list[StagePlan]] = field(default_factory=dict)
    verdicts: dict[str, list[StageVerdict]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def queue_plan(self, stage: str, *plans: StagePlan) -> FakePlanner:
        self.stage_plans.setdefault(stage, []).extend(plans)
        return self

    def queue_verdict(self, stage: str, *verdicts: StageVerdict) -> FakePlanner:
        self.verdicts.setdefault(stage, []).extend(verdicts)
        return self

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def select_workflow(
        self, request: RequestContract, catalog: dict[str, dict[str, str]]
    ) -> tuple[RouteDecision, ModelUsage | None]:
        self.calls.append("select_workflow")
        if self.route is None:
            raise KeyError("FakePlanner has no queued route decision")
        return self.route, None

    def plan_stage(
        self, stage: StageSpec, request: RequestContract, stage_inputs: dict[str, Any]
    ) -> tuple[StagePlan, ModelUsage | None]:
        self.calls.append(f"plan:{stage.name}")
        queued = self.stage_plans.get(stage.name)
        if not queued:
            raise KeyError(f"FakePlanner has no queued stage plan for {stage.name}")
        return (queued.pop(0) if len(queued) > 1 else queued[0]), None

    def review_stage(
        self,
        stage: StageSpec,
        plan: StagePlan,
        obligations: tuple[ObligationResult, ...],
        summary: dict[str, Any],
        *,
        attempts_remaining: int,
    ) -> tuple[StageVerdict, ModelUsage | None]:
        self.calls.append(f"review:{stage.name}")
        queued = self.verdicts.get(stage.name)
        if queued:
            return (queued.pop(0) if len(queued) > 1 else queued[0]), None
        return default_verdict(obligations, attempts_remaining=attempts_remaining), None


# --------------------------------------------------------------------------------------
# The goal-driven planner (§6)
# --------------------------------------------------------------------------------------

GOAL_SIGNATURE = "planner.select_goal"
CAPABILITY_SIGNATURE = "planner.select_capability"
JUDGE_SIGNATURE = "planner.judge_gate"


class GoalChoice(Contract):
    goal_id: str
    rationale: str = ""


class CapabilityChoice(Contract):
    capability_id: str
    rationale: str = ""


class Judgment(Contract):
    """A gate's semantic half.

    `satisfied` answers the framework's stopping question: would more work here be likely
    to materially change the next decision? Not whether the work was thorough.
    """

    satisfied: bool
    rationale: str = ""
    missing: tuple[str, ...] = ()


@dataclass
class ModelGoalPlanner:
    """Chooses what to pursue next and who answers it; judges semantic gates.

    It does not reason about component calls. That decomposition belongs to the
    capability, and the planner is deliberately never shown enough to do it.
    """

    gateway: Gateway
    profile: ModelProfile
    prompts: PromptStore = field(default_factory=PromptStore)
    route_version: str = "1.0.0"
    goal_version: str = "1.0.0"
    capability_version: str = "1.0.0"
    judge_version: str = "1.1.0"
    #: Called with (role, usage, prompt_version). Goal, capability and gate calls return
    #: plain values through the protocol, so without this their cost is spent and never
    #: attributed to anything.
    usage_sink: Any = None

    def _spend(self, role: str, usage: ModelUsage | None, version: str) -> None:
        if self.usage_sink is not None:
            self.usage_sink(role, usage, version)

    def select_workflow(
        self, request: RequestContract, catalog: dict[str, dict[str, str]]
    ) -> tuple[RouteDecision, ModelUsage | None]:
        artifact = self.prompts.load(ROUTE_SIGNATURE, self.route_version)
        messages = render_prompt(artifact, prompt=request.prompt, workflows=catalog)
        return self.gateway.complete(self.profile, messages, RouteDecision)

    def select_goal(self, open_goals: Any, artifacts: list[dict[str, Any]]) -> str:
        artifact = self.prompts.load(GOAL_SIGNATURE, self.goal_version)
        messages = render_prompt(
            artifact,
            open_goals=[
                {"id": goal.id, "statement": goal.statement, "requires": list(goal.gate.requires)}
                for goal in open_goals
            ],
            artifacts=artifacts,
        )
        choice, usage = self.gateway.complete(self.profile, messages, GoalChoice)
        self._spend("planner.select_goal", usage, self.goal_version)
        return choice.goal_id

    def select_capability(self, goal: Any, candidates: list[str]) -> str:
        artifact = self.prompts.load(CAPABILITY_SIGNATURE, self.capability_version)
        messages = render_prompt(
            artifact, goal=goal.statement, candidates=candidates
        )
        choice, usage = self.gateway.complete(self.profile, messages, CapabilityChoice)
        self._spend("planner.select_capability", usage, self.capability_version)
        return choice.capability_id

    def judge(
        self,
        *,
        goal: Any,
        rubric: str,
        artifacts: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> tuple[bool, str]:
        artifact = self.prompts.load(JUDGE_SIGNATURE, self.judge_version)
        messages = render_prompt(
            artifact,
            goal=goal.statement,
            rubric=rubric,
            artifacts=artifacts,
            evidence=evidence,
        )
        judgment, usage = self.gateway.complete(self.profile, messages, Judgment)
        self._spend("planner.judge_gate", usage, self.judge_version)
        return judgment.satisfied, judgment.rationale


@dataclass
class ScriptedGoalPlanner:
    """Offline planner for the goal loop."""

    route: RouteDecision | None = None
    goal_order: list[str] = field(default_factory=list)
    capability_choice: dict[str, str] = field(default_factory=dict)
    judgments: list[tuple[bool, str]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def select_workflow(
        self, request: RequestContract, catalog: dict[str, dict[str, str]]
    ) -> tuple[RouteDecision, ModelUsage | None]:
        self.calls.append("select_workflow")
        if self.route is None:
            raise KeyError("ScriptedGoalPlanner has no queued route decision")
        return self.route, None

    def select_goal(self, open_goals: Any, artifacts: list[dict[str, Any]]) -> str:
        self.calls.append("select_goal")
        available = [goal.id for goal in open_goals]
        for preferred in self.goal_order:
            if preferred in available:
                return preferred
        return str(available[0])

    def select_capability(self, goal: Any, candidates: list[str]) -> str:
        self.calls.append("select_capability")
        return self.capability_choice.get(goal.id, candidates[0])

    def judge(
        self,
        *,
        goal: Any,
        rubric: str,
        artifacts: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> tuple[bool, str]:
        self.calls.append(f"judge:{goal.id}")
        if self.judgments:
            return self.judgments.pop(0) if len(self.judgments) > 1 else self.judgments[0]
        return True, "scripted: satisfied"
