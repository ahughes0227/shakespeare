"""The single composition root and the run entry point.

The control model is the framework's (§19): the planner asks what it needs next and who
can answer it, a capability plans within its bounded domain, components execute
deterministically, artifacts are produced, and a gate decides whether the goal is
satisfied. There is no spine to walk.

Everything transactional is Shakespeare's own and unchanged by that: staging, balanced
accounting, two-phase commit, journalled reversal, and replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..capabilities import CapabilityRegistry, CapabilityRunner
from ..components.registry import OperatorRegistry
from ..contracts import (
    Bound,
    BudgetEnvelope,
    ChangePlan,
    ErrorCode,
    Measurement,
    MeasurementKind,
    RequestContract,
    content_digest,
)
from ..domain import mutation
from ..prompts import PromptStore
from ..workflows import RegisteredWorkflow, WorkflowRegistry
from .artifacts import ArtifactStore
from .audit import AuditStore
from .control import Controller, GoalAttempt, commit_if_verified, new_run_id
from .executor import Budget, Executor
from .goals import Goal
from .telemetry import Tracer
from .verifier import Denial, Verifier


class RuntimeError_(RuntimeError):
    def __init__(self, message: str, code: ErrorCode) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class RunResult:
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


class _NoSpan:
    """Stands in when no tracer is configured, so call sites need no branch."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _no_span() -> _NoSpan:
    return _NoSpan()


@dataclass
class Runtime:
    operators: OperatorRegistry
    capabilities: CapabilityRegistry
    workflows: WorkflowRegistry
    verifier: Verifier
    executor: Executor
    planner: Any
    agents: dict[str, Any]
    audit: AuditStore
    workspace_root: Path
    tracer: Tracer | None = None
    prompts: PromptStore = field(default_factory=PromptStore)
    config_root: str | None = None
    admission: Any = None
    grants: dict[str, set[str]] = field(default_factory=dict)
    #: The tracer bound to the run in flight, so a model call joins the tree at the point
    #: it was made rather than dangling at the root.
    _tracer: Any = None
    #: Per-capability budget. Resolved against the item count when a capability is asked
    #: to do something, since file counts are unbounded.
    budget: BudgetEnvelope = field(
        default_factory=lambda: BudgetEnvelope.model_validate(
            {
                "operator_calls": "40 + 6*n",
                "model_invocations": "20 + 2*n",
                "wall_time_seconds": 5400,
            }
        )
    )

    def run(self, request: RequestContract, *, commit: bool = True) -> RunResult:
        run_id = new_run_id()
        # Bind the tracer before the first model call, or routing is spent untraced.
        self._tracer = self.tracer.rebind(run_id) if self.tracer else None
        if hasattr(self.planner, "usage_sink"):
            self.planner.usage_sink = lambda role, usage, version: self._record_usage(
                run_id, role, usage, prompt_version=version
            )
        route, usage = self.planner.select_workflow(request, self.workflows.routing_catalog())
        if not route.supported or route.workflow_id not in self.workflows:
            # The router has just worked out what serving this would take. Printing that
            # and exiting throws away the one piece of analysis a person needs to close
            # the gap, so it is recorded the way an unmet operator request has been since
            # admission was written.
            self.audit.record_capability_gap(
                run_id=run_id,
                prompt_digest=content_digest(request.prompt),
                rationale=route.rationale,
                requires=route.requires,
            )
            return RunResult(
                run_id=run_id,
                workflow_id=route.workflow_id,
                outcome="unsupported",
                error_code=ErrorCode.UNSUPPORTED,
                detail=route.rationale or "no registered workflow handles this request",
            )

        workflow = self.workflows.get(route.workflow_id)
        workspace = self.workspace_root / run_id
        staging = workspace / "staging"
        workspace.mkdir(parents=True, exist_ok=True)

        self.audit.record_run(
            run_id=run_id,
            workflow_id=workflow.spec.id,
            workflow_version=workflow.spec.version,
            workflow_digest=workflow.digest(),
            request_digest=request.digest(),
            input_root_digest=content_digest(request.input_root),
        )
        self._record_usage(
            run_id,
            "planner.route",
            usage,
            prompt_version=getattr(self.planner, "route_version", None),
        )

        artifacts = ArtifactStore(root=workspace / "artifacts", run_id=run_id)
        # One tracer per run, so every span carries the run it belongs to.
        tracer = self._tracer
        controller = Controller(
            capabilities=self.capabilities,
            runner=CapabilityRunner(
                executor=self.executor,
                agents=self.agents,
                artifacts=artifacts,
                config_root=self.config_root,
                usage_sink=lambda role, usage, version: self._record_usage(
                    run_id, role, usage, prompt_version=version
                ),
                ask_sink=lambda ask, capability_id: self._handle_ask(
                    ask, capability_id=capability_id, run_id=run_id
                ),
                grants=self.grants,
                tracer=tracer,
                # The ceiling scheduling divides against is the model's own.
                capacity=getattr(
                    getattr(self.planner, "profile", None), "max_output_tokens", 16384
                ),
            ),
            artifacts=artifacts,
            audit=self.audit,
            planner=self.planner,
            workspace=workspace,
            tracer=tracer,
            # A goal that asks whether the staged tree matches the plan cannot be
            # satisfied if nothing is staged until the loop has finished. Staging is the
            # first phase of the commit, so it happens the moment there is a plan to
            # stage — and stays the runtime's, never a capability's.
            on_goal_satisfied=lambda context: self._stage_when_planned(
                run_id=run_id,
                context=context,
                input_root=Path(request.input_root),
                staging=staging,
            ),
            context={
                "run_id": run_id,
                "workflow_id": workflow.spec.id,
                "workflow_digest": workflow.digest(),
                "root": request.input_root,
                "input_root": request.input_root,
                "output_root": request.output_root,
                "staging_root": str(staging),
                "prompt": request.prompt,
            },
        )

        run_span = (
            tracer.span("run", digests={"workflow": workflow.digest()})
            if tracer
            else _no_span()
        )
        try:
            with run_span as span:
                attempts, satisfied, failure = controller.pursue(
                    graph=workflow.spec.graph,
                    request=request,
                    run_id=run_id,
                    budget_for=self._budget_for,
                )
                if span is not None:
                    span.add_count("goals_satisfied", len(satisfied))
                    span.add_count("goals_total", len(workflow.spec.goals))
                    span.record(outcome="planned" if failure is None else "aborted")
        except Denial as denial:
            mutation.discard(staging)
            self.audit.record_run_outcome(
                run_id=run_id, outcome="denied", error_code=str(denial.code)
            )
            return RunResult(
                run_id=run_id,
                workflow_id=workflow.spec.id,
                outcome="denied",
                error_code=denial.code,
                detail=denial.reason,
            )

        self._journal(run_id, workflow, attempts)
        self._remember(run_id, attempts)

        if failure is not None:
            mutation.discard(staging)
            # An impediment is not an abort. Aborting says the attempts did not get there;
            # escalating says no attempt of this shape would, and the difference decides
            # whether a person needs to look at it or a retry might have worked.
            raised = failure.startswith("impediment")
            outcome = "escalated" if raised else "aborted"
            code = ErrorCode.IMPEDIMENT if raised else ErrorCode.OBLIGATION_FAILED
            self.audit.record_run_outcome(
                run_id=run_id, outcome=outcome, error_code=str(code)
            )
            return RunResult(
                run_id=run_id,
                workflow_id=workflow.spec.id,
                outcome=outcome,
                attempts=attempts,
                satisfied=satisfied,
                error_code=code,
                detail=failure,
            )

        plan = self._plan_from(controller.context)
        if plan is not None:
            self._stage_when_planned(
                run_id=run_id,
                context=controller.context,
                input_root=Path(request.input_root),
                staging=staging,
            )

        if not commit:
            return RunResult(
                run_id=run_id,
                workflow_id=workflow.spec.id,
                outcome="planned",
                plan=plan,
                attempts=attempts,
                satisfied=satisfied,
                planned_output_root=request.output_root,
            )

        outcome, detail = commit_if_verified(
            plan=plan,
            staging=staging,
            output_root=Path(request.output_root),
            audit=self.audit,
            run_id=run_id,
        )
        self.audit.record_run_outcome(
            run_id=run_id,
            outcome=outcome,
            error_code=None if outcome == "committed" else str(
                ErrorCode.COMMIT_VERIFICATION_FAILED
            ),
        )
        return RunResult(
            run_id=run_id,
            workflow_id=workflow.spec.id,
            outcome=outcome,
            plan=plan,
            committed_to=str(request.output_root) if outcome == "committed" else None,
            attempts=attempts,
            satisfied=satisfied,
            detail=detail,
        )

    def commit_planned(self, result: RunResult) -> RunResult:
        """Commit the plan a previous planning run produced — that plan, not a new one."""
        if result.plan is None:
            raise RuntimeError_(
                "there is no plan to commit", ErrorCode.COMMIT_VERIFICATION_FAILED
            )
        workspace = self.workspace_root / result.run_id
        outcome, detail = commit_if_verified(
            plan=result.plan,
            staging=workspace / "staging",
            output_root=Path(result.planned_output_root or ""),
            audit=self.audit,
            run_id=result.run_id,
        )
        self.audit.record_run_outcome(
            run_id=result.run_id,
            outcome=outcome,
            error_code=None if outcome == "committed" else str(
                ErrorCode.COMMIT_VERIFICATION_FAILED
            ),
        )
        return RunResult(
            run_id=result.run_id,
            workflow_id=result.workflow_id,
            outcome=outcome,
            plan=result.plan,
            committed_to=result.planned_output_root if outcome == "committed" else None,
            attempts=result.attempts,
            satisfied=result.satisfied,
            detail=detail,
        )

    def _handle_ask(self, ask: Any, *, capability_id: str, run_id: str) -> str | None:
        """Evaluate a capability's component request once its round has run.

        Deliberately after: an admitted component becomes usable on the next round rather
        than mid-organization, which keeps a round a single coherent unit of work.
        """
        from ..contracts import (
            AdmissionChoice,
            AdmissionDisposition,
            DecidedBy,
            OperatorRequest,
        )

        request = OperatorRequest(
            request_id=uuid4().hex,
            run_id=run_id,
            domain_id=capability_id,
            kind=ask.kind,
            family=ask.family,
            name=ask.name,
            features=ask.features,
            dependencies=ask.dependencies,
            declared_side_effects=ask.declared_side_effects,
            rationale=ask.rationale,
        )
        if self.admission is None:
            # Still recorded: the request is evidence of a gap in the capability package.
            self.audit.record_operator_request(request)
            return None

        report, candidate = self.admission.evaluate(request)
        if report.disposition is not AdmissionDisposition.AUTO_ADMIT:
            return None
        self.admission.decide(
            report,
            candidate,
            decided_by=DecidedBy.PLANNER,
            choice=AdmissionChoice.APPROVE,
            rationale="auto-admitted: low-risk declarative variant",
        )
        self.grants.setdefault(capability_id, set()).add(candidate.spec.name)
        return str(candidate.spec.name)

    # -- helpers -----------------------------------------------------------------------

    def _budget_for(self, goal: Goal, context: dict[str, Any]) -> Budget:
        items = context.get("items")
        return Budget(envelope=self.budget, items=len(items) if isinstance(items, list) else 0)

    def _plan_from(self, context: dict[str, Any]) -> ChangePlan | None:
        payload = context.get("plan")
        return ChangePlan.model_validate(payload) if payload is not None else None

    def _stage_when_planned(
        self,
        *,
        run_id: str,
        context: dict[str, Any],
        input_root: Path,
        staging: Path,
    ) -> None:
        """Record and stage the plan, once, as soon as one exists.

        Called after every satisfied goal rather than at a named point, so the loop stays
        ignorant of which goal produces a plan and which one reviews it.
        """
        if context.get("_staged"):
            return
        plan = self._plan_from(context)
        if plan is None:
            return
        self.audit.record_plan(run_id=run_id, plan=plan)
        self._ensure_staged(
            run_id=run_id,
            context=context,
            plan=plan,
            input_root=input_root,
            staging=staging,
        )

    def _ensure_staged(
        self,
        *,
        run_id: str,
        context: dict[str, Any],
        plan: ChangePlan,
        input_root: Path,
        staging: Path,
    ) -> None:
        """Materialising a plan is a write, so the runtime does it — never a capability."""
        if context.get("_staged"):
            return
        reversals = mutation.stage_plan(
            plan=plan, input_root=input_root, staging_root=staging
        )
        for record in reversals:
            self.audit.record_mutation(
                run_id=run_id,
                target_ref=str(record.payload.get("target", "")),
                operation=record.operation,
                reversal=record,
                after_digest=record.payload.get("after_digest"),
            )
        context["_staged"] = True

    def _journal(
        self, run_id: str, workflow: RegisteredWorkflow, attempts: tuple[GoalAttempt, ...]
    ) -> None:
        """Record each goal attempt as a unit of work, with its gate result."""
        from ..contracts import (
            Composition,
            DomainGoal,
            ObligationResult,
            StageDecision,
            StagePlan,
            StageVerdict,
        )

        for index, attempt in enumerate(attempts, start=1):
            goal = workflow.spec.graph.goal(attempt.goal_id)
            # The runtime's own scheduling calls come first: they are why the capability
            # was asked what it was asked, and without them the journal shows the
            # batches but not the decision that produced them.
            compositions = [
                (composition, [item.journal_row() for item in results])
                for composition, results in attempt.outcome.scheduling
            ] + [
                (
                    Composition(
                        domain_id=attempt.capability,
                        invocations=round_.organization.invocations,
                        rationale=round_.organization.intent,
                    ),
                    [item.journal_row() for item in round_.results],
                )
                for round_ in attempt.outcome.rounds
            ]
            self.audit.record_attempt(
                run_id=run_id,
                stage_name=goal.id,
                stage_version=workflow.spec.version,
                attempt_no=index,
                started_at="",
                plan=StagePlan(
                    activated=(
                        DomainGoal(
                            domain_id=attempt.capability,
                            goal=goal.statement,
                            success_criterion=goal.gate.id,
                        ),
                    )
                ),
                compositions=compositions,
                obligations=(
                    ObligationResult(
                        obligation_id=attempt.gate.gate_id,
                        passed=attempt.gate.satisfied,
                        detail={"outcome": str(attempt.gate.outcome)},
                    ),
                ),
                verdict=StageVerdict(
                    met=attempt.gate.satisfied,
                    unmet=attempt.gate.failed_checks + attempt.gate.missing_kinds,
                    decision=(
                        StageDecision.ACCEPT if attempt.gate.satisfied else StageDecision.RERUN
                    ),
                    rationale=attempt.gate.rationale,
                ),
                error_code=None if attempt.gate.satisfied else str(ErrorCode.OBLIGATION_FAILED),
            )

    def _remember(self, run_id: str, attempts: tuple[GoalAttempt, ...]) -> None:
        """Record what this run measured about its own cost.

        Every run has always measured this and every run has always thrown it away, so the
        estimate each one starts from is the constant somebody typed into a manifest. The
        rows go to the ledger and nowhere else: nothing reads them during a run, and a
        measured constant reaches a run only by being written into the manifest that
        declares it. That is what keeps what a run does fixed and digested at its start.

        Recorded after the work, never during it, so a measurement is only ever a
        statement about something that already happened.
        """
        measurements: list[Measurement] = []
        for attempt in attempts:
            capability = self.capabilities.get(attempt.capability)
            for observation in attempt.outcome.observations:
                model = str(observation.get("resolved_model") or "")
                spent = int(observation.get("completion_tokens") or 0)
                weight = float(observation.get("weight") or 0)
                count = int(observation.get("items") or 0)
                # No model, no spend, or no material means nothing was measured. A fake
                # agent reports no usage, and recording its batches as costing zero would
                # put the offline suite's arithmetic into the evidence for a live one.
                if not model or spent <= 0 or weight <= 0 or count <= 0:
                    continue
                measurements.append(
                    Measurement(
                        kind=MeasurementKind.SCHEDULE_COST,
                        subject=capability.ref,
                        resolved_model=model,
                        prompt_version=capability.prompt_version,
                        value=float(spent),
                        weight=weight,
                        count=count,
                        outcome=not observation.get("failed"),
                        # A batch that was cut off never reported what it would have
                        # cost; it proved the cost is at least what fitted.
                        bound=Bound.LOWER if observation.get("truncated") else None,
                    )
                )
        self.audit.record_measurements(run_id=run_id, measurements=measurements)

    def _record_usage(
        self, run_id: str, role: str, usage: Any, *, prompt_version: str | None = None
    ) -> None:
        if usage is None:
            return
        if self._tracer is not None:
            with self._tracer.span(
                f"model.{role}",
                domain=role,
                prompt_version=prompt_version,
                requested_model=usage.requested_model,
                resolved_model=usage.resolved_model,
                provider=usage.provider,
                cost_usd=usage.cost_usd,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            ):
                pass
        self.audit.record_model_invocation(
            run_id=run_id,
            role=role,
            profile_id=role,
            requested_model=usage.requested_model,
            resolved_model=usage.resolved_model,
            provider=usage.provider,
            prompt_version=prompt_version,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
        )
