"""The workflow-agnostic driver.

Nothing here knows about renaming.  A stage is executed, its obligations are checked, the
planner reviews it, and the attempt loop runs until the stage is accepted or the package's
`max_attempts` is spent.  The commit is the runtime's, never an agent's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .admission import AdmissionService
from .agent import DomainAgent
from .audit import AuditStore
from .compose import CompositionError, compose
from .compose import catalog as hydra_catalog
from .contracts import (
    ChangePlan,
    Composition,
    DomainSpec,
    ErrorCode,
    Obligation,
    ObligationResult,
    RequestContract,
    StageDecision,
    StagePlan,
    StageSpec,
    StageVerdict,
    content_digest,
    utc_now,
)
from .executor import Budget, Executor, InvocationResult
from .gateway import GatewayError
from .operators import mutation
from .operators.planning import CHECK_REQUIREMENTS
from .planner import Planner, constrain
from .prompts import PromptStore
from .registry import OperatorRegistry
from .stages import StageRegistry
from .telemetry import Tracer
from .verifier import Denial, Verifier
from .workflows import RegisteredWorkflow, WorkflowRegistry


class RuntimeError_(RuntimeError):
    """Raised when a run cannot continue.  Carries a closed error code for the SLIs."""

    def __init__(self, message: str, code: ErrorCode) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class StageOutcome:
    stage: StageSpec
    attempts: int
    verdict: StageVerdict
    obligations: tuple[ObligationResult, ...]
    results: dict[str, tuple[InvocationResult, ...]] = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    workflow_id: str
    outcome: str
    plan: ChangePlan | None = None
    committed_to: str | None = None
    stages: tuple[StageOutcome, ...] = ()
    error_code: ErrorCode | None = None
    detail: str = ""

    @property
    def committed(self) -> bool:
        return self.outcome == "committed"


@dataclass
class Runtime:
    """The single composition root."""

    operators: OperatorRegistry
    stages: StageRegistry
    workflows: WorkflowRegistry
    verifier: Verifier
    executor: Executor
    planner: Planner
    agents: dict[str, DomainAgent]
    audit: AuditStore
    workspace_root: Path
    tracer: Tracer | None = None
    prompts: PromptStore = field(default_factory=PromptStore)
    config_root: str | None = None
    #: Optional. Without it a subagent's operator request is recorded and refused rather
    #: than evaluated, which is the right default for an unattended run.
    admission: AdmissionService | None = None
    #: Operators admitted during this run, per domain. Populated only by a completed
    #: admission decision, never by an agent.
    grants: dict[str, set[str]] = field(default_factory=dict)

    # -- entry point --------------------------------------------------------------------

    def run(self, request: RequestContract, *, commit: bool = True) -> RunResult:
        run_id = uuid4().hex
        route, usage = self.planner.select_workflow(request, self.workflows.routing_catalog())
        if not route.supported or route.workflow_id not in self.workflows:
            return RunResult(
                run_id=run_id,
                workflow_id=route.workflow_id,
                outcome="unsupported",
                error_code=ErrorCode.COMPOSITION_INVALID,
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

        self._record_usage(run_id, "planner.route", usage)
        context: dict[str, Any] = {
            "run_id": run_id,
            "workflow_id": workflow.spec.id,
            "workflow_digest": workflow.digest(),
            "root": request.input_root,
            "input_root": request.input_root,
            "output_root": request.output_root,
            "staging_root": str(staging),
            "prompt": request.prompt,
        }

        outcomes: list[StageOutcome] = []
        try:
            for stage in workflow.stages:
                outcome = self._run_stage(
                    stage=stage,
                    request=request,
                    context=context,
                    workspace=workspace,
                    run_id=run_id,
                )
                outcomes.append(outcome)
                if outcome.verdict.decision is StageDecision.ACCEPT:
                    # Materialising a plan is a write, so the runtime does it — never a
                    # domain agent.  Doing it here means the review stage has a real tree
                    # to verify while the source is still untouched.
                    self._ensure_staged(
                        run_id=run_id,
                        context=context,
                        input_root=Path(request.input_root),
                        staging=staging,
                    )
                if outcome.verdict.decision is not StageDecision.ACCEPT:
                    mutation.discard(staging)
                    self.audit.record_run_outcome(
                        run_id=run_id,
                        outcome="aborted",
                        error_code=str(ErrorCode.ATTEMPTS_EXHAUSTED),
                    )
                    return RunResult(
                        run_id=run_id,
                        workflow_id=workflow.spec.id,
                        outcome="aborted",
                        stages=tuple(outcomes),
                        error_code=ErrorCode.ATTEMPTS_EXHAUSTED,
                        detail=f"stage {stage.name} was not accepted: {outcome.verdict.rationale}",
                    )
                if stage.name == workflow.spec.commit_after:
                    break
        except Denial as denial:
            mutation.discard(staging)
            denial_outcome = (
                "aborted" if denial.code is ErrorCode.ATTEMPTS_EXHAUSTED else "denied"
            )
            self.audit.record_run_outcome(
                run_id=run_id, outcome=denial_outcome, error_code=str(denial.code)
            )
            return RunResult(
                run_id=run_id,
                workflow_id=workflow.spec.id,
                outcome=denial_outcome,
                stages=tuple(outcomes),
                error_code=denial.code,
                detail=denial.reason,
            )

        plan = self._plan_from_context(context)
        if plan is not None:
            self.audit.record_plan(run_id=run_id, plan=plan)
        if not commit:
            self.audit.record_run_outcome(run_id=run_id, outcome="planned")
            return RunResult(
                run_id=run_id,
                workflow_id=workflow.spec.id,
                outcome="planned",
                plan=plan,
                stages=tuple(outcomes),
            )

        return self._commit(
            run_id=run_id,
            workflow=workflow,
            plan=plan,
            staging=staging,
            output_root=Path(request.output_root),
            outcomes=tuple(outcomes),
        )

    # -- the attempt loop ---------------------------------------------------------------

    def _run_stage(
        self,
        *,
        stage: StageSpec,
        request: RequestContract,
        context: dict[str, Any],
        workspace: Path,
        run_id: str,
    ) -> StageOutcome:
        previous_plan: StagePlan | None = None
        outcome: StageOutcome | None = None

        for attempt in range(1, stage.max_attempts + 1):
            started_at = utc_now().isoformat()
            attempts_remaining = stage.max_attempts - attempt

            # The budget is resolved before planning so a stage plan is itself charged
            # against the stage's model allowance.
            budget = Budget(envelope=stage.budget, items=_item_count(context))
            plan, usage = self.planner.plan_stage(stage, request, context)
            self._record_usage(run_id, "planner.stage_plan", usage, budget=budget)
            self.verifier.verify_stage_plan(plan, stage)
            if previous_plan is not None:
                self.verifier.verify_rerun(previous_plan, plan)
            previous_plan = plan

            attempt_context = dict(context)
            compositions: list[tuple[Composition, list[dict[str, Any]]]] = []
            results: dict[str, tuple[InvocationResult, ...]] = {}

            denial: Denial | None = None
            for goal in plan.activated:
                domain = stage.domain(goal.domain_id)
                agent = self.agents.get(domain.id) or self.agents["*"]
                try:
                    composition, usage = agent.compose(
                        domain=domain,
                        goal=goal,
                        stage_inputs=attempt_context,
                        catalog_summary=_catalog_summary(
                            domain.config_groups, self.config_root
                        ),
                    )
                except GatewayError as failure:
                    self._record_usage(
                        run_id,
                        f"domain.{domain.id}",
                        failure.usage,
                        prompt_version=domain.prompt_version,
                    )
                    # A model that returns a response violating its contract is exactly
                    # the recoverable case the attempt loop exists for. Killing the run
                    # here would also lose the reason.
                    denial = Denial(
                        ErrorCode.COMPOSITION_INVALID,
                        f"{domain.id} returned no usable composition: {failure}",
                    )
                    results[domain.id] = ()
                    break
                self._record_usage(
                    run_id,
                    f"domain.{domain.id}",
                    usage,
                    budget=budget,
                    prompt_version=domain.prompt_version,
                )
                try:
                    config = compose(
                        _selections(composition),
                        allowed_groups=domain.config_groups,
                        config_root=self.config_root,
                    )
                    produced = self.executor.execute(
                        composition,
                        domain,
                        stage_inputs=attempt_context,
                        config=config,
                        workspace=workspace,
                        budget=budget,
                        stage=stage.name,
                        attempt=attempt,
                        granted=frozenset(self.grants.get(domain.id, set())),
                    )
                except (Denial, CompositionError) as refusal:
                    # A refusal is evidence. Journalling the composition that caused it is
                    # the only way anyone can see what the agent actually asked for, and an
                    # invalid composition is exactly the recoverable case the attempt loop
                    # exists for — so the planner reviews it rather than the run dying here.
                    denial = (
                        refusal
                        if isinstance(refusal, Denial)
                        else Denial(ErrorCode.COMPOSITION_INVALID, str(refusal))
                    )
                    compositions.append((composition, []))
                    results[domain.id] = ()
                    break

                results[domain.id] = produced
                if composition.ask is not None:
                    self._handle_ask(composition.ask, domain=domain, run_id=run_id)
                compositions.append((composition, [item.journal_row() for item in produced]))
                for item in produced:
                    if item.output:
                        attempt_context.update(item.output)

            obligations = tuple(
                Obligation(id=name, description=name, checker=name) for name in stage.obligations
            )
            payloads = {
                obligation.id: _payload_for(obligation.checker, attempt_context)
                for obligation in obligations
            }
            checked = self.verifier.check_obligations(
                obligations, {k: v for k, v in payloads.items() if v is not None}
            )

            if denial is not None and denial.code is not ErrorCode.COMPOSITION_INVALID:
                # Budget or approval failures are not recoverable by rerunning: the same
                # attempt would hit the same wall.
                self._journal_attempt(
                    run_id=run_id, stage=stage, attempt=attempt, started_at=started_at,
                    plan=plan, compositions=compositions, checked=checked,
                    verdict=StageVerdict(
                        met=False,
                        decision=StageDecision.ABORT,
                        rationale=denial.reason,
                    ),
                    error_code=str(denial.code),
                )
                raise denial

            verdict, usage = self.planner.review_stage(
                stage,
                plan,
                checked,
                _summary(results, denial),
                attempts_remaining=attempts_remaining,
            )
            self._record_usage(run_id, "planner.stage_review", usage, budget=budget)
            verdict = constrain(verdict, checked)
            if denial is not None and verdict.decision is StageDecision.ACCEPT:
                verdict = verdict.model_copy(
                    update={
                        "met": False,
                        "decision": StageDecision.RERUN
                        if attempts_remaining
                        else StageDecision.ABORT,
                        "rationale": f"composition refused: {denial.reason}",
                    }
                )
            if verdict.decision is StageDecision.RERUN and attempts_remaining == 0:
                verdict = verdict.model_copy(
                    update={
                        "decision": StageDecision.ABORT,
                        "rationale": (
                            f"{verdict.rationale} (max_attempts={stage.max_attempts} spent)"
                        ),
                    }
                )

            self._journal_attempt(
                run_id=run_id, stage=stage, attempt=attempt, started_at=started_at,
                plan=plan, compositions=compositions, checked=checked, verdict=verdict,
                error_code=(
                    str(denial.code)
                    if denial is not None
                    else None
                    if verdict.met
                    else str(ErrorCode.OBLIGATION_FAILED)
                ),
            )

            outcome = StageOutcome(
                stage=stage,
                attempts=attempt,
                verdict=verdict,
                obligations=checked,
                results=results,
            )
            if verdict.decision is StageDecision.ACCEPT:
                context.clear()
                context.update(attempt_context)
                return outcome
            if verdict.decision is StageDecision.ABORT:
                return outcome

        assert outcome is not None
        return outcome

    def _record_usage(
        self,
        run_id: str,
        role: str,
        usage: Any,
        *,
        budget: Budget | None = None,
        prompt_version: str | None = None,
    ) -> None:
        """Record and charge a model call.

        Without this `journal costs` reports zero for a real run and the model-invocation
        budget is never charged, so both the bill and the meter read empty while money is
        actually being spent.
        """
        if usage is None:
            return
        self.audit.record_model_invocation(
            run_id=run_id,
            role=role,
            profile_id=getattr(usage, "profile_id", role),
            requested_model=usage.requested_model,
            resolved_model=usage.resolved_model,
            provider=usage.provider,
            prompt_version=prompt_version,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
        )
        if budget is not None:
            budget.consume_model_invocation(
                tokens=usage.total_tokens, cost_usd=usage.cost_usd
            )

    def _journal_attempt(
        self,
        *,
        run_id: str,
        stage: StageSpec,
        attempt: int,
        started_at: str,
        plan: StagePlan,
        compositions: list[tuple[Composition, list[dict[str, Any]]]],
        checked: tuple[ObligationResult, ...],
        verdict: StageVerdict,
        error_code: str | None,
    ) -> None:
        self.audit.record_attempt(
            run_id=run_id,
            stage_name=stage.name,
            stage_version=stage.version,
            attempt_no=attempt,
            started_at=started_at,
            plan=plan,
            compositions=compositions,
            obligations=checked,
            verdict=verdict,
            error_code=error_code,
        )

    def _handle_ask(self, ask: Any, *, domain: DomainSpec, run_id: str) -> None:
        """Evaluate a subagent's operator request after its composition has run.

        Deliberately after: a subagent never observes its own results, so an admitted
        operator becomes usable on the next attempt rather than mid-composition. That
        keeps the compose-once rule intact while still letting a run recover.
        """
        from .contracts import AdmissionChoice, AdmissionDisposition, DecidedBy, OperatorRequest

        request = OperatorRequest(
            request_id=uuid4().hex,
            run_id=run_id,
            domain_id=domain.id,
            kind=ask.kind,
            family=ask.family,
            name=ask.name,
            features=ask.features,
            dependencies=ask.dependencies,
            declared_side_effects=ask.declared_side_effects,
            rationale=ask.rationale,
        )
        if self.admission is None:
            # Recording it still matters: the request is evidence of a gap in the
            # stage package, which is a human's problem to look at later.
            self.audit.record_operator_request(request)
            return

        report, candidate = self.admission.evaluate(request)
        if report.disposition is not AdmissionDisposition.AUTO_ADMIT:
            # Escalated. The run continues without the operator; a person decides later
            # through `shakespeare requests`.
            return

        self.admission.decide(
            report,
            candidate,
            decided_by=DecidedBy.PLANNER,
            choice=AdmissionChoice.APPROVE,
            rationale="auto-admitted: low-risk declarative variant",
        )
        self.grants.setdefault(domain.id, set()).add(candidate.spec.name)

    # -- commit -------------------------------------------------------------------------

    def _commit(
        self,
        *,
        run_id: str,
        workflow: RegisteredWorkflow,
        plan: ChangePlan | None,
        staging: Path,
        output_root: Path,
        outcomes: tuple[StageOutcome, ...],
    ) -> RunResult:
        if plan is None:
            self.audit.record_run_outcome(
                run_id=run_id,
                outcome="aborted",
                error_code=str(ErrorCode.COMMIT_VERIFICATION_FAILED),
            )
            return RunResult(
                run_id=run_id,
                workflow_id=workflow.spec.id,
                outcome="aborted",
                stages=outcomes,
                error_code=ErrorCode.COMMIT_VERIFICATION_FAILED,
                detail="the workflow produced no change plan",
            )

        existing = self.audit.find_commit(
            plan_digest=plan.fingerprint(), output_root=str(output_root)
        )
        if existing is not None and output_root.exists():
            # Idempotency receipt: this exact plan has already been committed here, so
            # re-applying it is a no-op rather than a collision with the output root.
            mutation.discard(staging)
            self.audit.record_run_outcome(run_id=run_id, outcome="already_committed")
            return RunResult(
                run_id=run_id,
                workflow_id=workflow.spec.id,
                outcome="committed",
                plan=plan,
                committed_to=str(output_root),
                stages=outcomes,
                detail=f"already committed by run {existing['run_id']}; nothing to do",
            )

        report = mutation.verify_tree(plan=plan, staging_root=staging)
        if not report["ok"]:
            mutation.discard(staging)
            self.audit.record_run_outcome(
                run_id=run_id,
                outcome="aborted",
                error_code=str(ErrorCode.COMMIT_VERIFICATION_FAILED),
            )
            return RunResult(
                run_id=run_id,
                workflow_id=workflow.spec.id,
                outcome="aborted",
                plan=plan,
                stages=outcomes,
                error_code=ErrorCode.COMMIT_VERIFICATION_FAILED,
                detail=f"staging does not match the plan: {report}",
            )

        staging_digest = content_digest(report)
        record = mutation.commit(staging_root=staging, output_root=output_root)
        self.audit.record_commit(
            run_id=run_id,
            plan=plan,
            staging_digest=staging_digest,
            output_root=str(output_root),
        )
        self.audit.record_mutation(
            run_id=run_id,
            target_ref=str(output_root),
            operation="commit",
            reversal=record,
            after_digest=staging_digest,
        )
        self.audit.record_run_outcome(run_id=run_id, outcome="committed")
        return RunResult(
            run_id=run_id,
            workflow_id=workflow.spec.id,
            outcome="committed",
            plan=plan,
            committed_to=str(output_root),
            stages=outcomes,
        )

    def _ensure_staged(
        self, *, run_id: str, context: dict[str, Any], input_root: Path, staging: Path
    ) -> None:
        if context.get("_staged") or "plan" not in context:
            return
        plan = self._plan_from_context(context)
        if plan is None:
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
        context["staged_files"] = len(reversals)

    def _plan_from_context(self, context: dict[str, Any]) -> ChangePlan | None:
        payload = context.get("plan")
        if payload is None:
            return None
        # No workflow type is imported here: ChangeEntry carries a subclass's extra fields
        # through validation, so the driver stays ignorant of what a plan entry means.
        return ChangePlan.model_validate(payload)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _item_count(context: dict[str, Any]) -> int:
    value = context.get("items")
    return len(value) if isinstance(value, list) else int(context.get("count", 0) or 0)


def _selections(composition: Composition) -> dict[str, str]:
    merged: dict[str, str] = {}
    for invocation in composition.invocations:
        merged.update(invocation.selections)
    return merged


def _catalog_summary(groups: frozenset[str], config_root: str | None) -> dict[str, list[str]]:
    available = hydra_catalog(config_root)
    return {group: sorted(available[group]) for group in sorted(groups) if group in available}


def _payload_for(checker: str, context: dict[str, Any]) -> dict[str, Any] | None:
    required = CHECK_REQUIREMENTS.get(checker, ())
    payload = {key: context[key] for key in required if key in context}
    return payload if len(payload) == len(required) else None


def _summary(
    results: dict[str, tuple[InvocationResult, ...]], denial: Denial | None = None
) -> dict[str, Any]:
    """What the planner sees about an attempt: shape and outcomes, never content."""
    summary: dict[str, Any] = {}
    if denial is not None:
        # The planner needs the refusal in words to revise a goal usefully.
        summary["refusal"] = {"code": str(denial.code), "reason": denial.reason}
    summary["domains"] = {
        domain: {
            "invocations": len(items),
            "succeeded": sum(1 for item in items if item.succeeded),
            # The detail, not just the code. A planner told only "operator_failed" cannot
            # revise a goal usefully, and its rerun repeats the same class of mistake.
            "failures": [
                {
                    "operator": item.operator,
                    "code": str(item.error_code),
                    "detail": (item.error_detail or "")[:400],
                }
                for item in items
                if not item.succeeded
            ],
        }
        for domain, items in results.items()
    }
    return summary
