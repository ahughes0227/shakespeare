"""Deterministic authorization.

Nothing a model produces reaches an operator without passing through here.  The verifier
checks *surfaces* — which operators, which config groups, which domains, how many calls —
and *obligations* — what must be true at a stage's output.  It never checks the route an
agent took to get there; that is deliberately none of its business.
"""

from __future__ import annotations

from dataclasses import dataclass

from .compose import CompositionError, validate_parameters, validate_selections
from .contracts import (
    Composition,
    DomainSpec,
    ErrorCode,
    Obligation,
    ObligationResult,
    StagePlan,
    StageSpec,
)
from .operators.builtin import RUNTIME_ONLY
from .operators.planning import run_check
from .registry import OperatorRegistry


@dataclass(frozen=True)
class Denial(Exception):
    """A refusal is evidence, so it carries a closed error code for the SLIs."""

    code: ErrorCode
    reason: str

    def __str__(self) -> str:
        return f"{self.code}: {self.reason}"


class Verifier:
    def __init__(self, registry: OperatorRegistry, *, config_root: str | None = None) -> None:
        self.registry = registry
        self.config_root = config_root

    # -- stage plans --------------------------------------------------------------------

    def verify_stage_plan(self, plan: StagePlan, stage: StageSpec) -> None:
        """Check the planner's activation choices against the stage package.

        The package bounds what may be skipped, not the planner: a safety domain declared
        non-skippable cannot be planned away however the planner justifies it.
        """
        declared = {domain.id for domain in stage.domains}
        named = {goal.domain_id for goal in plan.activated} | {
            skip.domain_id for skip in plan.skipped
        }

        unknown = named - declared
        if unknown:
            raise Denial(
                ErrorCode.COMPOSITION_INVALID,
                f"stage plan names domains not in {stage.name}: {sorted(unknown)}",
            )

        unaccounted = declared - named
        if unaccounted:
            raise Denial(
                ErrorCode.COMPOSITION_INVALID,
                f"every domain must be activated or skipped; missing: {sorted(unaccounted)}",
            )

        for skip in plan.skipped:
            if not stage.domain(skip.domain_id).skippable:
                raise Denial(
                    ErrorCode.COMPOSITION_INVALID,
                    f"domain {skip.domain_id} is not skippable in stage {stage.name}",
                )

    def verify_rerun(self, previous: StagePlan, revised: StagePlan) -> None:
        """A rerun must actually change something, or the attempt loop cannot converge."""
        if previous.digest() == revised.digest():
            raise Denial(
                ErrorCode.ATTEMPTS_EXHAUSTED,
                "rerun repeats the previous stage plan verbatim and would not progress",
            )

    # -- compositions -------------------------------------------------------------------

    def verify_composition(
        self,
        composition: Composition,
        domain: DomainSpec,
        *,
        operator_call_budget: int | None = None,
        granted: frozenset[str] = frozenset(),
    ) -> None:
        """`granted` holds operators admitted during this run for this domain.

        The surface is widened only by a completed admission — computed risk, passing
        test tiers, a reproducible render — never by an agent asking nicely.
        """
        if composition.domain_id != domain.id:
            raise Denial(
                ErrorCode.COMPOSITION_INVALID,
                f"composition claims domain {composition.domain_id}, issued to {domain.id}",
            )

        if operator_call_budget is not None and len(composition.invocations) > operator_call_budget:
            raise Denial(
                ErrorCode.BUDGET_EXHAUSTED,
                f"composition requests {len(composition.invocations)} operator calls,"
                f" budget allows {operator_call_budget}",
            )

        for invocation in composition.invocations:
            name = invocation.operator

            if name not in self.registry:
                raise Denial(ErrorCode.COMPOSITION_INVALID, f"unknown operator: {name}")

            # The domain catalog comes from the stage package, never from the goal text,
            # so a persuasive goal cannot widen what an agent may call.
            if name not in domain.catalog and name not in granted:
                raise Denial(
                    ErrorCode.COMPOSITION_INVALID,
                    f"operator {name} is outside the catalog for domain {domain.id}",
                )

            if name in RUNTIME_ONLY:
                raise Denial(
                    ErrorCode.COMPOSITION_INVALID,
                    f"{name} writes and is reserved to the runtime; agents plan, "
                    f"the runtime commits",
                )

            try:
                validate_selections(
                    invocation.selections,
                    allowed_groups=domain.config_groups,
                    config_root=self.config_root,
                )
                validate_parameters(invocation.parameters)
            except CompositionError as exc:
                raise Denial(ErrorCode.COMPOSITION_INVALID, str(exc)) from exc

    # -- obligations --------------------------------------------------------------------

    def check_obligations(
        self, obligations: tuple[Obligation, ...], payloads: dict[str, dict[str, object]]
    ) -> tuple[ObligationResult, ...]:
        """Run each obligation's deterministic checker.

        A missing payload fails closed: an unevaluated obligation is not a satisfied one.
        """
        results: list[ObligationResult] = []
        for obligation in obligations:
            payload = payloads.get(obligation.id)
            if payload is None:
                results.append(
                    ObligationResult(
                        obligation_id=obligation.id,
                        passed=False,
                        detail={"error": "no evidence was produced for this obligation"},
                    )
                )
                continue
            results.append(
                run_check(obligation.id, obligation.checker, {**obligation.parameters, **payload})
            )
        return tuple(results)


def unmet(results: tuple[ObligationResult, ...]) -> tuple[str, ...]:
    return tuple(item.obligation_id for item in results if not item.passed)
