"""Execution of validated compositions, with budget metering and journaling.

The agent that produced a composition never sees what happens here.  Results — including
failures — become stage output, which the planner reads at the stage boundary.  That is
the whole reason a domain subagent cannot adapt mid-stage: by design it has already
finished by the time anything runs.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..contracts import (
    BudgetEnvelope,
    BudgetUsage,
    Composition,
    DomainSpec,
    ErrorCode,
    Invocation,
    content_digest,
)
from ..operators.builtin import operation_of
from ..registry import OperatorRegistry
from .telemetry import Tracer
from .verifier import Denial, Verifier


@dataclass
class Budget:
    """Allowances resolved against the item count at stage start.

    File counts are unbounded, so a fixed constant would either strangle a large run or
    leave a small one unmetered.
    """

    envelope: BudgetEnvelope
    items: int
    usage: BudgetUsage = field(default_factory=BudgetUsage)
    _started: float = field(default_factory=time.monotonic)

    @property
    def operator_calls(self) -> int:
        return self.envelope.operator_calls.resolve(self.items)

    @property
    def model_invocations(self) -> int:
        return self.envelope.model_invocations.resolve(self.items)

    def remaining_operator_calls(self) -> int:
        return max(0, self.operator_calls - self.usage.operator_calls)

    def consume_operator_call(self) -> None:
        if self.usage.operator_calls >= self.operator_calls:
            raise Denial(
                ErrorCode.BUDGET_EXHAUSTED,
                f"operator call budget exhausted ({self.operator_calls})",
            )
        self.usage = self.usage.model_copy(
            update={"operator_calls": self.usage.operator_calls + 1}
        )

    def consume_model_invocation(self, *, tokens: int = 0, cost_usd: float = 0.0) -> None:
        if self.usage.model_invocations >= self.model_invocations:
            raise Denial(
                ErrorCode.BUDGET_EXHAUSTED,
                f"model invocation budget exhausted ({self.model_invocations})",
            )
        self.usage = self.usage.model_copy(
            update={
                "model_invocations": self.usage.model_invocations + 1,
                "total_tokens": self.usage.total_tokens + tokens,
                "cost_usd": self.usage.cost_usd + cost_usd,
            }
        )
        if self.usage.cost_usd > self.envelope.max_cost_usd:
            raise Denial(
                ErrorCode.BUDGET_EXHAUSTED,
                f"cost budget exhausted (${self.envelope.max_cost_usd})",
            )

    def check_wall_time(self) -> None:
        elapsed = time.monotonic() - self._started
        if elapsed > self.envelope.wall_time_seconds:
            raise Denial(
                ErrorCode.BUDGET_EXHAUSTED,
                f"wall-time budget exhausted ({self.envelope.wall_time_seconds}s)",
            )


@dataclass(frozen=True)
class InvocationResult:
    invocation_id: str
    operator: str
    operator_version: str
    succeeded: bool
    started_at: str
    ended_at: str
    output: dict[str, Any] | None = None
    output_digest: str | None = None
    error_code: ErrorCode | None = None
    error_detail: str | None = None

    def journal_row(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "operator_version": self.operator_version,
            "succeeded": self.succeeded,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "output_digest": self.output_digest,
            "error_code": str(self.error_code) if self.error_code else None,
        }


def _load_entrypoint(entrypoint: str) -> Any:
    module_name, _, attribute = entrypoint.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


class Executor:
    def __init__(
        self,
        registry: OperatorRegistry,
        verifier: Verifier,
        *,
        tracer: Tracer | None = None,
    ) -> None:
        self.registry = registry
        self.verifier = verifier
        self.tracer = tracer

    def execute(
        self,
        composition: Composition,
        domain: DomainSpec,
        *,
        stage_inputs: dict[str, Any],
        config: dict[str, Any],
        workspace: Path,
        budget: Budget,
        stage: str | None = None,
        attempt: int | None = None,
        granted: frozenset[str] = frozenset(),
        tracer: Tracer | None = None,
    ) -> tuple[InvocationResult, ...]:
        """Verify then run every invocation in order, resolving declared inputs."""
        self.verifier.verify_composition(
            composition,
            domain,
            operator_call_budget=budget.remaining_operator_calls(),
            granted=granted,
        )

        outputs: dict[str, dict[str, Any]] = {}
        results: list[InvocationResult] = []

        for invocation in composition.invocations:
            budget.check_wall_time()
            budget.consume_operator_call()
            result = self._run_one(
                invocation,
                stage_inputs=stage_inputs,
                prior=outputs,
                config=config,
                workspace=workspace,
                domain=domain,
                stage=stage,
                attempt=attempt,
                tracer=tracer,
            )
            results.append(result)
            if result.output is not None:
                outputs[invocation.invocation_id] = result.output
            if not result.succeeded:
                # Stop this composition, but do not raise: a failure is data the planner
                # reads at the stage boundary, not a crash.
                break

        return tuple(results)

    def _run_one(
        self,
        invocation: Invocation,
        *,
        stage_inputs: dict[str, Any],
        prior: dict[str, dict[str, Any]],
        config: dict[str, Any],
        workspace: Path,
        domain: DomainSpec,
        stage: str | None,
        attempt: int | None,
        tracer: Tracer | None = None,
    ) -> InvocationResult:
        registered = self.registry.get(invocation.operator)
        spec = registered.spec
        started = time.time()
        started_at = _isoformat(started)

        arguments: dict[str, Any] = {
            "operation": operation_of(invocation.operator),
            "config": config,
            **invocation.parameters,
            # Which keys the agent wrote itself, as opposed to values that flowed from a
            # prior operator.  Lets an operator refuse a hand-written value.
            "_agent_supplied": sorted(invocation.parameters),
            # Runtime bookkeeping, always available. A capability should not have to
            # thread its own progress through every invocation to make scheduling work,
            # and a live model reliably did not — so the same window was taken forever.
            **{
                key: item
                for key, item in stage_inputs.items()
                if key.startswith("_") and key != "_agent_supplied"
            },
        }
        for reference in invocation.inputs:
            # Two rules, deliberately different.  A prior invocation's output *splats*, so
            # chaining operators inside one composition needs no wiring.  A stage input
            # *binds by name*, so a mapping arrives whole rather than being scattered into
            # unrelated argument names.  Use `bindings` to rename either.
            if reference in prior:
                arguments.update(prior[reference])
            elif reference in stage_inputs:
                arguments[reference] = stage_inputs[reference]
            else:
                return InvocationResult(
                    invocation_id=invocation.invocation_id,
                    operator=invocation.operator,
                    operator_version=spec.version,
                    succeeded=False,
                    started_at=started_at,
                    ended_at=_isoformat(time.time()),
                    error_code=ErrorCode.COMPOSITION_INVALID,
                    error_detail=f"unresolved input reference: {reference}",
                )

        for target, source in invocation.bindings.items():
            if "." in source:
                origin, _, key = source.partition(".")
                nested = prior.get(origin, {})
                if key in nested:
                    arguments[target] = nested[key]
                    continue
            if source not in arguments:
                # The failure a live run kept hitting was binding an operator's own
                # output back into it, mirroring the catalog's `produces` list into
                # `bindings`. Saying only "no resolved source" left the next round to
                # guess; naming what is actually bindable ends it in one.
                from ..operators.contracts import OUTPUT_KEYS

                bindable = _bindable(arguments)
                if source in OUTPUT_KEYS.get(invocation.operator, ()):
                    detail = (
                        f"binding {target}={source} names an output of "
                        f"{invocation.operator}, not an input. Outputs need no declaring; "
                        f"a later invocation reads them by name"
                    )
                elif target in arguments and source in _required_of(invocation.operator):
                    # Written the other way round: the value it wants is here, under the
                    # name it put on the left. Same class as the alias map in ADR 0001,
                    # and as cheap to end by saying which way the arrow points.
                    detail = (
                        f"binding {target}={source} is reversed. A binding is "
                        f"argument=source, so this fills {target} from {source}. "
                        f"Write {source}={target}"
                    )
                elif "." in source:
                    # A dotted source names an earlier invocation and one of its keys.
                    # Listing the working values does not help when the mistake was the
                    # invocation id: a live run guessed `collide.resolutions` at an
                    # invocation actually called `resolve_collisions`, and was told what
                    # was bindable without being told what it had just produced.
                    _, _, wanted = source.partition(".")
                    offers = [
                        f"{name}.{key}"
                        for name, output in prior.items()
                        for key in (output or {})
                        if not wanted or key == wanted
                    ]
                    detail = (
                        f"binding {target}={source} names no earlier invocation; "
                        f"available: {', '.join(sorted(offers)) or 'nothing produced yet'}"
                    )
                else:
                    # Runtime plumbing is in the argument mapping but is not the
                    # composition's to bind, so offering it would only invite the next
                    # mistake.
                    detail = (
                        f"binding {target}={source} has no resolved source; "
                        f"bindable here: {', '.join(bindable) or 'nothing yet'}"
                    )
                return InvocationResult(
                    invocation_id=invocation.invocation_id,
                    operator=invocation.operator,
                    operator_version=spec.version,
                    succeeded=False,
                    started_at=started_at,
                    ended_at=_isoformat(time.time()),
                    error_code=ErrorCode.COMPOSITION_INVALID,
                    error_detail=detail,
                )
            arguments[target] = arguments[source]

        # The caller's tracer wins, so a component span joins the run's tree rather than
        # forming an orphan under whichever tracer the executor was constructed with.
        active = tracer or self.tracer
        span = (
            active.span(
                f"operator.{invocation.operator}",
                stage=stage,
                attempt=attempt,
                domain=domain.id,
                operator=invocation.operator,
                operator_version=spec.version,
            )
            if active
            else _null_span()
        )

        with span as state:
            try:
                # Before the runner: a missing or mistyped argument is a contract
                # violation, and saying so beats a KeyError from deep inside an operator.
                registered.check_input(arguments)
            except ValidationError as exc:
                if state is not None:
                    state.fail(ErrorCode.COMPOSITION_INVALID)
                return InvocationResult(
                    invocation_id=invocation.invocation_id,
                    operator=invocation.operator,
                    operator_version=spec.version,
                    succeeded=False,
                    started_at=started_at,
                    ended_at=_isoformat(time.time()),
                    error_code=ErrorCode.COMPOSITION_INVALID,
                    error_detail=_explain(
                        invocation.operator, exc, arguments, stage_inputs, prior
                    ),
                )

            try:
                runner = _load_entrypoint(spec.entrypoint)
                output = runner(arguments, workspace)
                output = registered.validate_output(output)
            except Denial:
                raise
            except Exception as exc:  # noqa: BLE001 - an operator failure is data
                if state is not None:
                    state.fail(ErrorCode.OPERATOR_FAILED)
                return InvocationResult(
                    invocation_id=invocation.invocation_id,
                    operator=invocation.operator,
                    operator_version=spec.version,
                    succeeded=False,
                    started_at=started_at,
                    ended_at=_isoformat(time.time()),
                    error_code=ErrorCode.OPERATOR_FAILED,
                    error_detail=f"{type(exc).__name__}: {exc}",
                )

            digest = content_digest(output)
            if state is not None:
                state.digests["output"] = digest

        return InvocationResult(
            invocation_id=invocation.invocation_id,
            operator=invocation.operator,
            operator_version=spec.version,
            succeeded=True,
            started_at=started_at,
            ended_at=_isoformat(time.time()),
            output=output,
            output_digest=digest,
        )


def _explain(
    operator: str,
    error: ValidationError,
    arguments: dict[str, Any] | None = None,
    stage_inputs: dict[str, Any] | None = None,
    prior: dict[str, dict[str, Any]] | None = None,
) -> str:
    """A message an agent can act on next attempt, not a stack of pydantic internals."""
    problems = []
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"]) or "<root>"
        problems.append(f"{field}: {item['msg']}")
    detail = f"{operator} argument contract not satisfied - " + "; ".join(problems[:5])
    # Naming what is missing without naming what is here leaves the next round to guess,
    # and it guessed wrong three attempts running on a value that was sitting in front
    # of it under another name.
    bindable = _bindable(arguments or {})
    detail = f"{detail}; bindable here: {', '.join(bindable) or 'nothing'}"

    # And what it has not pulled in yet. An invocation can only bind from what its own
    # `inputs` reference, so listing only that leaves out the one thing that would have
    # worked: a live run needed `decision_digest`, the value that fills it was in the
    # working set under `digest`, and every message it got listed everything except that.
    unreferenced = sorted(
        set(_bindable(stage_inputs or {}))
        - set(bindable)
        - set(prior or {})
    )
    produced = sorted(
        f"{name}.{key}" for name, output in (prior or {}).items() for key in (output or {})
    )
    if unreferenced:
        detail = f"{detail}; referenceable via inputs: {', '.join(unreferenced)}"
    if produced:
        detail = f"{detail}; produced by earlier invocations: {', '.join(produced)}"
    return detail


def _required_of(operator: str) -> frozenset[str]:
    """Argument names an operator cannot run without."""
    from ..operators.contracts import INPUT_MODELS

    model = INPUT_MODELS.get(operator)
    if model is None:
        return frozenset()
    return frozenset(
        name for name, field in model.model_fields.items() if field.is_required()
    )


def _bindable(arguments: dict[str, Any]) -> list[str]:
    """Argument names a composition may actually bind from, excluding runtime plumbing."""
    return sorted(
        key
        for key in arguments
        if not key.startswith("_") and key not in {"config", "operation"}
    )


def _isoformat(epoch: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, UTC).isoformat()


class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _null_span() -> _NullSpan:
    return _NullSpan()
