"""Capability-level planning and meta-organization.

This is the layer §8 puts adaptive organization in: "the capability-level planner can
decide which tools to use, sequential vs parallel execution, whether semantic reasoning
is needed, whether enough evidence has been gathered."

A capability runs in rounds. Each round it organizes some work, the runtime executes the
components deterministically, and it *sees what came back*. That last part is the whole
difference: a capability that cannot observe its own results cannot decide it has done
twenty of sixty items and should carry on — which is exactly how a live sixty-invoice run
failed under the previous design.

The boundary is unchanged where it matters. A capability organizes; it never executes.
Components run through the same verifier and the same executor as before, so the catalog,
the config catalog, the write containment and the path guard are all still exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field as PydanticField
from pydantic import model_validator

from ..artifacts import Artifact, ArtifactStore, Quality
from ..contracts import Composition, Contract, ErrorCode, Invocation, OperatorAsk
from ..executor import Budget, Executor, InvocationResult
from ..gateway import GatewayError, ModelUsage
from ..verifier import Denial
from .registry import CapabilitySpec


class _NoSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _no_span() -> _NoSpan:
    return _NoSpan()


class Organization(Contract):
    """What a capability decides to do this round.

    It is a composition plus a judgment about whether the work is finished. The judgment
    is the capability's own — nothing above it can tell whether twenty of sixty is enough.
    """

    invocations: tuple[Invocation, ...] = ()
    #: What this round is meant to establish. Recorded for audit, not executed.
    intent: str = ""
    #: True when the capability believes the request is answered. A capability that says
    #: this while its own evidence is thin is caught by the gate, not here.
    sufficient: bool = False
    #: Artifact kind this round's output should be published as, if any.
    publishes: str | None = None
    #: How complete that artifact is. PARTIAL is how a capability says "correct so far".
    quality: Quality = Quality.COMPLETE
    summary: dict[str, Any] = PydanticField(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_common_shapes(cls, value: Any) -> Any:
        """Tolerate two shapes a model reaches for.

        A response wrapped in {"organization": {...}} is the same answer with a label on
        it, and `publishes` given as a one-element list is the same choice written as the
        options it was offered. Neither is worth losing a round over.
        """
        if isinstance(value, dict) and "organization" in value and "invocations" not in value:
            value = value["organization"]
        if isinstance(value, dict) and isinstance(value.get("publishes"), (list, tuple)):
            published = value["publishes"]
            value = {**value, "publishes": published[0] if published else None}
        return value

    #: A component this capability lacked. Evaluated after the round runs, so an admitted
    #: component becomes usable on the next round rather than mid-organization.
    ask: OperatorAsk | None = None


class CapabilityAgent(Protocol):
    def organize(
        self,
        *,
        capability: CapabilitySpec,
        request: str,
        artifacts: list[dict[str, Any]],
        context: dict[str, Any],
        prior: list[dict[str, Any]],
        catalog_summary: dict[str, Any],
    ) -> tuple[Organization, ModelUsage | None]: ...


@dataclass(frozen=True)
class Round:
    number: int
    organization: Organization
    results: tuple[InvocationResult, ...]
    denial: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.denial is None and all(item.succeeded for item in self.results)

    def digest_row(self) -> dict[str, Any]:
        """What the next round is told about this one: outcomes, never payloads."""
        return {
            "round": self.number,
            "intent": self.organization.intent,
            "ran": [item.operator for item in self.results],
            "succeeded": self.succeeded,
            "failures": [
                {"operator": item.operator, "detail": (item.error_detail or "")[:300]}
                for item in self.results
                if not item.succeeded
            ],
            "denial": self.denial,
        }


@dataclass(frozen=True)
class CapabilityOutcome:
    capability: str
    rounds: tuple[Round, ...]
    artifacts: tuple[Artifact, ...]
    context: dict[str, Any]
    exhausted: bool = False

    @property
    def sufficient(self) -> bool:
        return bool(self.rounds) and self.rounds[-1].organization.sufficient


@dataclass
class CapabilityRunner:
    """Runs one capability against one bounded request."""

    executor: Executor
    agents: dict[str, CapabilityAgent]
    artifacts: ArtifactStore
    config_root: str | None = None
    usage_sink: Any = None
    #: Called with (ask, capability_id) when a capability requests a component.
    ask_sink: Any = None
    #: Components admitted during this run, per capability.
    grants: dict[str, set[str]] = field(default_factory=dict)
    tracer: Any = None
    _compose: Any = field(default=None, repr=False)

    def run(
        self,
        *,
        capability: CapabilitySpec,
        request: str,
        context: dict[str, Any],
        budget: Budget,
        workspace: Path,
        goal_id: str = "",
    ) -> CapabilityOutcome:
        from ..compose import CompositionError, compose

        agent = self.agents.get(capability.id) or self.agents["*"]
        rounds: list[Round] = []
        produced: list[Artifact] = []
        working = dict(context)

        for number in range(1, capability.max_rounds + 1):
            round_span = (
                self.tracer.span(
                    f"round.{capability.id}",
                    stage=goal_id,
                    attempt=number,
                    domain=capability.id,
                )
                if self.tracer
                else _no_span()
            )
            round_state = round_span.__enter__()
            try:
                organization, usage = agent.organize(
                    capability=capability,
                    request=request,
                    artifacts=self.artifacts.describe(),
                    context=_summarise(working),
                    # The capability sees its own prior rounds. This is what lets it carry
                    # on rather than restart, and what makes "enough evidence" its
                    # judgment.
                    prior=[item.digest_row() for item in rounds],
                    catalog_summary=_catalog_summary(capability, self.config_root),
                )
            except GatewayError as failure:
                # A response that violates its contract is a failed round, not a failed
                # run. The capability has rounds precisely so it can correct itself, and
                # the reason reaches the next one.
                if self.usage_sink is not None:
                    self.usage_sink(
                        f"capability.{capability.id}",
                        failure.usage,
                        capability.prompt_version,
                    )
                rounds.append(
                    Round(
                        number=number,
                        organization=Organization(intent="unusable response"),
                        results=(),
                        denial=str(failure),
                    )
                )
                if round_state is not None:
                    # The class of refusal, never its text: a denial message can quote
                    # the value that caused it.
                    round_state.fail(failure.code)
                    round_state.record(sufficient=False)
                round_span.__exit__(None, None, None)
                continue

            if self.usage_sink is not None:
                self.usage_sink(f"capability.{capability.id}", usage, capability.prompt_version)

            composition = Composition(
                domain_id=capability.id,
                invocations=organization.invocations,
                rationale=organization.intent,
            )
            denial: str | None = None
            results: tuple[InvocationResult, ...] = ()
            if organization.invocations:
                try:
                    config = compose(
                        _selections(composition),
                        allowed_groups=capability.config_groups,
                        config_root=self.config_root,
                    )
                    results = self.executor.execute(
                        composition,
                        _as_domain(capability),
                        stage_inputs=working,
                        config=config,
                        workspace=workspace,
                        budget=budget,
                        stage=goal_id,
                        attempt=number,
                        granted=frozenset(self.grants.get(capability.id, set())),
                        tracer=self.tracer,
                    )
                except (Denial, CompositionError) as refusal:
                    denial = str(refusal)

            for item in results:
                if item.output:
                    working.update(item.output)

            rounds.append(Round(number=number, organization=organization, results=results,
                                denial=denial))
            if round_state is not None:
                round_state.record(
                    sufficient=organization.sufficient,
                    published=organization.publishes,
                    quality=str(organization.quality),
                )
                # The capability's own progress numbers. This is where "20 of 60" lives,
                # and it is the difference between seeing that a run stalled and seeing
                # why.
                round_state.add_counts(organization.summary)
                round_state.add_count("components", len(results))
                round_state.add_count(
                    "failed", sum(1 for item in results if not item.succeeded)
                )
                if denial is not None:
                    round_state.fail(ErrorCode.COMPOSITION_INVALID)
            round_span.__exit__(None, None, None)

            if organization.ask is not None and self.ask_sink is not None:
                admitted = self.ask_sink(organization.ask, capability.id)
                if admitted:
                    self.grants.setdefault(capability.id, set()).add(admitted)

            if organization.publishes and denial is None:
                produced.append(
                    self.artifacts.put(
                        kind=organization.publishes,
                        payload=_publishable(working, organization),
                        produced_by=capability.id,
                        quality=organization.quality,
                        summary=organization.summary,
                    )
                )

            if organization.sufficient and denial is None:
                return CapabilityOutcome(
                    capability=capability.id,
                    rounds=tuple(rounds),
                    artifacts=tuple(produced),
                    context=working,
                )

        return CapabilityOutcome(
            capability=capability.id,
            rounds=tuple(rounds),
            artifacts=tuple(produced),
            context=working,
            exhausted=True,
        )


def _as_domain(capability: CapabilitySpec) -> Any:
    """Reuse the verifier's existing containment check.

    A capability bounds exactly what a domain used to, so the authorization surface is
    unchanged: the catalog and config groups still come from the package.
    """
    from ..contracts import DomainSpec

    return DomainSpec(
        id=capability.id,
        scope=capability.standing_goal,
        skippable=True,
        catalog=capability.catalog,
        config_groups=capability.config_groups,
        prompt_version=capability.prompt_version,
    )


def _selections(composition: Composition) -> dict[str, str]:
    merged: dict[str, str] = {}
    for invocation in composition.invocations:
        merged.update(invocation.selections)
    return merged


def _catalog_summary(capability: CapabilitySpec, config_root: str | None) -> dict[str, Any]:
    from ..compose import catalog as hydra_catalog
    from ..operators.contracts import argument_summary

    available = hydra_catalog(config_root)
    return {
        "components": {name: argument_summary(name) for name in sorted(capability.catalog)},
        "config": {
            group: sorted(available[group])
            for group in sorted(capability.config_groups)
            if group in available
        },
    }


def _summarise(working: dict[str, Any]) -> dict[str, Any]:
    """Shape, not payload. A capability is told what it has, not handed all of it."""
    described: dict[str, Any] = {}
    for key, value in working.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list):
            described[key] = {"kind": "list", "count": len(value)}
        elif isinstance(value, dict):
            described[key] = {"kind": "mapping", "keys": sorted(value)[:12]}
        elif isinstance(value, str) and len(value) > 120:
            described[key] = {"kind": "text", "chars": len(value)}
        else:
            described[key] = value
    return described


def _publishable(working: dict[str, Any], organization: Organization) -> dict[str, Any]:
    """The working values an artifact should carry, excluding runtime bookkeeping."""
    return {key: value for key, value in working.items() if not key.startswith("_")}





class ScriptedCapabilityAgent:
    """Offline agent: a queued organization per round.

    Scriptable per round so a test can drive a capability that carries on, one that
    declares itself finished early, and one that never converges.
    """

    def __init__(self) -> None:
        self.plans: dict[str, list[Organization]] = {}
        self.calls: list[str] = []
        self.seen_prior: list[list[dict[str, Any]]] = []

    def queue(self, capability_id: str, *plans: Organization) -> ScriptedCapabilityAgent:
        self.plans.setdefault(capability_id, []).extend(plans)
        return self

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def organize(
        self,
        *,
        capability: CapabilitySpec,
        request: str,
        artifacts: list[dict[str, Any]],
        context: dict[str, Any],
        prior: list[dict[str, Any]],
        catalog_summary: dict[str, Any],
    ) -> tuple[Organization, ModelUsage | None]:
        self.calls.append(capability.id)
        self.seen_prior.append(prior)
        queued = self.plans.get(capability.id)
        if not queued:
            raise KeyError(f"no organization queued for {capability.id}")
        return (queued.pop(0) if len(queued) > 1 else queued[0]), None


__all__ = [
    "CapabilityAgent",
    "CapabilityOutcome",
    "CapabilityRunner",
    "ErrorCode",
    "Organization",
    "Round",
    "ScriptedCapabilityAgent",
]
