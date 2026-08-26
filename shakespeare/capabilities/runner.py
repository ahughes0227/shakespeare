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
        if isinstance(value, dict) and "organization" in value:
            # Either the whole answer is wrapped, or a commentary block sits beside it.
            inner = value["organization"]
            value = inner if "invocations" not in value else {
                k: v for k, v in value.items() if k != "organization"
            }
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


@dataclass
class BatchCost:
    """What one batch actually cost, which is what sizes the next one.

    Measured rather than assumed: a fixed batch size is only right when every item costs
    the same, and invoices do not.
    """

    completion_tokens: int = 0
    truncated: bool = False
    finished: bool = False

    def observe(self, usage: ModelUsage | None) -> None:
        if usage is not None:
            self.completion_tokens += usage.completion_tokens


@dataclass(frozen=True)
class CapabilityOutcome:
    capability: str
    rounds: tuple[Round, ...]
    artifacts: tuple[Artifact, ...]
    context: dict[str, Any]
    exhausted: bool = False
    #: The runtime's own scheduling calls, so the journal records why a capability was
    #: asked what it was asked. Without them the audit log shows the batches but not the
    #: decision that produced them.
    scheduling: tuple[tuple[Composition, tuple[InvocationResult, ...]], ...] = ()

    @property
    def sufficient(self) -> bool:
        """Claimed *and* borne out. A round that failed has established nothing."""
        return (
            bool(self.rounds)
            and self.rounds[-1].succeeded
            and self.rounds[-1].organization.sufficient
        )


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
    #: Output tokens one response may use. The ceiling scheduling divides against.
    capacity: int = 16384
    #: How many times a batch may be re-sized after being cut off before the capability
    #: is called exhausted. Bounded because each attempt is billed.
    max_resize_attempts: int = 3
    _compose: Any = field(default=None, repr=False)

    def _plan_batch(
        self,
        capability: CapabilitySpec,
        remaining: list[Any],
        observations: list[dict[str, Any]],
        *,
        workspace: Path,
        budget: Budget,
        goal_id: str,
        journal: list[tuple[Composition, tuple[InvocationResult, ...]]] | None = None,
    ) -> dict[str, Any]:
        """Size the next batch, given what the earlier ones actually cost.

        Called by the runtime rather than by the capability, because sizing a batch is
        arithmetic over measured cost and §10 makes scheduling the runtime's job. It is
        still an operator, so the decision is verified, journalled and traced like any
        other — and no capability lists it, so none can schedule itself.
        """
        composition = Composition(
            domain_id=capability.id,
            invocations=(
                Invocation(
                    invocation_id="schedule",
                    operator="schedule.plan",
                    parameters={
                        "remaining": remaining,
                        "capacity": self.capacity,
                        "cost_per_item": capability.cost_per_item,
                        # Copied, because the live list keeps growing and the journal
                        # would otherwise record every decision with the final state.
                        "observations": list(observations),
                    },
                ),
            ),
            rationale="size the next batch to fit one response",
        )
        outcome = self.executor.execute(
            composition,
            _scheduling_domain(capability),
            stage_inputs={},
            config={},
            workspace=workspace,
            budget=budget,
            stage=goal_id,
        )
        if journal is not None:
            journal.append((composition, tuple(outcome)))
        planned = outcome[0].output if outcome and outcome[0].output else None
        if not planned:
            # Scheduling itself failed. Handing the whole set over is the honest
            # fallback: it is what an undivided capability would have received anyway.
            return {"needed": False, "batch": list(remaining), "batch_size": len(remaining)}
        return dict(planned)

    def run(
        self,
        *,
        capability: CapabilitySpec,
        request: str,
        context: dict[str, Any],
        budget: Budget,
        workspace: Path,
        goal_id: str = "",
        feedback: dict[str, Any] | None = None,
    ) -> CapabilityOutcome:
        working = dict(context)
        if feedback:
            # Why the last attempt was rejected. A retry that is told nothing can only
            # do the same thing again and hope.
            working["previous_attempt"] = feedback
        agent = self.agents.get(capability.id) or self.agents["*"]
        rounds: list[Round] = []
        produced: list[Artifact] = []

        scheduling: list[tuple[Composition, tuple[InvocationResult, ...]]] = []

        def outcome(*, exhausted: bool) -> CapabilityOutcome:
            return CapabilityOutcome(
                capability=capability.id,
                rounds=tuple(rounds),
                artifacts=tuple(produced),
                context=working,
                exhausted=exhausted,
                scheduling=tuple(scheduling),
            )

        divisible = working.get(capability.divides)
        if capability.cost_per_item is None or not isinstance(divisible, list) or not divisible:
            # Nothing measurable to divide — collision resolution and plan assembly need
            # the whole set at once, and say so by declaring no per-item cost.
            undivided = self._pursue_batch(
                capability=capability, request=request, working=working, rounds=rounds,
                produced=produced, budget=budget, workspace=workspace, goal_id=goal_id,
                agent=agent,
            )
            return outcome(exhausted=not undivided.finished)

        whole: list[Any] = list(divisible)
        # Work already carried in from an earlier attempt is not work. A live run kept
        # restarting at sixty items, once throwing away fifty-nine it had just resolved.
        remaining: list[Any] = _outstanding(whole, working, _progress_keys(capability))
        observations: list[dict[str, Any]] = []
        accumulated: dict[str, list[Any]] = {}
        number = 0
        attempts = 0
        if not remaining:
            # Every item is accounted for already; one round to publish what is there.
            final = self._pursue_batch(
                capability=capability, request=request, working=working, rounds=rounds,
                produced=produced, budget=budget, workspace=workspace, goal_id=goal_id,
                agent=agent, focus=_identify(whole),
            )
            working[capability.divides] = whole
            return outcome(exhausted=not final.finished)

        while remaining:
            plan = self._plan_batch(
                capability, remaining, observations,
                workspace=workspace, budget=budget, goal_id=goal_id, journal=scheduling,
            )
            batch = plan["batch"]
            number += 1
            working[capability.divides] = batch
            if plan["needed"] or observations:
                # Only say so when it is true. A capability handed the whole set should
                # not be told it is looking at batch one of one. The counts matter: shown
                # thirty items beside an artifact summary saying sixty, a capability
                # reported itself incomplete every round and never finished a batch.
                working["batch_number"] = number
                working["batch_remaining"] = len(remaining) - len(batch)
                working["batch_total"] = len(whole)

            spent = self._pursue_batch(
                capability=capability, request=request, working=working, rounds=rounds,
                produced=produced, budget=budget, workspace=workspace, goal_id=goal_id,
                agent=agent, focus=_identify(batch),
            )
            _accumulate(working, accumulated, exclude=capability.divides)
            observations.append(
                {
                    "items": len(batch),
                    "completion_tokens": spent.completion_tokens,
                    "truncated": spent.truncated,
                    "failed": not spent.finished,
                }
            )
            if spent.finished:
                attempts = 0
                remaining = remaining[len(batch) :]
                continue

            # It did not finish. If the reason was size, the observation just recorded
            # will make the next plan smaller, so the same items are worth another go.
            attempts += 1
            if not spent.truncated or len(batch) <= 1 or attempts >= self.max_resize_attempts:
                working[capability.divides] = whole
                return outcome(exhausted=True)

        working[capability.divides] = whole
        return outcome(exhausted=False)

    def _pursue_batch(
        self,
        *,
        capability: CapabilitySpec,
        request: str,
        working: dict[str, Any],
        rounds: list[Round],
        produced: list[Artifact],
        budget: Budget,
        workspace: Path,
        goal_id: str,
        agent: CapabilityAgent,
        focus: frozenset[str] | None = None,
    ) -> BatchCost:
        """Rounds within one batch, which are for self-correction rather than progress."""
        from ..compose import CompositionError, compose

        cost = BatchCost()

        # Rounds already recorded for earlier batches are history this batch should see,
        # but its own correction history is what matters for "what went wrong last time".
        opened = len(rounds)
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
                    context=_summarise(working, focus=focus, divides=capability.divides),
                    # The capability sees its own prior rounds. This is what lets it carry
                    # on rather than restart, and what makes "enough evidence" its
                    # judgment.
                    prior=[item.digest_row() for item in rounds[opened:]],
                    catalog_summary=_catalog_summary(capability, self.config_root),
                )
            except GatewayError as failure:
                cost.observe(failure.usage)
                cost.truncated = cost.truncated or failure.truncated
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

            cost.observe(usage)
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
            _record_progress(working)

            completed = Round(number=number, organization=organization, results=results,
                              denial=denial)
            rounds.append(completed)
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

            # A round whose components failed has established nothing, whatever it says
            # about itself. Publishing anyway put an empty FileInventory in the store,
            # the gate saw the kind it required and accepted, and a sixty-file run then
            # spent three attempts asking a capability to extract from nothing.
            if organization.publishes and completed.succeeded:
                produced.append(
                    self.artifacts.put(
                        kind=organization.publishes,
                        payload=_publishable(working, organization),
                        produced_by=capability.id,
                        quality=organization.quality,
                        summary=organization.summary,
                    )
                )

            if organization.sufficient and completed.succeeded:
                cost.finished = True
                return cost

        return cost


def _scheduling_domain(capability: CapabilitySpec) -> Any:
    """The runtime's own grant for dividing work.

    Narrow on purpose: it permits exactly one component, and no capability package lists
    it, so scheduling cannot be reached from inside a capability.
    """
    from ..contracts import DomainSpec

    return DomainSpec(
        id=capability.id,
        scope="divide work into batches that fit one response",
        catalog=frozenset({"schedule.plan"}),
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


#: Shown in full rather than described, because describing the shape of a diagnosis
#: leaves the next attempt exactly as uninformed as the last one.
_VERBATIM: frozenset[str] = frozenset({"previous_attempt"})

#: Output keys whose entries carry an item_id, and therefore count as work completed.
_PROGRESS_KEYS: tuple[str, ...] = ("candidates", "unrendered", "extractions", "results")


def _accumulate(
    working: dict[str, Any], accumulated: dict[str, list[Any]], *, exclude: str
) -> None:
    """Carry per-item results across batches instead of letting each one replace the last.

    An operator's output replaces the key it writes, which is right within a batch and
    wrong across them: the second batch's extractions would erase the first's, and the
    gate would then be told thirty of sixty files were the whole world. Anything keyed by
    item_id is merged by item instead, so what a capability accumulates survives being
    asked in pieces.
    """
    for key, value in list(working.items()):
        if key.startswith("_") or key == exclude or not isinstance(value, list) or not value:
            continue
        if not all(isinstance(row, dict) and "item_id" in row for row in value):
            continue
        merged = {row["item_id"]: row for row in accumulated.get(key, [])}
        merged.update({row["item_id"]: row for row in value})
        accumulated[key] = list(merged.values())
        working[key] = accumulated[key]


def _record_progress(working: dict[str, Any]) -> None:
    """Accumulate the item ids earlier rounds have already dealt with.

    Kept under a reserved key so it reaches every invocation without the capability
    having to thread it, and stays out of prompts.
    """
    seen: list[str] = list(working.get("_completed") or [])
    known = set(seen)
    for key in _PROGRESS_KEYS:
        for item in working.get(key) or []:
            if isinstance(item, dict) and (item_id := item.get("item_id")) not in known:
                if item_id is not None:
                    seen.append(str(item_id))
                    known.add(item_id)
    working["_completed"] = seen


def _progress_keys(capability: CapabilitySpec) -> tuple[str, ...]:
    """Which working keys count as this capability's own progress.

    Derived from what its own components produce, never shared. One capability's per-item
    output says nothing about whether another has done its work, and a single global
    record of "done" would let each inherit the other's progress and skip its own.
    """
    from ..operators.contracts import OUTPUT_KEYS

    produced = {key for name in capability.catalog for key in OUTPUT_KEYS.get(name, ())}
    return tuple(key for key in _PROGRESS_KEYS if key in produced)


def _outstanding(
    whole: list[Any], working: dict[str, Any], keys: tuple[str, ...]
) -> list[Any]:
    done = {
        str(row["item_id"])
        for key in keys
        for row in working.get(key) or []
        if isinstance(row, dict) and row.get("item_id") is not None
    }
    return [row for row in whole if str(row.get("item_id")) not in done]


def _identify(batch: list[Any]) -> frozenset[str]:
    return frozenset(
        str(row["item_id"]) for row in batch if isinstance(row, dict) and "item_id" in row
    )


def _summarise(
    working: dict[str, Any],
    *,
    focus: frozenset[str] | None = None,
    divides: str = "",
) -> dict[str, Any]:
    """Shape, not payload — except for the batch, which is handed over whole.

    Describing everything is right for a capability whose components do the reading: it
    binds `items` by name and the operator gets the real list. It is wrong for one that
    must do the reading itself. A live run put it plainly: "the available context
    provides only aggregate item and extraction counts, not the individual item IDs,
    paths, extensions, or extracted invoice text" — resolve was asked to read text it was
    never shown, three attempts running.

    So when a capability has been handed a batch, that batch and the per-item evidence
    belonging to it arrive in full. The batch was sized to fit one response; this is what
    it was sized for.
    """
    described: dict[str, Any] = {}
    for key, value in working.items():
        if key.startswith("_"):
            continue
        if key in _VERBATIM:
            described[key] = value
            continue
        if focus is not None and isinstance(value, list):
            if key == divides:
                described[key] = value
                continue
            belonging = [
                row
                for row in value
                if isinstance(row, dict) and str(row.get("item_id")) in focus
            ]
            if belonging:
                described[key] = belonging
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
