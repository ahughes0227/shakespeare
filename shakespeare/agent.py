"""Level-2 domain subagents.

A subagent receives a goal, its declared scope, its operator catalog and the stage
inputs, and returns exactly one `Composition`.  It then stops.  It does not call
operators, does not see their results, and cannot adapt — if the results need
interpreting, that is the next stage's job.

The response model is a draft rather than a `Composition`, so an agent cannot claim to be
acting for a different domain than the one it was issued.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import Composition, Contract, DomainGoal, DomainSpec, Invocation
from .gateway import Gateway, ModelProfile, ModelUsage, render_prompt
from .operators.contracts import argument_summary
from .prompts import PromptStore


class CompositionDraft(Contract):
    """What the model is allowed to return."""

    invocations: tuple[Invocation, ...]
    rationale: str = ""


class DomainAgent(Protocol):
    def compose(
        self,
        *,
        domain: DomainSpec,
        goal: DomainGoal,
        stage_inputs: dict[str, Any],
        catalog_summary: dict[str, Any],
    ) -> tuple[Composition, ModelUsage | None]: ...


@dataclass
class ModelDomainAgent:
    gateway: Gateway
    profile: ModelProfile
    prompts: PromptStore = field(default_factory=PromptStore)

    def compose(
        self,
        *,
        domain: DomainSpec,
        goal: DomainGoal,
        stage_inputs: dict[str, Any],
        catalog_summary: dict[str, Any],
    ) -> tuple[Composition, ModelUsage | None]:
        artifact = self.prompts.load(goal.domain_id, domain.prompt_version)
        messages = render_prompt(
            artifact,
            scope=domain.scope,
            goal=goal.goal,
            success_criterion=goal.success_criterion,
            available_operators={
                name: argument_summary(name) for name in sorted(domain.catalog)
            },
            available_config_groups=sorted(domain.config_groups),
            catalog=catalog_summary,
            stage_inputs=_for_prompt(stage_inputs),
        )
        draft, usage = self.gateway.complete(self.profile, messages, CompositionDraft)
        return (
            Composition(
                domain_id=domain.id,
                invocations=draft.invocations,
                rationale=draft.rationale,
            ),
            usage,
        )


def _for_prompt(stage_inputs: dict[str, Any]) -> dict[str, Any]:
    """Stage inputs as the model must see them.

    A prompt *does* carry document content — reading an invoice is the entire point of
    the field-resolution domain, and stripping the text would make the work impossible.
    The redaction boundary is the telemetry channel, not this one: content stays in
    process and in the prompt, and only digests are ever exported. See telemetry.py.

    Runtime bookkeeping (leading underscore) is dropped because it is noise to a model,
    not because it is sensitive.
    """
    return {key: value for key, value in stage_inputs.items() if not key.startswith("_")}


@dataclass
class FakeDomainAgent:
    """Scripted subagent so the suite runs offline.

    Compositions are keyed by domain id, which lets a test drive two agents down
    deliberately different routes and assert the plan is identical anyway.
    """

    compositions: dict[str, list[Composition]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def queue(self, domain_id: str, *values: Composition) -> FakeDomainAgent:
        self.compositions.setdefault(domain_id, []).extend(values)
        return self

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def compose(
        self,
        *,
        domain: DomainSpec,
        goal: DomainGoal,
        stage_inputs: dict[str, Any],
        catalog_summary: dict[str, Any],
    ) -> tuple[Composition, ModelUsage | None]:
        self.calls.append(domain.id)
        queued = self.compositions.get(domain.id)
        if not queued:
            raise KeyError(f"FakeDomainAgent has no queued composition for {domain.id}")
        composition = queued.pop(0) if len(queued) > 1 else queued[0]
        return composition.model_copy(update={"domain_id": domain.id}), None
