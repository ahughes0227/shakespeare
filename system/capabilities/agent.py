"""Capability-level agents.

A capability agent performs the meta-organization §8 places inside the capability: given
a bounded request, the evidence available and what its own earlier rounds produced, it
decides what to do next and whether the work is finished.

It organizes; it never executes. Components run through the verifier and the executor, so
the catalog, the config groups and the write boundary remain exact whatever it decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..model_access import Gateway, ModelProfile, ModelUsage, render_prompt
from ..prompt_store import PromptStore
from .registry import CapabilitySpec
from .runner import Organization


@dataclass
class ModelCapabilityAgent:
    gateway: Gateway
    profile: ModelProfile
    prompts: PromptStore = field(default_factory=PromptStore)

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
        artifact = self.prompts.load(capability.id, capability.prompt_version)
        messages = render_prompt(
            artifact,
            standing_goal=capability.standing_goal,
            request=request,
            components=catalog_summary.get("components", {}),
            config_groups=catalog_summary.get("config", {}),
            artifacts_available=artifacts,
            working_context=context,
            previous_rounds=prior,
            rounds_remaining=capability.max_rounds - len(prior),
            publishes_choose_one=list(capability.produces),
        )
        return self.gateway.complete(self.profile, messages, Organization)


@dataclass
class FakeCapabilityAgent:
    """Scripted agent for offline tests."""

    plans: dict[str, list[Organization]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def queue(self, capability_id: str, *plans: Organization) -> FakeCapabilityAgent:
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
        queued = self.plans.get(capability.id)
        if not queued:
            raise KeyError(f"FakeCapabilityAgent has no organization for {capability.id}")
        return (queued.pop(0) if len(queued) > 1 else queued[0]), None
