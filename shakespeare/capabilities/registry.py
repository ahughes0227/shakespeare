"""Capability packages.

A capability is a bounded, goal-directed bag of tools (§7). It provides abstraction — so
the planner never decomposes down to component calls — and containment — so what the
system may do while pursuing one kind of goal is limited by the package, not by the
model's discretion.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from ..contracts import Contract, SemanticCard

CAPABILITY_ROOT = Path(__file__).resolve().parent


class CapabilityRegistryError(RuntimeError):
    pass


class CapabilitySpec(Contract):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    #: What this capability is always for. The planner's request is a bounded instance of
    #: it; the standing goal is the bound.
    standing_goal: str = Field(min_length=1)
    #: The components it may call. The whole executable surface, fixed by the package.
    catalog: frozenset[str] = Field(min_length=1)
    config_groups: frozenset[str] = frozenset()
    #: Artifact kinds it can produce. A goal's gate requires kinds; this is how the
    #: planner knows which capability could possibly satisfy it.
    produces: tuple[str, ...] = Field(min_length=1)
    #: How many times it may reorganise internally before giving up. This is where
    #: adaptive meta-organization lives (§8), and this is its bound.
    max_rounds: int = Field(default=4, ge=1, le=100)
    #: Output tokens one item costs this capability, measured rather than guessed. Set it
    #: only for a capability that reports something per item: that is what makes the work
    #: divisible. A whole-set capability leaves it unset and is never scheduled.
    cost_per_item: int | None = Field(default=None, ge=1, le=100_000)
    #: The working key holding the set to divide.
    divides: str = "items"
    prompt_version: str = "1.0.0"

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"


class CapabilityRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or CAPABILITY_ROOT
        self._specs: dict[str, CapabilitySpec] = {}
        self._cards: dict[str, SemanticCard] = {}
        if self.root.is_dir():
            for manifest in sorted(self.root.glob("*/capability.yml")):
                self._load(manifest)

    def _load(self, manifest: Path) -> None:
        directory = manifest.parent
        try:
            spec = CapabilitySpec.model_validate(yaml.safe_load(manifest.read_text()) or {})
        except Exception as exc:
            raise CapabilityRegistryError(
                f"{manifest} is not a usable capability package ({type(exc).__name__}: {exc})"
            ) from exc
        if spec.id != directory.name:
            raise CapabilityRegistryError(
                f"{manifest}: declares {spec.id} but lives in {directory.name}"
            )
        card_path = directory / "capability-context.yml"
        if not card_path.is_file():
            raise CapabilityRegistryError(f"{spec.id} has no capability-context.yml")
        try:
            card = SemanticCard.model_validate(yaml.safe_load(card_path.read_text()) or {})
        except Exception as exc:
            raise CapabilityRegistryError(
                f"{spec.id}: capability-context.yml must populate all ten fields ({exc})"
            ) from exc
        self._specs[spec.id] = spec
        self._cards[spec.id] = card

    def register(self, spec: CapabilitySpec, card: SemanticCard) -> None:
        if spec.id in self._specs:
            raise CapabilityRegistryError(f"capability already registered: {spec.id}")
        self._specs[spec.id] = spec
        self._cards[spec.id] = card

    def get(self, capability_id: str) -> CapabilitySpec:
        try:
            return self._specs[capability_id]
        except KeyError as exc:
            raise CapabilityRegistryError(
                f"unknown capability: {capability_id}; registered: {sorted(self._specs)}"
            ) from exc

    def card(self, capability_id: str) -> SemanticCard:
        return self._cards[capability_id]

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def __contains__(self, capability_id: object) -> bool:
        return capability_id in self._specs

    def producing(self, kind: str) -> tuple[CapabilitySpec, ...]:
        """Capabilities that could produce a required artifact kind."""
        return tuple(spec for spec in self._specs.values() if kind in spec.produces)

    def routing_catalog(self) -> dict[str, dict[str, str]]:
        """What the planner reads when choosing who can answer a goal: the cards."""
        return {
            capability_id: self._cards[capability_id].model_dump(mode="json")
            for capability_id in sorted(self._specs)
        }
