"""Workflows as goal graphs.

A workflow is a predefined graph of goals and dependencies (§3). It defines what must
become true, not the procedure for making it true, so registration validates the *shape*
of the intent rather than the order of an execution path.

Only causal dependencies belong here. If two goals can be pursued independently the graph
must not force them into sequence, so nothing checks for a total order and none is
implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..capabilities import CapabilityRegistry
from ..components.builtin import RUNTIME_ONLY
from ..components.registry import OperatorRegistry
from ..contracts import Contract, SemanticCard, content_digest
from ..runtime.checks import CHECKS
from ..runtime.compose import catalog as hydra_catalog
from ..runtime.goals import Goal, GoalGraph

WORKFLOW_ROOT = Path(__file__).resolve().parent


class WorkflowRegistryError(RuntimeError):
    pass


class WorkflowSpec(Contract):
    id: str
    version: str
    goals: tuple[Goal, ...]
    #: The goal after which the irreversible commit runs.
    commit_after: str
    entry_contract: str = "RequestContract"

    @property
    def graph(self) -> GoalGraph:
        return GoalGraph(goals=self.goals)


@dataclass(frozen=True)
class RegisteredWorkflow:
    spec: WorkflowSpec
    card: SemanticCard

    def digest(self) -> str:
        """Identity of everything that determines behaviour: the goals and their gates."""
        return content_digest({"workflow": self.spec})


class WorkflowRegistry:
    def __init__(
        self,
        *,
        capabilities: CapabilityRegistry,
        operators: OperatorRegistry,
        root: Path | None = None,
        config_root: str | None = None,
    ) -> None:
        self.root = root or WORKFLOW_ROOT
        self.capabilities = capabilities
        self.operators = operators
        self.config_root = config_root
        self._workflows: dict[str, RegisteredWorkflow] = {}
        if self.root.is_dir():
            for manifest in sorted(self.root.glob("*/workflow.yml")):
                self._load_one(manifest)

    def _load_one(self, manifest: Path) -> None:
        directory = manifest.parent
        try:
            spec = WorkflowSpec.model_validate(yaml.safe_load(manifest.read_text()) or {})
        except Exception as exc:
            raise WorkflowRegistryError(
                f"{manifest} is not a usable workflow ({type(exc).__name__}: {exc})"
            ) from exc
        if spec.id != directory.name:
            raise WorkflowRegistryError(
                f"{manifest}: declares {spec.id} but lives in {directory.name}"
            )
        card_path = directory / "workflow-context.yml"
        if not card_path.is_file():
            raise WorkflowRegistryError(f"{spec.id} has no workflow-context.yml")
        try:
            card = SemanticCard.model_validate(yaml.safe_load(card_path.read_text()) or {})
        except Exception as exc:
            raise WorkflowRegistryError(
                f"{spec.id}: workflow-context.yml must populate all ten fields ({exc})"
            ) from exc
        self.register(spec, card)

    def register(self, spec: WorkflowSpec, card: SemanticCard) -> None:
        if spec.id in self._workflows:
            raise WorkflowRegistryError(f"workflow already registered: {spec.id}")
        self.validate(spec)
        self._workflows[spec.id] = RegisteredWorkflow(spec=spec, card=card)

    def validate(self, spec: WorkflowSpec) -> None:
        """Check the graph is coherent and answerable before a run starts."""
        graph = spec.graph  # rejects cycles and unknown dependencies

        if spec.commit_after not in {goal.id for goal in spec.goals}:
            raise WorkflowRegistryError(
                f"{spec.id}: commit_after names a goal not in the graph: {spec.commit_after}"
            )

        groups = hydra_catalog(self.config_root)
        for goal in graph.goals:
            for name in goal.capabilities:
                if name not in self.capabilities:
                    raise WorkflowRegistryError(
                        f"{spec.id}.{goal.id}: unknown capability {name!r}"
                    )
                capability = self.capabilities.get(name)
                for component in sorted(capability.catalog):
                    if component not in self.operators:
                        raise WorkflowRegistryError(
                            f"{name}: catalog names an unregistered component: {component}"
                        )
                    if component in RUNTIME_ONLY:
                        raise WorkflowRegistryError(
                            f"{name}: {component} writes and is reserved to the runtime; "
                            f"a capability catalog must never contain it"
                        )
                for group in sorted(capability.config_groups):
                    if group not in groups:
                        raise WorkflowRegistryError(
                            f"{name}: unknown config group: {group}"
                        )

            for check in goal.gate.checks:
                if check not in CHECKS:
                    raise WorkflowRegistryError(
                        f"{spec.id}.{goal.id}: gate names an unknown check: {check}"
                    )

            # A goal whose gate requires evidence nothing can produce is unsatisfiable,
            # and the loop would spend its attempts discovering that at run time.
            for kind in goal.gate.requires:
                producers = {
                    name
                    for name in goal.capabilities
                    if kind in self.capabilities.get(name).produces
                }
                if not producers:
                    raise WorkflowRegistryError(
                        f"{spec.id}.{goal.id}: requires artifact {kind!r} but none of its "
                        f"capabilities {list(goal.capabilities)} produce it"
                    )

    def get(self, workflow_id: str) -> RegisteredWorkflow:
        try:
            return self._workflows[workflow_id]
        except KeyError as exc:
            raise WorkflowRegistryError(
                f"unknown workflow: {workflow_id}; registered: {sorted(self._workflows)}"
            ) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._workflows))

    def __contains__(self, workflow_id: object) -> bool:
        return workflow_id in self._workflows

    def routing_catalog(self) -> dict[str, dict[str, str]]:
        return {
            workflow_id: registered.card.model_dump(mode="json")
            for workflow_id, registered in sorted(self._workflows.items())
        }
