"""Load workflows and type-check their spines at registration.

A spine is programmer-authored, so its errors are build-time errors.  Checking contract
compatibility, stage resolution, catalog membership and config groups here means a
malformed workflow fails before a run starts rather than three stages in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..compose import catalog as hydra_catalog
from ..contracts import SemanticCard, StageSpec, WorkflowSpec, content_digest
from ..operators.builtin import RUNTIME_ONLY
from ..operators.planning import CHECKS
from ..registry import OperatorRegistry
from ..stages import StageRegistry

WORKFLOW_ROOT = Path(__file__).resolve().parents[2] / "_workflows"


class WorkflowRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RegisteredWorkflow:
    spec: WorkflowSpec
    card: SemanticCard
    stages: tuple[StageSpec, ...]

    def digest(self) -> str:
        """Identity of everything that determines the run's behaviour.

        Prompt versions are included, so promoting a compiled prompt changes the digest
        and `replay` can never silently use a newer prompt than the run did.
        """
        return content_digest(
            {
                "workflow": self.spec.model_dump(mode="json"),
                "stages": [stage.model_dump(mode="json") for stage in self.stages],
                "prompts": {
                    f"{stage.name}.{domain.id}": domain.prompt_version
                    for stage in self.stages
                    for domain in stage.domains
                },
            }
        )


class WorkflowRegistry:
    def __init__(
        self,
        *,
        stages: StageRegistry,
        operators: OperatorRegistry,
        root: Path | None = None,
        config_root: str | None = None,
    ) -> None:
        self.root = root or WORKFLOW_ROOT
        self.stages = stages
        self.operators = operators
        self.config_root = config_root
        self._workflows: dict[str, RegisteredWorkflow] = {}
        if self.root.is_dir():
            for manifest in sorted(self.root.glob("*/workflow.yml")):
                self._load_one(manifest)

    def _load_one(self, manifest: Path) -> None:
        directory = manifest.parent
        spec = WorkflowSpec.model_validate(yaml.safe_load(manifest.read_text()) or {})
        if spec.id != directory.name:
            raise WorkflowRegistryError(
                f"{manifest}: declares id {spec.id} but lives in {directory.name}"
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
        stages = self.type_check(spec)
        self._workflows[spec.id] = RegisteredWorkflow(spec=spec, card=card, stages=stages)

    def type_check(self, spec: WorkflowSpec) -> tuple[StageSpec, ...]:
        """Validate a spine end to end.  Raises on the first real problem."""
        stages: list[StageSpec] = []
        for ref in spec.spine:
            if ref not in self.stages:
                raise WorkflowRegistryError(
                    f"{spec.id}: spine pins {ref}, which is not a registered stage"
                )
            stages.append(self.stages.get(ref))

        expected = spec.entry_contract
        for stage in stages:
            if stage.input_contract != expected:
                raise WorkflowRegistryError(
                    f"{spec.id}: stage {stage.ref} expects {stage.input_contract!r} but the"
                    f" previous stage produces {expected!r}"
                )
            expected = stage.output_contract

        groups = hydra_catalog(self.config_root)
        for stage in stages:
            for obligation in stage.obligations:
                if obligation not in CHECKS and obligation not in _known_checkers(stage):
                    raise WorkflowRegistryError(
                        f"{stage.ref}: obligation {obligation!r} has no deterministic checker"
                    )
            for domain in stage.domains:
                for operator in sorted(domain.catalog):
                    if operator not in self.operators:
                        raise WorkflowRegistryError(
                            f"{stage.ref}.{domain.id}: catalog names an unregistered"
                            f" operator: {operator}"
                        )
                    if operator in RUNTIME_ONLY:
                        raise WorkflowRegistryError(
                            f"{stage.ref}.{domain.id}: {operator} writes and is reserved to"
                            f" the runtime; a domain catalog must never contain it"
                        )
                for group in sorted(domain.config_groups):
                    if group not in groups:
                        raise WorkflowRegistryError(
                            f"{stage.ref}.{domain.id}: unknown config group: {group}"
                        )
        return tuple(stages)

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
        """What the planner reads about workflows: the ten-field cards, nothing else."""
        return {
            workflow_id: registered.card.model_dump(mode="json")
            for workflow_id, registered in sorted(self._workflows.items())
        }


def _known_checkers(stage: StageSpec) -> frozenset[str]:
    """An obligation id may itself name a checker, which keeps simple stages terse."""
    return frozenset(CHECKS)
