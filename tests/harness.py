"""Shared runtime harness for the spine tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shakespeare.audit import AuditStore
from shakespeare.contracts import (
    Composition,
    DomainGoal,
    DomainSpec,
    Invocation,
    SemanticCard,
    StageSpec,
    WorkflowSpec,
)
from shakespeare.executor import Executor
from shakespeare.operators.builtin import build_registry
from shakespeare.planner import FakePlanner
from shakespeare.runtime import Runtime
from shakespeare.stages import StageRegistry
from shakespeare.telemetry import RecordingExporter, Tracer
from shakespeare.verifier import Verifier
from shakespeare.workflows import WorkflowRegistry


def card(purpose: str) -> SemanticCard:
    filler = "declared for the test harness"
    return SemanticCard(
        purpose=purpose,
        lifecycle=filler,
        contracts=filler,
        allowed_configuration=filler,
        side_effects=filler,
        risks=filler,
        failure_modes=filler,
        resource_limits=filler,
        examples=filler,
        provenance=filler,
    )


def goal(domain_id: str, text: str = "do the work") -> DomainGoal:
    return DomainGoal(domain_id=domain_id, goal=text, success_criterion="obligations pass")


def composition(domain_id: str, *invocations: Invocation) -> Composition:
    return Composition(domain_id=domain_id, invocations=invocations)


def build_runtime(
    tmp_path: Path,
    *,
    stages: list[tuple[StageSpec, SemanticCard]],
    workflow: WorkflowSpec,
    planner: FakePlanner,
    agents: dict[str, Any],
    exporter: RecordingExporter | None = None,
) -> tuple[Runtime, AuditStore, RecordingExporter]:
    operators = build_registry()
    stage_registry = StageRegistry(root=tmp_path / "_stages_empty")
    for spec, semantic in stages:
        stage_registry.register(spec, semantic)

    workflow_registry = WorkflowRegistry(
        stages=stage_registry, operators=operators, root=tmp_path / "_workflows_empty"
    )
    workflow_registry.register(workflow, card(f"{workflow.id} test workflow"))

    recorder = exporter or RecordingExporter()
    tracer = Tracer("harness", [recorder])
    verifier = Verifier(operators)
    audit = AuditStore(tmp_path / "audit.sqlite3")

    runtime = Runtime(
        operators=operators,
        stages=stage_registry,
        workflows=workflow_registry,
        verifier=verifier,
        executor=Executor(operators, verifier, tracer=tracer),
        planner=planner,
        agents=agents,
        audit=audit,
        workspace_root=tmp_path / "work",
        tracer=tracer,
    )
    return runtime, audit, recorder


def seed_tree(root: Path) -> Path:
    """A nested tree with the awkward cases: subfolders, an unreadable type, a collision."""
    (root / "2024" / "q1").mkdir(parents=True)
    (root / "2024" / "q1" / "scan001.pdf").write_bytes(b"invoice one")
    (root / "2024" / "scan002.pdf").write_bytes(b"invoice two")
    (root / "notes.txt").write_text("a loose note")
    return root


def noop_stages() -> list[tuple[StageSpec, SemanticCard]]:
    """A workflow that shares nothing with rename_files except the spine itself."""
    inventory = StageSpec(
        name="inventory",
        version="1.0.0",
        purpose="Inventory the input tree.",
        goal="Every file is inventoried.",
        input_contract="RequestContract",
        output_contract="FileInventory",
        domains=(
            DomainSpec(
                id="survey",
                scope="Walk the input root.",
                skippable=False,
                catalog=frozenset({"fs.scan"}),
            ),
        ),
    )
    compose_changes = StageSpec(
        name="compose_changes",
        version="1.0.0",
        purpose="Plan a pure passthrough copy.",
        goal="Every item has a plan entry.",
        input_contract="FileInventory",
        output_contract="ChangePlan",
        domains=(
            DomainSpec(
                id="passthrough",
                scope="Plan each file unchanged.",
                skippable=False,
                catalog=frozenset({"plan.assemble"}),
            ),
        ),
        obligations=("balanced", "resolved_or_quarantined"),
    )
    return [(inventory, card("inventory")), (compose_changes, card("compose"))]


def noop_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        id="noop_passthrough",
        version="1.0.0",
        spine=("inventory@1.0.0", "compose_changes@1.0.0"),
        commit_after="compose_changes",
    )
