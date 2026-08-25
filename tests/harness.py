"""Shared harness for the goal-driven runtime.

One place that knows how to assemble a runtime with scripted capabilities, so the tests
below it are about behaviour rather than wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shakespeare.agent import FakeCapabilityAgent
from shakespeare.artifacts import Quality
from shakespeare.audit import AuditStore
from shakespeare.capabilities import CapabilityRegistry
from shakespeare.capabilities.runner import Organization
from shakespeare.contracts import Invocation, RequestContract, RouteDecision, SemanticCard
from shakespeare.executor import Executor
from shakespeare.operators.builtin import build_registry
from shakespeare.planner import ScriptedGoalPlanner
from shakespeare.runtime import Runtime
from shakespeare.telemetry import RecordingExporter, Tracer
from shakespeare.verifier import Verifier
from shakespeare.workflows import WorkflowRegistry

INVOICES: dict[str, tuple[str, str, str, str]] = {
    "2024/q1/scan001.pdf": ("ACME Corporation", "INV-99812", "PO-44117", "2024-01-15"),
    "2024/q1/scan002.pdf": ("ACME Corporation", "INV-99813", "PO-44118", "2024-01-22"),
    "2024/q2/scan003.pdf": ("Globex Ltd", "INV-20001", "PO-77310", "2024-04-02"),
}

SPEC: dict[str, Any] = {
    "template": "{invoice_date}, {vendor}, {invoice_number}, {po_number}",
    "fields": [
        {"name": "invoice_date", "kind": "date", "format": "%Y%m"},
        {"name": "vendor"},
        {"name": "invoice_number"},
        {"name": "po_number", "required": False},
    ],
    "policy": {"separator": ", "},
    "collision_policy": "suffix_n",
}


def card(purpose: str) -> SemanticCard:
    filler = "declared for the test harness"
    return SemanticCard(
        purpose=purpose, lifecycle=filler, contracts=filler, allowed_configuration=filler,
        side_effects=filler, risks=filler, failure_modes=filler, resource_limits=filler,
        examples=filler, provenance=filler,
    )


def seed_invoices(root: Path, contents: dict[str, tuple[str, str, str, str]] | None = None) -> Path:
    for relpath in contents or INVOICES:
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"invoice body for {relpath}".encode())
    return root


def values_for(
    root: Path, contents: dict[str, tuple[str, str, str, str]] | None = None
) -> list[dict[str, Any]]:
    """Per-item field values, computed against the bytes the run will actually see.

    Item ids are content-addressed, so these must be built after the tree is seeded.
    """
    from shakespeare.operators.filesystem import scan

    contents = contents or INVOICES
    resolved: list[dict[str, Any]] = []
    for item in scan(root)[0]:
        vendor, number, po, date = contents[item.relpath]
        resolved.append(
            {
                "item_id": item.item_id,
                "directory": str(Path(item.relpath).parent),
                "extension": Path(item.relpath).suffix,
                "values": {
                    "vendor": vendor,
                    "invoice_number": number,
                    "po_number": po,
                    "invoice_date": date,
                },
            }
        )
    return resolved


def org(
    *invocations: Invocation,
    publishes: str | None = None,
    quality: Quality = Quality.COMPLETE,
    summary: dict[str, Any] | None = None,
    intent: str = "",
    sufficient: bool = True,
) -> Organization:
    return Organization(
        invocations=invocations,
        intent=intent,
        sufficient=sufficient,
        publishes=publishes,
        quality=quality,
        summary=summary or {},
    )


def rename_agent(items: list[dict[str, Any]], spec: dict[str, Any] | None = None):
    """Scripted capabilities that answer every goal in the rename graph."""
    agent = FakeCapabilityAgent()
    agent.queue(
        "survey",
        org(
            Invocation(invocation_id="scan", operator="fs.scan", inputs=("root",)),
            publishes="FileInventory",
            intent="walk the tree",
        ),
    )
    agent.queue(
        "acquire",
        org(
            Invocation(
                invocation_id="extract",
                operator="doc.extract",
                selections={"extract": "auto_chain"},
                inputs=("root", "items"),
            ),
            publishes="ExtractedContent",
            intent="read every file",
        ),
    )
    agent.queue(
        "convene",
        org(
            Invocation(
                invocation_id="freeze",
                operator="spec.freeze",
                parameters={"spec": spec or SPEC},
            ),
            publishes="NamingSpec",
            intent="freeze the convention",
        ),
    )
    agent.queue(
        "resolve",
        org(
            Invocation(
                invocation_id="render",
                operator="name.render",
                inputs=("spec",),
                parameters={"items": items},
            ),
            publishes="ResolvedNames",
            intent="render names",
        ),
    )
    agent.queue(
        "compose",
        org(
            Invocation(
                invocation_id="collide",
                operator="name.collide",
                selections={"collision": "suffix_n"},
                inputs=("candidates", "unrendered"),
            ),
            Invocation(
                invocation_id="assemble",
                operator="plan.assemble",
                inputs=(
                    "run_id",
                    "workflow_id",
                    "workflow_digest",
                    "items",
                    "skipped",
                    "collide",
                ),
                bindings={"scanned": "items", "planned": "resolutions"},
                parameters={"decision_digest": "spec"},
            ),
            publishes="ChangePlan",
            intent="assemble the plan",
        ),
    )
    agent.queue(
        "review",
        org(
            Invocation(
                invocation_id="verify", operator="fs.verify", inputs=("plan", "staging_root")
            ),
            publishes="ReviewEvidence",
            intent="verify staging",
        ),
    )
    return agent


def build(
    tmp_path: Path,
    *,
    contents: dict[str, tuple[str, str, str, str]] | None = None,
    planner: Any | None = None,
    agents: dict[str, Any] | None = None,
    spec: dict[str, Any] | None = None,
) -> tuple[Runtime, RequestContract, AuditStore, RecordingExporter]:
    source = seed_invoices(tmp_path / "in", contents)
    operators = build_registry()
    verifier = Verifier(operators)
    capabilities = CapabilityRegistry()
    recorder = RecordingExporter()
    tracer = Tracer("harness", [recorder])
    audit = AuditStore(tmp_path / "audit.sqlite3")

    runtime = Runtime(
        operators=operators,
        capabilities=capabilities,
        workflows=WorkflowRegistry(capabilities=capabilities, operators=operators),
        verifier=verifier,
        executor=Executor(operators, verifier, tracer=tracer),
        planner=planner or ScriptedGoalPlanner(route=RouteDecision(workflow_id="rename_files")),
        agents=agents
        if agents is not None
        else {"*": rename_agent(values_for(source, contents), spec)},
        audit=audit,
        workspace_root=tmp_path / "work",
        tracer=tracer,
    )
    request = RequestContract(
        request_id="req-1",
        prompt="rename these invoices to YYYYMM, vendor, invoice number, PO number",
        input_root=str(source),
        output_root=str(tmp_path / "out"),
    )
    return runtime, request, audit, recorder
