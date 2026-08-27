"""The closest thing to a live run that works without an API key.

Everything is real except the model: ModelGoalPlanner and ModelCapabilityAgent run, real
prompts load and digest-check, real components execute, real files are committed. Only
the gateway is scripted.

This is what would catch a prompt whose declared response shape no longer matches its
contract — the failure a live smoke test exists to find.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.agent import ModelCapabilityAgent
from shakespeare.audit import AuditStore
from shakespeare.capabilities import CapabilityRegistry
from shakespeare.capabilities.runner import Organization
from shakespeare.contracts import ChangeAction, RequestContract, RouteDecision
from shakespeare.domain.filesystem import scan
from shakespeare.executor import Executor
from shakespeare.gateway import FakeGateway, ModelProfile
from shakespeare.operators.builtin import build_registry
from shakespeare.planner import CapabilityChoice, GoalChoice, Judgment, ModelGoalPlanner
from shakespeare.prompts import PromptStore
from shakespeare.runtime import Runtime
from shakespeare.telemetry import RecordingExporter, Tracer
from shakespeare.verifier import Verifier
from shakespeare.workflows import WorkflowRegistry

PROFILE = ModelProfile(profile_id="scripted", model="openrouter/openai/gpt-5-mini")

TREE = {
    "2024/q1/scan_0001.pdf": ("Northwind Traders", "INV-4471", "PO-88120", "2024-02-11"),
    "2024/q1/scan_0002.pdf": ("Northwind Traders", "INV-4472", "PO-88121", "2024-03-03"),
    "2024/q2/IMG_9931.pdf": ("Contoso Supply Co", "INV-10233", "PO-55004", "2024-05-19"),
}

SPEC = {
    "template": "{invoice_date}, {vendor}, {invoice_number}, {po_number}",
    "fields": [
        {"name": "invoice_date", "kind": "date", "format": "%Y%m"},
        {"name": "vendor"},
        {"name": "invoice_number"},
        {"name": "po_number", "required": False},
    ],
    "policy": {"separator": ", ", "max_length": 200},
    "collision_policy": "suffix_n",
}


def _organization(*invocations, publishes: str, intent: str = "") -> dict:
    return {
        "invocations": list(invocations),
        "intent": intent,
        "sufficient": True,
        "publishes": publishes,
        "quality": "complete",
        "summary": {},
    }


def _items(root: Path) -> list[dict]:
    resolved = []
    for item in scan(root)[0]:
        vendor, number, po, date = TREE[item.relpath]
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


def _script(gateway: FakeGateway, items: list[dict]) -> FakeGateway:
    gateway.queue(
        RouteDecision,
        {"workflow_id": "rename_files", "supported": True, "rationale": "renaming by content"},
    )
    # One ambiguous choice in the graph: readable and convention_frozen open together.
    gateway.queue(GoalChoice, {"goal_id": "readable", "rationale": "text first"})
    gateway.queue(CapabilityChoice, {"capability_id": "acquire", "rationale": "reads files"})
    gateway.queue(
        Organization,
        _organization(
            {"invocation_id": "scan", "operator": "fs.scan", "inputs": ["root"]},
            publishes="FileInventory",
            intent="walk the tree",
        ),
        _organization(
            {
                "invocation_id": "extract",
                "operator": "doc.extract",
                "selections": {"extract": "auto_chain"},
                "inputs": ["root", "items"],
            },
            publishes="ExtractedContent",
            intent="read them",
        ),
        _organization(
            {"invocation_id": "freeze", "operator": "spec.freeze", "parameters": {"spec": SPEC}},
            publishes="NamingSpec",
            intent="freeze the convention",
        ),
        _organization(
            {
                "invocation_id": "render",
                "operator": "name.render",
                "inputs": ["spec"],
                "parameters": {"items": items},
            },
            publishes="ResolvedNames",
            intent="render names",
        ),
        _organization(
            {
                "invocation_id": "collide",
                "operator": "name.collide",
                "selections": {"collision": "suffix_n"},
                "inputs": ["candidates", "unrendered"],
            },
            {
                "invocation_id": "assemble",
                "operator": "plan.assemble",
                "inputs": [
                    "run_id",
                    "workflow_id",
                    "workflow_digest",
                    "items",
                    "skipped",
                    "collide",
                ],
                "bindings": {"scanned": "items", "planned": "resolutions"},
                "parameters": {"decision_digest": "frozen-spec"},
            },
            publishes="ChangePlan",
            intent="assemble the plan",
        ),
        _organization(
            {
                "invocation_id": "verify",
                "operator": "fs.verify",
                "inputs": ["plan", "staging_root"],
            },
            publishes="ReviewEvidence",
            intent="verify staging",
        ),
    )
    # Three gates in this graph ask for judgment: readable, named and reviewed. The
    # fake gateway consumes one response per call, so each needs its own.
    judgment = {"satisfied": True, "rationale": "nothing more would change the decision"}
    gateway.queue(Judgment, judgment, judgment, judgment)
    return gateway


@pytest.fixture
def scripted(tmp_path: Path):
    source = tmp_path / "invoices"
    for relpath in TREE:
        path = source / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"scanned invoice: {relpath}".encode())

    gateway = _script(FakeGateway(), _items(source))
    prompts = PromptStore()
    operators = build_registry()
    verifier = Verifier(operators)
    recorder = RecordingExporter()
    tracer = Tracer("e2e", [recorder])
    audit = AuditStore(tmp_path / "audit.sqlite3")
    capabilities = CapabilityRegistry()

    runtime = Runtime(
        operators=operators,
        capabilities=capabilities,
        workflows=WorkflowRegistry(capabilities=capabilities, operators=operators),
        verifier=verifier,
        executor=Executor(operators, verifier, tracer=tracer),
        planner=ModelGoalPlanner(gateway=gateway, profile=PROFILE, prompts=prompts),
        agents={"*": ModelCapabilityAgent(gateway=gateway, profile=PROFILE, prompts=prompts)},
        audit=audit,
        workspace_root=tmp_path / "work",
        tracer=tracer,
    )
    request = RequestContract(
        request_id="e2e",
        prompt="rename these invoices to YYYYMM, vendor, invoice number, PO number",
        input_root=str(source),
        output_root=str(tmp_path / "renamed"),
    )
    yield runtime, request, audit, recorder, gateway
    audit.close()


class TestFullRun:
    def test_commits_the_expected_tree(self, scripted) -> None:
        runtime, request, _, _, _ = scripted
        result = runtime.run(request)
        assert result.outcome == "committed", result.detail

        output = Path(request.output_root)
        assert sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        ) == [
            "2024/q1/202402, Northwind Traders, INV-4471, PO-88120.pdf",
            "2024/q1/202403, Northwind Traders, INV-4472, PO-88121.pdf",
            "2024/q2/202405, Contoso Supply Co, INV-10233, PO-55004.pdf",
        ]

    def test_every_goal_was_satisfied(self, scripted) -> None:
        runtime, request, _, _, _ = scripted
        result = runtime.run(request)
        assert len(result.satisfied) == 6
        assert all(attempt.gate.satisfied for attempt in result.attempts)

    def test_accounting_balances(self, scripted) -> None:
        runtime, request, _, _, _ = scripted
        result = runtime.run(request)
        assert result.plan is not None
        assert result.plan.balanced(3)
        assert result.plan.count(ChangeAction.CHANGED) == 3

    def test_source_tree_is_untouched(self, scripted) -> None:
        runtime, request, _, _, _ = scripted
        before = sorted(p.name for p in Path(request.input_root).rglob("*") if p.is_file())
        runtime.run(request)
        assert sorted(
            p.name for p in Path(request.input_root).rglob("*") if p.is_file()
        ) == before == ["IMG_9931.pdf", "scan_0001.pdf", "scan_0002.pdf"]

    def test_no_content_reaches_telemetry(self, scripted) -> None:
        runtime, request, _, recorder, _ = scripted
        runtime.run(request)
        shipped = recorder.serialized()
        for secret in ("Northwind", "Contoso", "INV-4471", "PO-88120"):
            assert secret not in shipped

    def test_every_goal_attempt_is_auditable(self, scripted) -> None:
        runtime, request, audit, _, _ = scripted
        result = runtime.run(request)
        for goal_id in result.satisfied:
            dag = audit.dag(result.run_id, goal_id)
            assert dag["attempts"], goal_id

    def test_the_model_is_called_only_where_judgment_is_needed(self, scripted) -> None:
        """One route, one goal choice, one capability choice, one organization per
        capability, and one judgment per semantic gate — nothing else."""
        runtime, request, _, _, gateway = scripted
        runtime.run(request)
        kinds = [name for name, _ in gateway.calls]
        assert kinds.count("RouteDecision") == 1
        assert kinds.count("Organization") == 6
        # Only one point in the graph has two open goals, and only some gates ask.
        assert kinds.count("GoalChoice") == 1
        assert kinds.count("CapabilityChoice") <= 1
        assert kinds.count("Judgment") >= 1
