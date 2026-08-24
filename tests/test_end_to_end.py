"""The closest thing to a live run that works without an API key.

Everything is real except the model itself: ModelPlanner and ModelDomainAgent run, real
prompts load and digest-check, real operators execute, and real files are committed. Only
the gateway is scripted.

This is what would catch a prompt whose declared response shape no longer matches its
contract — the failure a live smoke test exists to find.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.agent import CompositionDraft, ModelDomainAgent
from shakespeare.audit import AuditStore
from shakespeare.contracts import (
    ChangeAction,
    RequestContract,
    RouteDecision,
    StageDecision,
    StagePlan,
    StageVerdict,
)
from shakespeare.executor import Executor
from shakespeare.gateway import FakeGateway, ModelProfile
from shakespeare.operators.builtin import build_registry
from shakespeare.operators.filesystem import scan
from shakespeare.planner import ModelPlanner
from shakespeare.prompts import PromptStore
from shakespeare.runtime import Runtime
from shakespeare.stages import StageRegistry
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


def _draft(*invocations: dict[str, object], rationale: str = "") -> dict[str, object]:
    return {"invocations": list(invocations), "rationale": rationale}


def _plan(*domain_ids: str, skipped: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "activated": [
            {
                "domain_id": domain_id,
                "goal": f"complete the {domain_id} work for this request",
                "success_criterion": "the stage obligations pass",
                "obligation_refs": [],
            }
            for domain_id in domain_ids
        ],
        "skipped": [
            {"domain_id": domain_id, "reason": "not required for this request"}
            for domain_id in skipped
        ],
    }


ACCEPT = {"met": True, "decision": "accept", "unmet": [], "revised_goals": [], "rationale": "ok"}


def _items(root: Path) -> list[dict[str, object]]:
    scanned, _ = scan(root)
    output = []
    for item in scanned:
        vendor, number, po, date = TREE[item.relpath]
        output.append(
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
    return output


def _script(gateway: FakeGateway, items: list[dict[str, object]]) -> FakeGateway:
    gateway.queue(
        RouteDecision,
        {"workflow_id": "rename_files", "supported": True, "rationale": "renaming by content"},
    )
    gateway.queue(
        StagePlan,
        _plan("file_validity", "safety_preflight"),
        _plan("content_acquisition"),
        _plan("convention_design"),
        _plan("field_resolution"),
        _plan("change_composition"),
        _plan("structural_review", "exception_review"),
    )
    gateway.queue(
        CompositionDraft,
        _draft({"invocation_id": "scan", "operator": "fs.scan", "inputs": ["root"]}),
        _draft({"invocation_id": "dirs", "operator": "fs.dirs", "inputs": ["root"]}),
        _draft(
            {
                "invocation_id": "extract",
                "operator": "doc.extract",
                "selections": {"extract": "auto_chain"},
                "inputs": ["root", "items"],
            }
        ),
        _draft(
            {"invocation_id": "freeze", "operator": "spec.freeze", "parameters": {"spec": SPEC}}
        ),
        _draft(
            {
                "invocation_id": "render",
                "operator": "name.render",
                "inputs": ["spec"],
                "parameters": {"items": items},
            }
        ),
        _draft(
            {
                "invocation_id": "collide",
                "operator": "name.collide",
                "selections": {"collision": "suffix_n"},
                "inputs": ["candidates", "unrendered"],
            },
            {
                "invocation_id": "assemble",
                "operator": "plan.assemble",
                "inputs": ["run_id", "workflow_id", "workflow_digest", "items", "collide"],
                "bindings": {"scanned": "items", "planned": "resolutions"},
                "parameters": {"decision_digest": "frozen-spec"},
            },
        ),
        _draft(
            {"invocation_id": "verify", "operator": "fs.verify", "inputs": ["plan", "staging_root"]}
        ),
        _draft({"invocation_id": "dirs", "operator": "fs.dirs", "inputs": ["staging_root"]}),
    )
    gateway.queue(StageVerdict, *([ACCEPT] * 6))
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
    stages = StageRegistry()

    runtime = Runtime(
        operators=operators,
        stages=stages,
        workflows=WorkflowRegistry(stages=stages, operators=operators),
        verifier=verifier,
        executor=Executor(operators, verifier, tracer=tracer),
        planner=ModelPlanner(gateway=gateway, profile=PROFILE, prompts=prompts),
        agents={"*": ModelDomainAgent(gateway=gateway, profile=PROFILE, prompts=prompts)},
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
        produced = sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        )
        assert produced == [
            "2024/q1/202402, Northwind Traders, INV-4471, PO-88120.pdf",
            "2024/q1/202403, Northwind Traders, INV-4472, PO-88121.pdf",
            "2024/q2/202405, Contoso Supply Co, INV-10233, PO-55004.pdf",
        ]

    def test_every_stage_was_accepted_on_its_first_attempt(self, scripted) -> None:
        runtime, request, _, _, _ = scripted
        result = runtime.run(request)
        assert len(result.stages) == 6
        for outcome in result.stages:
            assert outcome.verdict.decision is StageDecision.ACCEPT, outcome.stage.name
            assert outcome.attempts == 1, outcome.stage.name

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
        after = sorted(p.name for p in Path(request.input_root).rglob("*") if p.is_file())
        assert after == before == ["IMG_9931.pdf", "scan_0001.pdf", "scan_0002.pdf"]

    def test_no_content_reaches_telemetry(self, scripted) -> None:
        runtime, request, _, recorder, _ = scripted
        runtime.run(request)
        shipped = recorder.serialized()
        for secret in ("Northwind", "Contoso", "INV-4471", "PO-88120"):
            assert secret not in shipped

    def test_the_full_dag_is_auditable(self, scripted) -> None:
        runtime, request, audit, _, _ = scripted
        result = runtime.run(request)
        for stage in ("intake", "extract", "convention", "resolve", "compose_changes", "review"):
            dag = audit.dag(result.run_id, stage)
            assert dag["attempts"], stage
            assert dag["attempts"][0]["nodes"], f"{stage} recorded no invocations"

    def test_the_model_was_called_only_at_the_declared_points(self, scripted) -> None:
        """Two planner calls per stage plus one per activated domain, and nothing else."""
        runtime, request, _, _, gateway = scripted
        runtime.run(request)
        kinds = [name for name, _ in gateway.calls]
        assert kinds.count("RouteDecision") == 1
        assert kinds.count("StagePlan") == 6
        assert kinds.count("StageVerdict") == 6
        assert kinds.count("CompositionDraft") == 8
