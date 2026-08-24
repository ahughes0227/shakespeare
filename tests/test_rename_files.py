"""The rename_files workflow end to end, offline.

Every model touchpoint is faked, so this exercises the real stages, real operators and
the real two-phase commit without a network or an API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.agent import FakeDomainAgent
from shakespeare.audit import AuditStore
from shakespeare.contracts import (
    ChangeAction,
    Composition,
    Invocation,
    RequestContract,
    RouteDecision,
    StagePlan,
)
from shakespeare.executor import Executor
from shakespeare.operators.builtin import build_registry
from shakespeare.planner import FakePlanner
from shakespeare.runtime import Runtime
from shakespeare.stages import StageRegistry
from shakespeare.telemetry import RecordingExporter, Tracer
from shakespeare.verifier import Verifier
from shakespeare.workflows import WorkflowRegistry

from harness import goal

INVOICES = {
    "2024/q1/scan001.pdf": ("ACME Corporation", "INV-99812", "PO-44117", "2024-01-15"),
    "2024/q1/scan002.pdf": ("ACME Corporation", "INV-99813", "PO-44118", "2024-01-22"),
    "2024/q2/scan003.pdf": ("Globex Ltd", "INV-20001", "PO-77310", "2024-04-02"),
}

SPEC = {
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


def seed_invoices(root: Path, extra: dict[str, tuple[str, str, str, str]] | None = None) -> Path:
    for relpath in {**INVOICES, **(extra or {})}:
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"invoice body for {relpath}".encode())
    return root


def _values(root: Path, contents: dict[str, tuple[str, str, str, str]]) -> list[dict[str, object]]:
    from shakespeare.operators.filesystem import scan

    items, _ = scan(root)
    resolved: list[dict[str, object]] = []
    for item in items:
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


def build_agents(items: list[dict[str, object]]) -> dict[str, FakeDomainAgent]:
    """Fake agents that emit real compositions over the real operators."""
    return {
        "file_validity": FakeDomainAgent().queue(
            "file_validity",
            Composition(
                domain_id="file_validity",
                invocations=(
                    Invocation(invocation_id="scan", operator="fs.scan", inputs=("root",)),
                ),
            ),
        ),
        "safety_preflight": FakeDomainAgent().queue(
            "safety_preflight",
            Composition(
                domain_id="safety_preflight",
                invocations=(
                    Invocation(invocation_id="dirs", operator="fs.dirs", inputs=("root",)),
                ),
            ),
        ),
        "content_acquisition": FakeDomainAgent().queue(
            "content_acquisition",
            Composition(
                domain_id="content_acquisition",
                invocations=(
                    Invocation(
                        invocation_id="extract",
                        operator="doc.extract",
                        selections={"extract": "auto_chain"},
                        inputs=("root", "items"),
                    ),
                ),
            ),
        ),
        "convention_design": FakeDomainAgent().queue(
            "convention_design",
            Composition(
                domain_id="convention_design",
                invocations=(
                    Invocation(
                        invocation_id="freeze",
                        operator="spec.freeze",
                        parameters={"spec": SPEC},
                    ),
                ),
            ),
        ),
        "field_resolution": FakeDomainAgent().queue(
            "field_resolution",
            Composition(
                domain_id="field_resolution",
                invocations=(
                    Invocation(
                        invocation_id="render",
                        operator="name.render",
                        inputs=("spec",),
                        parameters={"items": items},
                    ),
                ),
            ),
        ),
        "change_composition": FakeDomainAgent().queue(
            "change_composition",
            Composition(
                domain_id="change_composition",
                invocations=(
                    Invocation(
                        invocation_id="collide",
                        operator="name.collide",
                        selections={"collision": "suffix_n"},
                        inputs=("candidates", "unrendered"),
                    ),
                    Invocation(
                        invocation_id="assemble",
                        operator="plan.assemble",
                        inputs=("run_id", "workflow_id", "workflow_digest", "items", "collide"),
                        bindings={"scanned": "items", "planned": "resolutions"},
                        parameters={"decision_digest": "spec"},
                    ),
                ),
            ),
        ),
        "structural_review": FakeDomainAgent().queue(
            "structural_review",
            Composition(
                domain_id="structural_review",
                invocations=(
                    Invocation(
                        invocation_id="verify",
                        operator="fs.verify",
                        inputs=("plan", "staging_root"),
                    ),
                ),
            ),
        ),
        "exception_review": FakeDomainAgent().queue(
            "exception_review",
            Composition(
                domain_id="exception_review",
                invocations=(
                    Invocation(
                        invocation_id="dirs", operator="fs.dirs", inputs=("staging_root",)
                    ),
                ),
            ),
        ),
    }


def build_planner(*, skip_resolution: bool = False) -> FakePlanner:
    planner = FakePlanner(route=RouteDecision(workflow_id="rename_files"))
    planner.queue_plan(
        "intake", StagePlan(activated=(goal("file_validity"), goal("safety_preflight")))
    )
    planner.queue_plan("extract", StagePlan(activated=(goal("content_acquisition"),)))
    planner.queue_plan("convention", StagePlan(activated=(goal("convention_design"),)))
    if skip_resolution:
        from shakespeare.contracts import SkipDecision

        planner.queue_plan(
            "resolve",
            StagePlan(
                skipped=(
                    SkipDecision(
                        domain_id="field_resolution",
                        reason="sequential convention needs no per-file extraction",
                    ),
                )
            ),
        )
    else:
        planner.queue_plan("resolve", StagePlan(activated=(goal("field_resolution"),)))
    planner.queue_plan("compose_changes", StagePlan(activated=(goal("change_composition"),)))
    planner.queue_plan(
        "review", StagePlan(activated=(goal("structural_review"), goal("exception_review")))
    )
    return planner


def build(tmp_path: Path, *, contents=INVOICES, planner=None, agents=None):
    source = seed_invoices(tmp_path / "in", None if contents is INVOICES else contents)
    request = RequestContract(
        request_id="req-1",
        prompt="rename these invoices to YYYYMM, vendor, invoice number, PO number",
        input_root=str(source),
        output_root=str(tmp_path / "out"),
    )
    operators = build_registry()
    verifier = Verifier(operators)
    recorder = RecordingExporter()
    tracer = Tracer("rename", [recorder])
    audit = AuditStore(tmp_path / "audit.sqlite3")
    stages = StageRegistry()
    runtime = Runtime(
        operators=operators,
        stages=stages,
        workflows=WorkflowRegistry(stages=stages, operators=operators),
        verifier=verifier,
        executor=Executor(operators, verifier, tracer=tracer),
        planner=planner or build_planner(),
        agents=agents if agents is not None else build_agents(_values(source, contents)),
        audit=audit,
        workspace_root=tmp_path / "work",
        tracer=tracer,
    )
    return runtime, request, audit, recorder


@pytest.fixture
def invoices(tmp_path: Path):
    runtime, request, audit, recorder = build(tmp_path)
    yield runtime, request, audit, recorder
    audit.close()


class TestInvoiceRename:
    def test_renames_by_content_preserving_structure(self, invoices, tmp_path: Path) -> None:
        runtime, request, _, _ = invoices
        result = runtime.run(request)
        assert result.outcome == "committed", result.detail

        output = Path(request.output_root)
        produced = sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        )
        assert produced == [
            "2024/q1/202401, ACME Corporation, INV-99812, PO-44117.pdf",
            "2024/q1/202401, ACME Corporation, INV-99813, PO-44118.pdf",
            "2024/q2/202404, Globex Ltd, INV-20001, PO-77310.pdf",
        ]

    def test_source_tree_is_untouched(self, invoices, tmp_path: Path) -> None:
        runtime, request, _, _ = invoices
        before = sorted(
            p.relative_to(request.input_root).as_posix()
            for p in Path(request.input_root).rglob("*")
            if p.is_file()
        )
        runtime.run(request)
        after = sorted(
            p.relative_to(request.input_root).as_posix()
            for p in Path(request.input_root).rglob("*")
            if p.is_file()
        )
        assert after == before

    def test_plan_is_balanced_and_reproducible(self, tmp_path: Path) -> None:
        first, request_a, audit_a, _ = build(tmp_path / "a")
        second, request_b, audit_b, _ = build(tmp_path / "b")
        plan_a = first.run(request_a, commit=False).plan
        plan_b = second.run(request_b, commit=False).plan
        assert plan_a is not None and plan_b is not None
        assert plan_a.balanced(3)
        # Digests differ only by run id, so compare the decision content itself.
        assert [e.target_relpath for e in plan_a.entries] == [
            e.target_relpath for e in plan_b.entries
        ]
        audit_a.close()
        audit_b.close()

    def test_no_document_content_reaches_telemetry(self, invoices) -> None:
        runtime, request, _, recorder = invoices
        runtime.run(request)
        shipped = recorder.serialized()
        for secret in ("ACME Corporation", "INV-99812", "PO-44117", "Globex"):
            assert secret not in shipped, f"leaked into telemetry: {secret}"


class TestCollisions:
    def test_two_identical_invoices_are_suffixed_not_lost(self, tmp_path: Path) -> None:
        contents = {
            **INVOICES,
            "2024/q1/duplicate.pdf": ("ACME Corporation", "INV-99812", "PO-44117", "2024-01-15"),
        }
        runtime, request, audit, _ = build(tmp_path, contents=contents)
        result = runtime.run(request)
        assert result.outcome == "committed", result.detail
        output = Path(request.output_root)
        names = sorted(p.name for p in (output / "2024" / "q1").glob("*.pdf"))
        assert names == [
            "202401, ACME Corporation, INV-99812, PO-44117 (2).pdf",
            "202401, ACME Corporation, INV-99812, PO-44117.pdf",
            "202401, ACME Corporation, INV-99813, PO-44118.pdf",
        ]
        assert result.plan is not None and result.plan.balanced(4)
        audit.close()


SEQUENTIAL_SPEC = {
    "template": "{seq:04d}",
    "fields": [{"name": "seq", "kind": "sequence", "format": "04d"}],
    "policy": {},
    "collision_policy": "suffix_n",
}


class TestSequentialRename:
    """A convention that needs nothing read from the documents.

    The earlier version of this test only asserted "no model calls", which is trivially
    true when a run aborts — and it was aborting. It now asserts the run actually
    produces sequentially named files.
    """

    def _agents(self) -> dict[str, FakeDomainAgent]:
        agents = build_agents([])
        agents["convention_design"] = FakeDomainAgent().queue(
            "convention_design",
            Composition(
                domain_id="convention_design",
                invocations=(
                    Invocation(
                        invocation_id="freeze",
                        operator="spec.freeze",
                        parameters={"spec": SEQUENTIAL_SPEC},
                    ),
                ),
            ),
        )
        agents["field_resolution"] = FakeDomainAgent().queue(
            "field_resolution",
            Composition(
                domain_id="field_resolution",
                invocations=(
                    # No parameters at all: the renderer derives every item from the
                    # inventory and fills {seq} from the deterministic scan order.
                    Invocation(
                        invocation_id="render",
                        operator="name.render",
                        inputs=("spec", "items"),
                    ),
                ),
            ),
        )
        return agents

    def test_produces_sequential_names(self, tmp_path: Path) -> None:
        agents = self._agents()
        runtime, request, audit, _ = build(tmp_path, agents=agents)
        result = runtime.run(request)
        assert result.outcome == "committed", result.detail

        output = Path(request.output_root)
        produced = sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        )
        assert produced == ["2024/q1/0001.pdf", "2024/q1/0002.pdf", "2024/q2/0003.pdf"]
        audit.close()

    def test_needs_no_per_item_transcription(self, tmp_path: Path) -> None:
        """The agent emits one invocation regardless of how many files there are."""
        agents = self._agents()
        runtime, request, audit, _ = build(tmp_path, agents=agents)
        runtime.run(request)
        assert agents["field_resolution"].call_count == 1
        audit.close()


class TestFailureContainment:
    def test_a_failed_review_leaves_no_output_root(self, tmp_path: Path) -> None:
        runtime, request, audit, _ = build(tmp_path)
        original = runtime.executor.execute

        def corrupt(composition, domain, **kwargs):
            results = original(composition, domain, **kwargs)
            if domain.id == "structural_review":
                staging = Path(kwargs["stage_inputs"]["staging_root"])
                for path in staging.rglob("*.pdf"):
                    path.write_bytes(b"tampered")
                    break
            return results

        runtime.executor.execute = corrupt  # type: ignore[method-assign]
        result = runtime.run(request)
        assert result.outcome == "aborted"
        assert result.error_code is not None
        assert not Path(request.output_root).exists(), "a failed run must leave nothing behind"
        audit.close()

    def test_agent_cannot_hand_write_a_filename(self, tmp_path: Path) -> None:
        """Names must flow from the renderer, never be invented by an agent."""
        agents = build_agents(_values(seed_invoices(tmp_path / "in"), INVOICES))
        agents["change_composition"] = FakeDomainAgent().queue(
            "change_composition",
            Composition(
                domain_id="change_composition",
                invocations=(
                    Invocation(
                        invocation_id="assemble",
                        operator="plan.assemble",
                        inputs=("run_id", "workflow_id", "workflow_digest", "items"),
                        bindings={"scanned": "items"},
                        parameters={
                            "decision_digest": "forged",
                            "planned": [{"item_id": "x", "directory": "", "name": "mine.pdf"}],
                        },
                    ),
                ),
            ),
        )
        runtime, request, audit, _ = build(tmp_path, agents=agents)
        result = runtime.run(request)
        assert result.outcome == "aborted"
        assert not Path(request.output_root).exists()
        audit.close()


class TestQuarantine:
    def test_unresolvable_item_is_quarantined_under_its_original_name(
        self, tmp_path: Path
    ) -> None:
        """An item whose fields did not resolve is never given a guessed name."""
        # Seed first: item ids are content-addressed, so values must be computed against
        # the bytes the run will actually see.
        contents = {**INVOICES, "mystery.pdf": ("", "", "", "2024-01-01")}
        source = seed_invoices(tmp_path / "in", contents)
        items = _values(source, contents)
        for entry in items:
            if entry["directory"] == ".":
                entry["values"] = {}  # type: ignore[index]
        runtime, request, audit, _ = build(
            tmp_path, contents=contents, agents=build_agents(items)
        )
        result = runtime.run(request)
        assert result.outcome == "committed", result.detail
        output = Path(request.output_root)
        assert (output / "_unresolved" / "mystery.pdf").is_file()
        assert result.plan is not None
        assert result.plan.count(ChangeAction.UNRESOLVED) == 1
        audit.close()
