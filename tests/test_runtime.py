"""The spine driver, proved generic before any real workflow exists.

`noop_passthrough` shares nothing with rename_files but the runtime itself. If the driver
ever needs to know which workflow it is running, this file is where that shows up.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.agent import FakeDomainAgent
from shakespeare.contracts import (
    ChangeAction,
    Invocation,
    ObligationResult,
    RequestContract,
    RouteDecision,
    StageDecision,
    StagePlan,
    StageVerdict,
)
from shakespeare.planner import FakePlanner

from harness import (
    build_runtime,
    composition,
    goal,
    noop_stages,
    noop_workflow,
    seed_tree,
)


def _agents() -> dict[str, FakeDomainAgent]:
    survey = FakeDomainAgent().queue(
        "survey",
        composition(
            "survey",
            Invocation(invocation_id="scan", operator="fs.scan", inputs=("root",)),
        ),
    )
    passthrough = FakeDomainAgent().queue(
        "passthrough",
        composition(
            "passthrough",
            Invocation(
                invocation_id="assemble",
                operator="plan.assemble",
                inputs=("run_id", "workflow_id", "workflow_digest", "items"),
                parameters={"decision_digest": "noop", "default_action": "unchanged"},
                bindings={"scanned": "items"},
            ),
        ),
    )
    return {"survey": survey, "passthrough": passthrough}


def _planner(inventory_plans: tuple[StagePlan, ...] = ()) -> FakePlanner:
    planner = FakePlanner(route=RouteDecision(workflow_id="noop_passthrough"))
    planner.queue_plan(
        "inventory", *(inventory_plans or (StagePlan(activated=(goal("survey"),)),))
    )
    planner.queue_plan("compose_changes", StagePlan(activated=(goal("passthrough"),)))
    return planner


def _request(tmp_path: Path) -> RequestContract:
    source = seed_tree(tmp_path / "in")
    return RequestContract(
        request_id="req-1",
        prompt="copy everything through unchanged",
        input_root=str(source),
        output_root=str(tmp_path / "out"),
    )


@pytest.fixture
def harness(tmp_path: Path):
    (tmp_path / "in").mkdir()
    request = _request(tmp_path)
    runtime, audit, recorder = build_runtime(
        tmp_path,
        stages=noop_stages(),
        workflow=noop_workflow(),
        planner=_planner(),
        agents=_agents(),
    )
    yield runtime, audit, recorder, request
    audit.close()


class TestGenericSpine:
    def test_runs_end_to_end_through_the_shared_driver(self, harness, tmp_path: Path) -> None:
        runtime, _, _, request = harness
        result = runtime.run(request)
        assert result.outcome == "committed", result.detail
        assert result.plan is not None
        assert result.plan.balanced(3)
        assert result.plan.count(ChangeAction.UNCHANGED) == 3

    def test_output_mirrors_the_input_structure(self, harness, tmp_path: Path) -> None:
        runtime, _, _, request = harness
        runtime.run(request)
        output = Path(request.output_root)
        produced = sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        )
        assert produced == ["2024/q1/scan001.pdf", "2024/scan002.pdf", "notes.txt"]

    def test_source_tree_is_never_mutated(self, harness, tmp_path: Path) -> None:
        runtime, _, _, request = harness
        before = sorted(p.name for p in Path(request.input_root).rglob("*"))
        runtime.run(request)
        assert sorted(p.name for p in Path(request.input_root).rglob("*")) == before

    def test_no_workflow_or_stage_id_is_hardcoded_in_the_driver(self) -> None:
        """The spine must not learn the names of the things it drives."""
        import shakespeare

        root = Path(shakespeare.__file__).parent
        forbidden = ("rename_files", "noop_passthrough", "compose_changes", "convention")
        for module in ("runtime.py", "executor.py", "planner.py", "verifier.py"):
            source = (root / module).read_text()
            for name in forbidden:
                assert name not in source, f"{module} hardcodes {name!r}"

    def test_dry_run_plans_without_creating_an_output_root(self, harness) -> None:
        runtime, _, _, request = harness
        result = runtime.run(request, commit=False)
        assert result.outcome == "planned"
        assert result.plan is not None
        assert not Path(request.output_root).exists()


class TestAuditTrail:
    def test_records_the_dag_for_every_stage(self, harness) -> None:
        runtime, audit, _, request = harness
        result = runtime.run(request)
        dag = audit.dag(result.run_id, "inventory")
        assert len(dag["attempts"]) == 1
        assert [node["operator"] for node in dag["attempts"][0]["nodes"]] == ["fs.scan"]
        assert dag["attempts"][0]["verdict"]["decision"] == "accept"

    def test_commit_is_recorded(self, harness) -> None:
        runtime, audit, _, request = harness
        runtime.run(request)
        from shakespeare.audit import schema
        from sqlalchemy import select

        with audit.engine.begin() as connection:
            commits = connection.execute(select(schema.commits)).mappings().all()
            outcomes = connection.execute(select(schema.run_outcomes)).mappings().all()
        assert len(commits) == 1
        assert commits[0]["entry_count"] == 3
        assert outcomes[0]["outcome"] == "committed"


class TestTelemetry:
    def test_no_document_content_reaches_the_exporter(self, harness) -> None:
        runtime, _, recorder, request = harness
        runtime.run(request)
        shipped = recorder.serialized()
        for secret in ("invoice one", "invoice two", "a loose note"):
            assert secret not in shipped
        assert "operator.fs.scan" in shipped


class TestAttemptLoop:
    def _run_with_verdicts(self, tmp_path: Path, *verdicts: StageVerdict):
        (tmp_path / "in").mkdir()
        request = _request(tmp_path)
        # A rerun must differ from the previous attempt, so the second plan carries a
        # distinguishable goal.
        planner = _planner(
            (
                StagePlan(activated=(goal("survey"),)),
                StagePlan(activated=(goal("survey", "retry the items that failed"),)),
            )
        )
        planner.queue_verdict("inventory", *verdicts)
        runtime, audit, _ = build_runtime(
            tmp_path,
            stages=noop_stages(),
            workflow=noop_workflow(),
            planner=planner,
            agents=_agents(),
        )
        return runtime.run(request), audit

    def test_rerun_then_accept_records_both_attempts(self, tmp_path: Path) -> None:
        result, audit = self._run_with_verdicts(
            tmp_path,
            StageVerdict(met=False, decision=StageDecision.RERUN, rationale="thin coverage"),
            StageVerdict(met=True, decision=StageDecision.ACCEPT),
        )
        assert result.outcome == "committed", result.detail
        attempts = audit.dag(result.run_id, "inventory")["attempts"]
        assert [item["attempt"]["attempt_no"] for item in attempts] == [1, 2]
        assert attempts[0]["verdict"]["decision"] == "rerun"
        audit.close()

    def test_exhausting_attempts_aborts_without_creating_output(self, tmp_path: Path) -> None:
        result, audit = self._run_with_verdicts(
            tmp_path,
            StageVerdict(met=False, decision=StageDecision.RERUN, rationale="still thin"),
            StageVerdict(met=False, decision=StageDecision.RERUN, rationale="still thin"),
        )
        assert result.outcome == "aborted"
        assert not (tmp_path / "out").exists(), "an aborted run must leave nothing behind"
        audit.close()

    def test_planner_cannot_accept_over_a_failed_obligation(self, tmp_path: Path) -> None:
        """The planner judges whether the goal was met, not whether a check passed."""
        (tmp_path / "in").mkdir()
        request = _request(tmp_path)
        planner = _planner()
        # compose_changes has real obligations; starve them by dropping the assemble step.
        planner.queue_verdict(
            "compose_changes",
            StageVerdict(met=True, decision=StageDecision.ACCEPT, rationale="looks fine"),
        )
        agents = _agents()
        agents["passthrough"] = FakeDomainAgent().queue(
            "passthrough", composition("passthrough")
        )
        runtime, audit, _ = build_runtime(
            tmp_path,
            stages=noop_stages(),
            workflow=noop_workflow(),
            planner=planner,
            agents=agents,
        )
        result = runtime.run(request)
        # Either the planner exhausts its attempts or its non-progressing rerun is
        # refused; what matters is that a bare "accept" never reaches a commit.
        assert result.outcome == "aborted"
        assert not (tmp_path / "out").exists()
        audit.close()


class TestObligationEvidence:
    def test_missing_evidence_fails_closed(self) -> None:
        from shakespeare.operators.planning import run_check

        result: ObligationResult = run_check("balanced", "balanced", {})
        assert not result.passed
        assert "missing_evidence" in result.detail
