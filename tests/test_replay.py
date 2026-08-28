"""Replay and two-phase apply.

Replay swaps only the planner and the domain agents for journal-backed ones, so a replay
that reproduces the original plan is evidence that the recorded compositions really do
determine the result. If replay had its own driver it would prove nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from system.capabilities import CapabilityRegistry
from system.components.builtin import build_registry
from system.components.filesystem_mutation import mutation
from system.contracts import ChangeAction
from system.runtime.audit import AuditStore
from system.runtime.engine import Runtime
from system.runtime.executor import Executor
from system.runtime.replay import ReplayError, assert_same_workflow, journal_components
from system.runtime.telemetry import RecordingExporter, Tracer
from system.runtime.verifier import Verifier
from system.workflows import WorkflowRegistry

from harness import build


def _replay_runtime(audit: AuditStore, run_id: str, tmp_path: Path) -> tuple[Runtime, object]:
    operators = build_registry()
    verifier = Verifier(operators)
    capabilities = CapabilityRegistry()
    planner, agents, _, _ = journal_components(audit, run_id)
    tracer = Tracer("replay", [RecordingExporter()])
    runtime = Runtime(
        operators=operators,
        capabilities=capabilities,
        workflows=WorkflowRegistry(capabilities=capabilities, operators=operators),
        verifier=verifier,
        executor=Executor(operators, verifier, tracer=tracer),
        planner=planner,
        agents=agents,
        audit=audit,
        workspace_root=tmp_path / "replay-work",
        tracer=tracer,
    )
    return runtime, planner


@pytest.fixture
def original(tmp_path: Path):
    runtime, request, audit, _ = build(tmp_path)
    result = runtime.run(request)
    assert result.outcome == "committed", result.detail
    yield result, request, audit
    audit.close()


class TestReplay:
    def test_reproduces_the_recorded_plan(self, original, tmp_path: Path) -> None:
        result, request, audit = original
        runtime, _ = _replay_runtime(audit, result.run_id, tmp_path)
        replayed = runtime.run(
            request.model_copy(update={"output_root": str(tmp_path / "again")}), commit=False
        )
        assert replayed.plan is not None
        assert [
            (e.source_ref, e.action, getattr(e, "target_relpath", None))
            for e in replayed.plan.entries
        ] == [
            (e.source_ref, e.action, getattr(e, "target_relpath", None))
            for e in result.plan.entries
        ]

    def test_makes_no_model_call(self, original, tmp_path: Path) -> None:
        """The whole point: the journal, not a model, determines the outcome."""
        result, request, audit = original
        runtime, planner = _replay_runtime(audit, result.run_id, tmp_path)
        runtime.run(
            request.model_copy(update={"output_root": str(tmp_path / "again")}), commit=False
        )
        assert planner.model_calls == 0

    def test_can_commit_the_replay_to_a_fresh_root(self, original, tmp_path: Path) -> None:
        result, request, audit = original
        runtime, _ = _replay_runtime(audit, result.run_id, tmp_path)
        replayed = runtime.run(
            request.model_copy(update={"output_root": str(tmp_path / "again")}), commit=True
        )
        assert replayed.outcome == "committed", replayed.detail
        first = sorted(
            p.relative_to(request.output_root).as_posix()
            for p in Path(request.output_root).rglob("*")
            if p.is_file()
        )
        second = sorted(
            p.relative_to(tmp_path / "again").as_posix()
            for p in (tmp_path / "again").rglob("*")
            if p.is_file()
        )
        assert first == second

    def test_the_recorded_plan_is_retrievable(self, original) -> None:
        result, _, audit = original
        stored = audit.recorded_plan(result.run_id)
        assert stored is not None
        assert stored.digest() == result.plan.digest()

    def test_an_unknown_run_is_refused(self, original, tmp_path: Path) -> None:
        _, _, audit = original
        with pytest.raises(KeyError, match="unknown run"):
            journal_components(audit, "does-not-exist", stage_of={})

    def test_replaying_a_changed_workflow_is_refused(self) -> None:
        """The digest covers pinned stage and prompt versions.

        A mismatch means the recorded run is not the run this code would produce, so a
        replay would give a confident answer to the wrong question.
        """
        with pytest.raises(ReplayError, match="workflow has changed"):
            assert_same_workflow("a" * 64, "b" * 64)


class TestApplyPhaseTwo:
    def test_a_plan_can_be_committed_later(self, tmp_path: Path) -> None:
        runtime, request, audit, _ = build(tmp_path)
        planned = runtime.run(request, commit=False)
        assert planned.plan is not None
        assert not Path(request.output_root).exists()

        staging = tmp_path / "later-staging"
        mutation.stage_plan(
            plan=planned.plan,
            input_root=Path(request.input_root),
            staging_root=staging,
        )
        assert mutation.verify_tree(plan=planned.plan, staging_root=staging)["ok"]
        mutation.commit(staging_root=staging, output_root=Path(request.output_root))
        assert (Path(request.output_root) / "2024" / "q1").is_dir()
        audit.close()

    def test_a_source_file_changed_since_planning_is_detected(self, tmp_path: Path) -> None:
        """Applying a stale plan would rename a file based on content it no longer has."""
        from system.cli import _verify_sources

        runtime, request, audit, _ = build(tmp_path)
        planned = runtime.run(request, commit=False)
        assert planned.plan is not None
        assert _verify_sources(planned.plan, Path(request.input_root)) == []

        changed = Path(request.input_root) / "2024" / "q1" / "scan001.pdf"
        changed.write_bytes(b"this document was replaced after planning")
        drifted = _verify_sources(planned.plan, Path(request.input_root))
        assert drifted == ["2024/q1/scan001.pdf"]
        audit.close()

    def test_a_deleted_source_is_detected(self, tmp_path: Path) -> None:
        from system.cli import _verify_sources

        runtime, request, audit, _ = build(tmp_path)
        planned = runtime.run(request, commit=False)
        assert planned.plan is not None
        (Path(request.input_root) / "2024" / "q1" / "scan002.pdf").unlink()
        assert "2024/q1/scan002.pdf" in _verify_sources(planned.plan, Path(request.input_root))
        audit.close()


class TestPlanRecording:
    def test_a_dry_run_still_records_its_plan(self, tmp_path: Path) -> None:
        """Otherwise a plan produced today could not be replayed or applied tomorrow."""
        runtime, request, audit, _ = build(tmp_path)
        result = runtime.run(request, commit=False)
        stored = audit.recorded_plan(result.run_id)
        assert stored is not None
        assert stored.count(ChangeAction.CHANGED) == 3
        audit.close()


class TestIdempotency:
    """Principle 12: re-applying a satisfied plan is a no-op.

    Until now a second identical run failed with "output root already exists", which is
    the same message a genuine collision produces — so a retry after a network blip looked
    exactly like a mistake.
    """

    def test_recommitting_the_same_plan_is_a_no_op(self, tmp_path: Path) -> None:
        runtime, request, audit, _ = build(tmp_path)
        first = runtime.run(request)
        assert first.outcome == "committed", first.detail
        before = sorted(p.name for p in Path(request.output_root).rglob("*") if p.is_file())

        runtime.grants.clear()
        again = runtime.run(request)
        assert again.outcome == "committed"
        assert "already committed" in again.detail
        after = sorted(p.name for p in Path(request.output_root).rglob("*") if p.is_file())
        assert after == before
        audit.close()

    def test_a_different_plan_to_the_same_root_still_collides(self, tmp_path: Path) -> None:
        """Idempotency must not become 'silently accept a conflicting write'."""
        runtime, request, audit, _ = build(tmp_path)
        assert runtime.run(request).outcome == "committed"

        other = request.model_copy(update={"request_id": "second", "prompt": "different"})
        (Path(request.input_root) / "2024" / "extra.pdf").write_bytes(b"a new invoice")
        result = runtime.run(other)
        assert result.outcome != "committed"
        audit.close()

    def test_the_receipt_is_keyed_on_plan_and_destination(self, tmp_path: Path) -> None:
        runtime, request, audit, _ = build(tmp_path)
        result = runtime.run(request)
        assert result.plan is not None
        assert audit.find_commit(
            plan_digest=result.plan.fingerprint(), output_root=request.output_root
        )
        assert (
            audit.find_commit(plan_digest=result.plan.fingerprint(), output_root="/elsewhere")
            is None
        )
        assert audit.find_commit(plan_digest="0" * 64, output_root=request.output_root) is None
        audit.close()


class TestPreviewCommitsWhatItShowed:
    """`run` previews a plan and then commits it — that plan, not another one.

    It used to re-run the whole workflow after approval, so the model was re-invoked and
    the committed plan could differ from the one the user approved. It also doubled the
    cost of every run.
    """

    def test_committing_a_previewed_plan_makes_no_further_model_call(
        self, tmp_path: Path
    ) -> None:
        runtime, request, audit, _ = build(tmp_path)
        planned = runtime.run(request, commit=False)
        assert planned.plan is not None

        before = {domain: agent.call_count for domain, agent in runtime.agents.items()}
        committed = runtime.commit_planned(planned)
        after = {domain: agent.call_count for domain, agent in runtime.agents.items()}

        assert committed.outcome == "committed", committed.detail
        assert after == before, "committing must not re-invoke a single agent"
        audit.close()

    def test_the_committed_tree_matches_the_previewed_plan_exactly(
        self, tmp_path: Path
    ) -> None:
        runtime, request, audit, _ = build(tmp_path)
        planned = runtime.run(request, commit=False)
        assert planned.plan is not None
        expected = sorted(
            e.target_relpath
            for e in planned.plan.entries
            if getattr(e, "target_relpath", None)
        )

        runtime.commit_planned(planned)
        output = Path(request.output_root)
        produced = sorted(
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file() and not path.as_posix().count("_unresolved")
        )
        assert produced == expected
        audit.close()

    def test_committing_without_a_plan_is_refused(self, tmp_path: Path) -> None:
        from system.runtime.engine import RunResult

        runtime, _, audit, _ = build(tmp_path)
        with pytest.raises(Exception, match="no plan to commit"):
            runtime.commit_planned(RunResult(run_id="x", workflow_id="rename_files",
                                             outcome="planned"))
        audit.close()
