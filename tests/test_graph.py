"""LangGraph orchestration: durability and human approval.

LangGraph coordinates; it does not make policy. These tests assert the graph reaches the
same outcomes as the plain driver, and that the approval interrupt genuinely gates the
irreversible step.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.graph import WorkflowGraph, resume, run_with_graph, sqlite_checkpointer

from test_rename_files import build


@pytest.fixture
def harness(tmp_path: Path):
    runtime, request, audit, recorder = build(tmp_path)
    yield runtime, request, audit, recorder
    audit.close()


class TestApprovalGate:
    def test_run_suspends_before_committing(self, harness, tmp_path: Path) -> None:
        runtime, request, _, _ = harness
        with sqlite_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
            state, graph, run_id = run_with_graph(runtime, request, checkpointer=saver)
            assert "__interrupt__" in state, "the run must pause before an irreversible write"
            assert not Path(request.output_root).exists()

            payload = state["__interrupt__"][0].value
            assert payload["kind"] == "commit_approval"
            summary = payload["summary"]
            assert summary["entries"] == 3
            assert summary["changed"] == 3
            assert summary["plan_digest"]

    def test_approving_commits(self, harness, tmp_path: Path) -> None:
        runtime, request, _, _ = harness
        with sqlite_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
            _, graph, _ = run_with_graph(runtime, request, checkpointer=saver)
            result = resume(graph, approved=True)
        assert result.outcome == "committed", result.detail
        assert (
            Path(request.output_root) / "2024/q1/202401, ACME Corporation, INV-99812, PO-44117.pdf"
        ).is_file()

    def test_declining_leaves_nothing_behind(self, harness, tmp_path: Path) -> None:
        runtime, request, _, _ = harness
        with sqlite_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
            _, graph, _ = run_with_graph(runtime, request, checkpointer=saver)
            result = resume(graph, approved=False)
        assert result.outcome != "committed"
        assert not Path(request.output_root).exists()
        assert not graph.staging.exists(), "declining must discard staging"

    def test_approval_can_be_waived_for_an_unattended_run(self, harness, tmp_path: Path) -> None:
        runtime, request, _, _ = harness
        with sqlite_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
            state, _, _ = run_with_graph(
                runtime, request, checkpointer=saver, require_approval=False
            )
        assert state["outcome"] == "committed"
        assert Path(request.output_root).is_dir()


class TestDurability:
    def test_state_is_checkpointed_at_every_stage_boundary(self, harness, tmp_path: Path) -> None:
        runtime, request, _, _ = harness
        path = tmp_path / "checkpoints.sqlite3"
        with sqlite_checkpointer(path) as saver:
            _, graph, run_id = run_with_graph(runtime, request, checkpointer=saver)
            config = {"configurable": {"thread_id": run_id}}
            history = list(graph.compiled.get_state_history(config))
        # One checkpoint per stage plus the interrupt: enough to resume at a boundary
        # rather than restarting the whole run.
        assert len(history) > len(graph.workflow.stages)

    def test_checkpoint_is_local_and_protected(self, harness, tmp_path: Path) -> None:
        """The checkpointer holds working state, including content-derived values.

        That is fine — it lives in the run's own workspace beside the extracted text and
        the staged files — provided it stays local and is protected like the rest of it.
        """
        runtime, request, _, _ = harness
        path = tmp_path / "checkpoints.sqlite3"
        with sqlite_checkpointer(path) as saver:
            run_with_graph(runtime, request, checkpointer=saver)
        assert path.stat().st_mode & 0o077 == 0, "checkpoint must not be group/world readable"

    def test_langchain_autotracing_is_disabled(self, tmp_path: Path, monkeypatch) -> None:
        """LangChain tracing would ship whole node payloads, bypassing TelemetryEnvelope."""
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        with sqlite_checkpointer(tmp_path / "cp.sqlite3"):
            pass
        import os

        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
        assert os.environ["LANGSMITH_TRACING"] == "false"

    def test_the_opt_out_is_explicit(self, tmp_path: Path, monkeypatch) -> None:
        from shakespeare.graph import disable_autotracing

        monkeypatch.setenv("SHAKESPEARE_ALLOW_LANGCHAIN_TRACING", "1")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        disable_autotracing()
        import os

        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"


class TestParity:
    def test_graph_and_plain_driver_agree(self, tmp_path: Path) -> None:
        """The graph must not become a second execution path."""
        plain_runtime, plain_request, plain_audit, _ = build(tmp_path / "plain")
        plain = plain_runtime.run(plain_request)

        graph_runtime, graph_request, graph_audit, _ = build(tmp_path / "graph")
        with sqlite_checkpointer(tmp_path / "cp.sqlite3") as saver:
            _, graph, _ = run_with_graph(graph_runtime, graph_request, checkpointer=saver)
            through_graph = resume(graph, approved=True)

        assert plain.outcome == through_graph.outcome == "committed"
        assert plain.plan is not None and through_graph.plan is not None
        assert [e.target_relpath for e in plain.plan.entries] == [
            e.target_relpath for e in through_graph.plan.entries
        ]
        plain_audit.close()
        graph_audit.close()


class TestGraphShape:
    def test_every_stage_is_a_node(self, harness, tmp_path: Path) -> None:
        runtime, request, _, _ = harness
        workflow = runtime.workflows.get("rename_files")
        graph = WorkflowGraph(
            runtime=runtime,
            workflow=workflow,
            request=request,
            staging=tmp_path / "staging",
            workspace=tmp_path / "work",
        )
        with sqlite_checkpointer(tmp_path / "cp.sqlite3") as saver:
            compiled = graph.build(saver)
        nodes = set(compiled.get_graph().nodes)
        for stage in workflow.stages:
            assert stage.name in nodes
        assert {"approve", "commit", "abandon"} <= nodes
