"""Durability, cost reconciliation, the OCR rerun case, and the live lane.

The remaining verification items from the plan.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from system.capabilities import CapabilityRegistry
from system.capabilities.agent import FakeCapabilityAgent
from system.components.catalog import build_registry
from system.contracts import (
    BudgetEnvelope,
    RouteDecision,
)
from system.planning.planner import ScriptedGoalPlanner
from system.runtime.audit import AuditStore
from system.runtime.durability import WorkflowGraph, sqlite_checkpointer
from system.runtime.engine import Runtime
from system.runtime.executor import Budget, Executor
from system.runtime.telemetry import RecordingExporter, Tracer
from system.runtime.verifier import Verifier
from system.workflows import WorkflowRegistry

from harness import INVOICES, build, rename_agent, seed_invoices, values_for


def _revive(
    tmp_path: Path, request, *, audit_path: Path | None = None
) -> tuple[Runtime, AuditStore]:
    """A runtime built fresh, reading the audit log from disk rather than from memory."""
    source = seed_invoices(tmp_path / "in", INVOICES)
    operators = build_registry()
    verifier = Verifier(operators)
    capabilities = CapabilityRegistry()
    tracer = Tracer("revived", [RecordingExporter()])
    audit = AuditStore(audit_path or (tmp_path / "audit.sqlite3"))

    return (
        Runtime(
            operators=operators,
            capabilities=capabilities,
            workflows=WorkflowRegistry(capabilities=capabilities, operators=operators),
            verifier=verifier,
            executor=Executor(operators, verifier, tracer=tracer),
            planner=ScriptedGoalPlanner(
                route=RouteDecision(workflow_id="rename_files")
            ),
            agents={"*": rename_agent(values_for(source, INVOICES))},
            audit=audit,
            workspace_root=tmp_path / "work",
            tracer=tracer,
        ),
        audit,
    )


class TestDurability:
    def test_a_run_resumes_from_persisted_state_in_a_new_process_image(
        self, tmp_path: Path
    ) -> None:
        """Resume through the checkpoint file, not through objects held in memory.

        The second half uses a fresh runtime, a fresh graph and a fresh checkpointer
        connection, so nothing but the SQLite file carries the run forward — which is what
        surviving a killed process actually means.
        """
        from system.runtime.durability import run_with_graph

        runtime, request, audit, _ = build(tmp_path)
        checkpoint = tmp_path / "checkpoints.sqlite3"
        with sqlite_checkpointer(checkpoint) as saver:
            state, first_graph, run_id = run_with_graph(runtime, request, checkpointer=saver)
            assert "__interrupt__" in state
        assert not Path(request.output_root).exists()

        # Everything from the first half is now discarded.
        del first_graph, runtime, state
        audit.close()

        revived_runtime, revived_audit = _revive(tmp_path, request)
        revived = WorkflowGraph(runtime=revived_runtime, request=request)
        with sqlite_checkpointer(checkpoint) as saver:
            from langgraph.types import Command

            compiled = revived.build(saver)
            final = compiled.invoke(
                Command(resume=True), {"configurable": {"thread_id": run_id}}
            )

        assert final["outcome"] == "committed"
        assert (Path(request.output_root) / "2024" / "q1").is_dir()
        revived_audit.close()

    def test_resume_requires_the_same_audit_log_not_just_the_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """The checkpoint carries the work; the audit log carries the plan.

        A resume pointed at a fresh audit database finds no plan and refuses, rather than
        committing something it cannot account for.
        """
        from langgraph.types import Command
        from system.runtime.durability import run_with_graph

        runtime, request, audit, _ = build(tmp_path)
        checkpoint = tmp_path / "cp.sqlite3"
        with sqlite_checkpointer(checkpoint) as saver:
            _, _, thread = run_with_graph(runtime, request, checkpointer=saver)
        audit.close()

        stranded, stranded_audit = _revive(tmp_path, request, audit_path=tmp_path / "other.db")
        graph = WorkflowGraph(runtime=stranded, request=request)
        with sqlite_checkpointer(checkpoint) as saver:
            compiled = graph.build(saver)
            final = compiled.invoke(
                Command(resume=True), {"configurable": {"thread_id": thread}}
            )

        assert final["outcome"] == "aborted"
        assert "no plan to commit" in final["detail"]
        assert not Path(request.output_root).exists()
        stranded_audit.close()

    def test_the_audit_log_holds_no_partial_facts_after_an_interrupt(
        self, tmp_path: Path
    ) -> None:
        """A suspended run must leave completed facts only, never half-written ones."""
        from sqlalchemy import select
        from system.runtime.audit import schema
        from system.runtime.durability import run_with_graph

        runtime, request, audit, _ = build(tmp_path)
        with sqlite_checkpointer(tmp_path / "cp.sqlite3") as saver:
            run_with_graph(runtime, request, checkpointer=saver)

        with audit.engine.begin() as connection:
            attempts = connection.execute(select(schema.stage_attempts)).mappings().all()
            commits = connection.execute(select(schema.commits)).mappings().all()
            verdicts = connection.execute(select(schema.stage_verdicts)).mappings().all()

        assert attempts, "stages that completed must be recorded"
        assert len(verdicts) == len(attempts), "every recorded attempt carries its verdict"
        assert not commits, "nothing may be recorded as committed before approval"
        audit.close()


class TestCostReconciliation:
    def test_journal_costs_matches_what_was_metered(self, tmp_path: Path) -> None:
        """`journal costs` is the bill; the budget is the meter. They must agree."""
        audit = AuditStore(tmp_path / "audit.sqlite3")
        audit.record_run(
            run_id="r",
            workflow_id="w",
            workflow_version="1.0.0",
            workflow_digest="d",
            request_digest="q",
            input_root_digest="i",
        )
        budget = Budget(envelope=BudgetEnvelope(model_invocations="10"), items=0)
        charges = [("planner", 0.011, 120, 40), ("domain", 0.004, 90, 30), ("domain", 0.002, 10, 5)]
        for role, cost, prompt_tokens, completion_tokens in charges:
            budget.consume_model_invocation(tokens=prompt_tokens + completion_tokens, cost_usd=cost)
            audit.record_model_invocation(
                run_id="r",
                role=role,
                profile_id="p",
                requested_model="openrouter/openai/gpt-5-mini",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
            )

        costs = audit.costs("r")
        assert costs["model_invocations"] == budget.usage.model_invocations
        assert costs["cost_usd"] == pytest.approx(budget.usage.cost_usd)
        assert costs["prompt_tokens"] + costs["completion_tokens"] == budget.usage.total_tokens
        assert costs["by_role"]["domain"] == pytest.approx(0.006)
        audit.close()


class TestUnsatisfiableGoal:
    """The framework's version of the OCR rerun case.

    A capability that cannot produce the evidence its goal requires does not fail
    silently: the gate reports what is missing, the goal is retried, and the run aborts
    without creating an output root rather than committing a partial answer.
    """

    def test_a_goal_whose_evidence_never_arrives_aborts_cleanly(
        self, tmp_path: Path
    ) -> None:
        from system.capabilities.runner import Organization

        agent = FakeCapabilityAgent()
        agent.queue("survey", Organization(intent="produce nothing", sufficient=True))
        runtime, request, audit, _ = build(tmp_path, agents={"*": agent})

        result = runtime.run(request)
        assert result.outcome == "aborted", result.detail
        assert not Path(request.output_root).exists()
        assert "inventoried" not in result.satisfied
        audit.close()

    def test_the_gate_says_what_is_missing(self, tmp_path: Path) -> None:
        from system.capabilities.runner import Organization
        from system.runtime.goals import GateOutcome

        agent = FakeCapabilityAgent()
        agent.queue("survey", Organization(intent="produce nothing", sufficient=True))
        runtime, request, audit, _ = build(tmp_path, agents={"*": agent})
        result = runtime.run(request, commit=False)

        blocked = result.attempts[0].gate
        assert blocked.outcome is GateOutcome.BLOCKED
        assert blocked.missing_kinds == ("FileInventory",)
        audit.close()

    def test_a_goal_is_retried_before_the_run_gives_up(self, tmp_path: Path) -> None:
        from system.capabilities.runner import Organization

        agent = FakeCapabilityAgent()
        agent.queue("survey", Organization(intent="produce nothing", sufficient=True))
        runtime, request, audit, _ = build(tmp_path, agents={"*": agent})
        result = runtime.run(request, commit=False)

        attempts = [a for a in result.attempts if a.goal_id == "inventoried"]
        assert len(attempts) > 1, "the goal is retried, not abandoned on one failure"
        audit.close()

    def test_a_goal_that_achieves_nothing_is_not_attempted_a_third_time(
        self, tmp_path: Path
    ) -> None:
        """Repeating an attempt that changed nothing is spending money on a coin flip."""
        from system.capabilities.runner import Organization

        agent = FakeCapabilityAgent()
        agent.queue("survey", Organization(intent="produce nothing", sufficient=True))
        runtime, request, audit, _ = build(tmp_path, agents={"*": agent})
        result = runtime.run(request, commit=False)

        attempts = [a for a in result.attempts if a.goal_id == "inventoried"]
        assert len(attempts) == 2, "one attempt, one retry, then the repetition is seen"
        assert "achieved nothing new" in (result.detail or "")
        audit.close()


@pytest.mark.skipif(
    os.environ.get("SHAKESPEARE_LIVE") != "1",
    reason="live lane: set SHAKESPEARE_LIVE=1 and SHAKESPEARE_MODEL to run",
)
class TestLiveSmoke:
    """Opt-in, and skipped by default so it can never be mistaken for coverage.

    This is the only test that spends money. It exercises the real planner and real
    domain agents against a real provider over the fixture tree.
    """

    def test_a_real_model_produces_a_balanced_plan(self, tmp_path: Path) -> None:
        from system.contracts import RequestContract
        from system.services import build_runtime

        from fixtures.build import build_tree, cleanup

        source = tmp_path / "invoices"
        build_tree(source)
        try:
            services = build_runtime(state_root=tmp_path / "state")
            result = services.runtime.run(
                RequestContract(
                    request_id="live",
                    prompt=(
                        "rename these invoices to YYYYMM, vendor, invoice number, PO number"
                    ),
                    input_root=str(source),
                    output_root=str(tmp_path / "out"),
                ),
                commit=False,
            )
            assert result.plan is not None, result.detail
            scanned = sum(1 for path in source.rglob("*") if path.is_file())
            assert result.plan.balanced(scanned), "the plan must account for every file"
            print("\nfrozen convention and decisions:")
            for entry in result.plan.entries:
                print(f"  {entry.action:11} {entry.source_ref} -> "
                      f"{getattr(entry, 'target_relpath', None) or entry.reason}")
            print(services.audit.costs(result.run_id))
            services.audit.close()
        finally:
            cleanup(source)
