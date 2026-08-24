"""LangGraph orchestration.

LangGraph coordinates; it does not make policy.  Every stage node calls the same
`Runtime._run_stage` the plain driver uses, so the graph adds durability and human
approval without becoming a second execution path — the verifier still authorizes, the
executor still runs, and the audit log still records.

What the graph buys:

- **Durability.** State is checkpointed after every stage, so a killed run resumes at the
  stage boundary rather than starting over.
- **Human approval.** `interrupt()` suspends the run before an irreversible commit, or
  before admitting a high-risk operator, and `Command(resume=...)` continues it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from .contracts import (
    ErrorCode,
    RequestContract,
    StageDecision,
    content_digest,
)
from .operators import mutation
from .runtime import RunResult, Runtime, StageOutcome
from .verifier import Denial
from .workflows import RegisteredWorkflow


class GraphState(TypedDict, total=False):
    """Checkpointed working state.

    This is *local* state, written to a SQLite file inside the run's own workspace
    alongside the extracted text and the staged files.  It therefore does contain
    content-derived values — rendered filenames, and the stage context — and is exactly
    as sensitive as the workspace it lives in, which is to say: it must never leave the
    machine.

    The export boundary is `telemetry.TelemetryEnvelope`, not this.  The one way this
    state could escape is LangChain's automatic tracing of graph nodes, so
    `disable_autotracing()` turns that off: our own gateway and tracer are the only
    sanctioned export path, and they emit digests.
    """

    run_id: str
    workflow_id: str
    stage_index: int
    context: dict[str, Any]
    outcome: str
    detail: str
    error_code: str | None
    approved: bool


def disable_autotracing() -> None:
    """Stop LangChain exporting graph node inputs and outputs.

    LangChain tracing would ship whole node payloads to LangSmith, bypassing
    TelemetryEnvelope entirely.  Our tracer is the only sanctioned export path, so
    auto-tracing is turned off unless someone deliberately overrides it.
    """
    import os

    if os.environ.get("SHAKESPEARE_ALLOW_LANGCHAIN_TRACING") == "1":
        return
    for name in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
        os.environ[name] = "false"


@contextmanager
def sqlite_checkpointer(path: Path) -> Iterator[Any]:
    """A durable checkpointer, private to the workspace."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    disable_autotracing()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
    # The checkpointer holds working state, so it inherits the workspace's protection.
    if path.exists():
        path.chmod(0o600)


@dataclass
class WorkflowGraph:
    """Compiles one registered workflow into a graph of stage nodes."""

    runtime: Runtime
    workflow: RegisteredWorkflow
    request: RequestContract
    staging: Path
    workspace: Path
    require_approval: bool = True

    def __post_init__(self) -> None:
        self.outcomes: list[StageOutcome] = []

    def build(self, checkpointer: Any) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph: Any = StateGraph(GraphState)
        names = [stage.name for stage in self.workflow.stages]

        for index, stage in enumerate(self.workflow.stages):
            graph.add_node(stage.name, self._stage_node(index))

        graph.add_node("approve", self._approval_node)
        graph.add_node("commit", self._commit_node)
        graph.add_node("abandon", self._abandon_node)

        graph.add_edge(START, names[0])
        for index, name in enumerate(names):
            following = names[index + 1] if index + 1 < len(names) else "approve"
            after = (
                "approve" if name == self.workflow.spec.commit_after else following
            )
            graph.add_conditional_edges(
                name,
                self._advance(after),
                {after: after, "abandon": "abandon"},
            )
        graph.add_conditional_edges(
            "approve", self._after_approval, {"commit": "commit", "abandon": "abandon"}
        )
        graph.add_edge("commit", END)
        graph.add_edge("abandon", END)
        return graph.compile(checkpointer=checkpointer)

    # -- nodes --------------------------------------------------------------------------

    def _stage_node(self, index: int) -> Any:
        stage = self.workflow.stages[index]

        def node(state: GraphState) -> dict[str, Any]:
            context = dict(state.get("context", {}))
            try:
                outcome = self.runtime._run_stage(
                    stage=stage,
                    request=self.request,
                    context=context,
                    workspace=self.workspace,
                    run_id=state["run_id"],
                )
            except Denial as denial:
                return {
                    "outcome": "aborted"
                    if denial.code is ErrorCode.ATTEMPTS_EXHAUSTED
                    else "denied",
                    "detail": denial.reason,
                    "error_code": str(denial.code),
                    "context": context,
                }

            self.outcomes.append(outcome)
            if outcome.verdict.decision is not StageDecision.ACCEPT:
                return {
                    "outcome": "aborted",
                    "detail": f"stage {stage.name} was not accepted:"
                    f" {outcome.verdict.rationale}",
                    "error_code": str(ErrorCode.ATTEMPTS_EXHAUSTED),
                    "context": context,
                }

            self.runtime._ensure_staged(
                run_id=state["run_id"],
                context=context,
                input_root=Path(self.request.input_root),
                staging=self.staging,
            )
            return {"stage_index": index + 1, "context": context}

        return node

    def _approval_node(self, state: GraphState) -> dict[str, Any]:
        """Suspend before the irreversible step.

        The interrupt payload is a summary — counts and digests — because the whole plan
        may be large and because this value is checkpointed.
        """
        if state.get("outcome"):
            return {}
        if not self.require_approval:
            return {"approved": True}

        from langgraph.types import interrupt

        plan = self.runtime._plan_from_context(state.get("context", {}))
        summary = {
            "run_id": state["run_id"],
            "workflow": self.workflow.spec.id,
            "entries": len(plan.entries) if plan else 0,
            "changed": sum(1 for e in plan.entries if e.action == "changed") if plan else 0,
            "unresolved": (
                sum(1 for e in plan.entries if e.action == "unresolved") if plan else 0
            ),
            "output_root": self.request.output_root,
            "plan_digest": plan.digest() if plan else None,
        }
        decision = interrupt({"kind": "commit_approval", "summary": summary})
        return {"approved": bool(decision)}

    def _commit_node(self, state: GraphState) -> dict[str, Any]:
        plan = self.runtime._plan_from_context(state.get("context", {}))
        if plan is not None:
            self.runtime.audit.record_plan(run_id=state["run_id"], plan=plan)
        result = self.runtime._commit(
            run_id=state["run_id"],
            workflow=self.workflow,
            plan=plan,
            staging=self.staging,
            output_root=Path(self.request.output_root),
            outcomes=tuple(self.outcomes),
        )
        return {"outcome": result.outcome, "detail": result.detail}

    def _abandon_node(self, state: GraphState) -> dict[str, Any]:
        """Nothing user-visible was ever created, so rollback is discarding staging."""
        mutation.discard(self.staging)
        outcome = state.get("outcome") or "aborted"
        self.runtime.audit.record_run_outcome(
            run_id=state["run_id"], outcome=outcome, error_code=state.get("error_code")
        )
        return {"outcome": outcome}

    # -- edges --------------------------------------------------------------------------

    @staticmethod
    def _advance(following: str) -> Any:
        def route(state: GraphState) -> str:
            return "abandon" if state.get("outcome") else following

        return route

    @staticmethod
    def _after_approval(state: GraphState) -> str:
        if state.get("outcome"):
            return "abandon"
        return "commit" if state.get("approved") else "abandon"


def run_with_graph(
    runtime: Runtime,
    request: RequestContract,
    *,
    checkpointer: Any,
    require_approval: bool = True,
    thread_id: str | None = None,
) -> tuple[Any, WorkflowGraph, str]:
    """Start a durable run.  Returns (state-or-interrupt, graph, run_id)."""
    route, _ = runtime.planner.select_workflow(request, runtime.workflows.routing_catalog())
    if not route.supported or route.workflow_id not in runtime.workflows:
        raise Denial(
            ErrorCode.COMPOSITION_INVALID,
            route.rationale or "no registered workflow handles this request",
        )

    workflow = runtime.workflows.get(route.workflow_id)
    run_id = thread_id or uuid4().hex
    workspace = runtime.workspace_root / run_id
    workspace.mkdir(parents=True, exist_ok=True)

    runtime.audit.record_run(
        run_id=run_id,
        workflow_id=workflow.spec.id,
        workflow_version=workflow.spec.version,
        workflow_digest=workflow.digest(),
        request_digest=request.digest(),
        input_root_digest=content_digest(request.input_root),
    )

    graph = WorkflowGraph(
        runtime=runtime,
        workflow=workflow,
        request=request,
        staging=workspace / "staging",
        workspace=workspace,
        require_approval=require_approval,
    )
    compiled = graph.build(checkpointer)
    config = {"configurable": {"thread_id": run_id}}
    state = compiled.invoke(
        {
            "run_id": run_id,
            "workflow_id": workflow.spec.id,
            "stage_index": 0,
            "context": {
                "run_id": run_id,
                "workflow_id": workflow.spec.id,
                "workflow_digest": workflow.digest(),
                "root": request.input_root,
                "input_root": request.input_root,
                "output_root": request.output_root,
                "staging_root": str(workspace / "staging"),
                "prompt": request.prompt,
            },
        },
        config,
    )
    graph.compiled = compiled  # type: ignore[attr-defined]
    graph.config = config  # type: ignore[attr-defined]
    return state, graph, run_id


def resume(graph: WorkflowGraph, *, approved: bool) -> RunResult:
    """Continue a suspended run after a human decision."""
    from langgraph.types import Command

    state = graph.compiled.invoke(Command(resume=approved), graph.config)  # type: ignore[attr-defined]
    return RunResult(
        run_id=state["run_id"],
        workflow_id=state["workflow_id"],
        outcome=state.get("outcome", "unknown"),
        plan=graph.runtime._plan_from_context(state.get("context", {})),
        committed_to=graph.request.output_root if state.get("outcome") == "committed" else None,
        stages=tuple(graph.outcomes),
        detail=state.get("detail", ""),
    )
