"""LangGraph orchestration.

LangGraph coordinates; it does not make policy.

The control loop is now dynamic — which goal is pursued next depends on what evidence
exists — so a static node per goal would misrepresent it. The graph models what is
genuinely static about a run instead: pursue the goals, pause for approval, commit or
abandon. Durability and the human-in-the-loop interrupt are what LangGraph is here for,
and both sit at exactly those boundaries.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from .contracts import RequestContract
from .operators import mutation
from .runtime import RunResult, Runtime


class GraphState(TypedDict, total=False):
    """Checkpointed working state.

    Local state, written inside the run's own workspace beside the extracted text and the
    staged files, so it is exactly as sensitive as the workspace and must never leave the
    machine. The export boundary is TelemetryEnvelope, and `disable_autotracing` stops
    LangChain shipping node payloads around it.
    """

    run_id: str
    workflow_id: str
    outcome: str
    detail: str
    approved: bool
    planned_output_root: str | None


def disable_autotracing() -> None:
    """Stop LangChain exporting graph node inputs and outputs."""
    if os.environ.get("SHAKESPEARE_ALLOW_LANGCHAIN_TRACING") == "1":
        return
    for name in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
        os.environ[name] = "false"


@contextmanager
def sqlite_checkpointer(path: Path) -> Iterator[Any]:
    from langgraph.checkpoint.sqlite import SqliteSaver

    disable_autotracing()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
    if path.exists():
        path.chmod(0o600)


@dataclass
class WorkflowGraph:
    runtime: Runtime
    request: RequestContract
    require_approval: bool = True
    planned: RunResult | None = field(default=None, init=False)

    def build(self, checkpointer: Any) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph: Any = StateGraph(GraphState)
        graph.add_node("pursue", self._pursue)
        graph.add_node("approve", self._approve)
        graph.add_node("commit", self._commit)
        graph.add_node("abandon", self._abandon)

        graph.add_edge(START, "pursue")
        graph.add_conditional_edges(
            "pursue", self._after_pursue, {"approve": "approve", "abandon": "abandon"}
        )
        graph.add_conditional_edges(
            "approve", self._after_approval, {"commit": "commit", "abandon": "abandon"}
        )
        graph.add_edge("commit", END)
        graph.add_edge("abandon", END)
        return graph.compile(checkpointer=checkpointer)

    # -- nodes --------------------------------------------------------------------------

    def _pursue(self, state: GraphState) -> dict[str, Any]:
        result = self.runtime.run(self.request, commit=False)
        self.planned = result
        return {
            "run_id": result.run_id,
            "workflow_id": result.workflow_id,
            "outcome": "" if result.outcome == "planned" else result.outcome,
            "detail": result.detail,
            "planned_output_root": result.planned_output_root,
        }

    def _approve(self, state: GraphState) -> dict[str, Any]:
        if not self.require_approval:
            return {"approved": True}
        from langgraph.types import interrupt

        plan = self.planned.plan if self.planned else None
        decision = interrupt(
            {
                "kind": "commit_approval",
                "summary": {
                    "run_id": state.get("run_id"),
                    "entries": len(plan.entries) if plan else 0,
                    "changed": sum(1 for e in plan.entries if e.action == "changed")
                    if plan
                    else 0,
                    "unresolved": sum(1 for e in plan.entries if e.action == "unresolved")
                    if plan
                    else 0,
                    "output_root": state.get("planned_output_root"),
                    "plan_fingerprint": plan.fingerprint() if plan else None,
                },
            }
        )
        return {"approved": bool(decision)}

    def _commit(self, state: GraphState) -> dict[str, Any]:
        assert self.planned is not None
        committed = self.runtime.commit_planned(self.planned)
        return {"outcome": committed.outcome, "detail": committed.detail}

    def _abandon(self, state: GraphState) -> dict[str, Any]:
        run_id = state.get("run_id")
        if run_id:
            mutation.discard(self.runtime.workspace_root / run_id / "staging")
        return {"outcome": state.get("outcome") or "abandoned"}

    # -- edges --------------------------------------------------------------------------

    @staticmethod
    def _after_pursue(state: GraphState) -> str:
        return "abandon" if state.get("outcome") else "approve"

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
    """Start a durable run. Returns (state-or-interrupt, graph, thread id)."""
    from uuid import uuid4

    graph = WorkflowGraph(
        runtime=runtime, request=request, require_approval=require_approval
    )
    compiled = graph.build(checkpointer)
    thread = thread_id or uuid4().hex
    config = {"configurable": {"thread_id": thread}}
    state = compiled.invoke({}, config)
    graph.compiled = compiled  # type: ignore[attr-defined]
    graph.config = config  # type: ignore[attr-defined]
    return state, graph, thread


def resume(graph: WorkflowGraph, *, approved: bool) -> RunResult:
    """Continue a suspended run after a human decision."""
    from langgraph.types import Command

    state = graph.compiled.invoke(Command(resume=approved), graph.config)  # type: ignore[attr-defined]
    return RunResult(
        run_id=state.get("run_id", ""),
        workflow_id=state.get("workflow_id", ""),
        outcome=state.get("outcome", "unknown"),
        plan=graph.planned.plan if graph.planned else None,
        committed_to=state.get("planned_output_root")
        if state.get("outcome") == "committed"
        else None,
        detail=state.get("detail", ""),
    )
