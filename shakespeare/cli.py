"""Command line interface.

Every command goes through `bootstrap.build_runtime`, so the CLI has no privileged path
of its own and a GUI can be layered on later without a second one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .audit.metrics import snapshot
from .bootstrap import Services, build_runtime, default_state_root
from .capabilities import CapabilityRegistryError
from .contracts import (
    AdmissionChoice,
    AdmissionDecision,
    ChangeAction,
    ChangePlan,
    DecidedBy,
    OptimizationRun,
    RequestContract,
    ReversalRecord,
)
from .gateway import GatewayError
from .operators import mutation
from .planner import FakePlanner, Planner
from .workflows import WorkflowRegistryError

app = typer.Typer(
    name="shakespeare",
    help="Staged, transactional, agent-driven file operations.",
    no_args_is_help=True,
    add_completion=False,
)
workflows_app = typer.Typer(help="Inspect registered workflows.", no_args_is_help=True)
journal_app = typer.Typer(help="Read the immutable audit log.", no_args_is_help=True)
requests_app = typer.Typer(help="Review requested operators.", no_args_is_help=True)
prompts_app = typer.Typer(help="Inspect and promote prompt artifacts.", no_args_is_help=True)
canary_app = typer.Typer(help="Golden-fixture drift detection.", no_args_is_help=True)
app.add_typer(workflows_app, name="workflows")
app.add_typer(journal_app, name="journal")
app.add_typer(requests_app, name="requests")
app.add_typer(prompts_app, name="prompts")
app.add_typer(canary_app, name="canary")

console = Console()
err = Console(stderr=True)


def _fail(message: str) -> None:
    err.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


def _services(state_root: Path | None = None, **overrides: object) -> Services:
    """Build the runtime, reporting configuration problems as messages not tracebacks."""
    try:
        return build_runtime(state_root=state_root, **overrides)  # type: ignore[arg-type]
    except GatewayError as exc:
        _fail(
            f"{exc}\n\nSet SHAKESPEARE_MODEL to a fixed LiteLLM model id, for example:\n"
            f"  export SHAKESPEARE_MODEL=openrouter/openai/gpt-5-mini"
        )
    except (CapabilityRegistryError, WorkflowRegistryError) as exc:
        _fail(f"package error: {exc}")
    raise AssertionError("unreachable")


def _render_plan(plan: ChangePlan) -> None:
    table = Table(title=f"Change plan · {plan.workflow_id}", show_lines=False)
    table.add_column("Action", style="bold", no_wrap=True)
    table.add_column("From")
    table.add_column("To / reason")
    for entry in plan.entries:
        target = getattr(entry, "target_relpath", None) or entry.reason or "—"
        colour = {
            ChangeAction.CHANGED: "green",
            ChangeAction.UNCHANGED: "dim",
            ChangeAction.UNRESOLVED: "yellow",
        }[entry.action]
        table.add_row(f"[{colour}]{entry.action}[/{colour}]", entry.source_ref, target)
    console.print(table)
    console.print(
        f"  changed {plan.count(ChangeAction.CHANGED)}"
        f" · unchanged {plan.count(ChangeAction.UNCHANGED)}"
        f" · unresolved {plan.count(ChangeAction.UNRESOLVED)}"
    )


def _report(result: object) -> None:
    outcome = getattr(result, "outcome", "unknown")
    if outcome == "committed":
        console.print(
            Panel(
                f"Committed to [bold]{getattr(result, 'committed_to', '')}[/bold]\n"
                f"run {getattr(result, 'run_id', '')}",
                border_style="green",
            )
        )
        return
    detail = getattr(result, "detail", "")
    code = getattr(result, "error_code", None)
    err.print(
        Panel(
            f"{outcome}{f' ({code})' if code else ''}\n{detail}",
            border_style="yellow" if outcome == "planned" else "red",
        )
    )


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


@app.command()
def run(
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="What to do.")],
    input_root: Annotated[Path, typer.Option("--input", "-i", exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option("--output", "-o")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Commit without confirming.")] = False,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Plan, preview, and (after confirmation) commit."""
    services = _services(state_root)
    request = RequestContract(
        request_id=uuid4().hex,
        prompt=prompt,
        input_root=str(input_root.resolve()),
        output_root=str(output_root.expanduser()),
    )
    planned = services.runtime.run(request, commit=False)
    if planned.plan is None:
        _report(planned)
        raise typer.Exit(code=1)

    _render_plan(planned.plan)
    if not yes and not typer.confirm("Commit this plan?", default=False):
        console.print("[dim]Nothing was written.[/dim]")
        raise typer.Exit(code=0)

    # Commit the plan that was shown, not a freshly derived one.
    _report(services.runtime.commit_planned(planned))


@app.command("plan")
def plan_only(
    prompt: Annotated[str, typer.Option("--prompt", "-p")],
    input_root: Annotated[Path, typer.Option("--input", "-i", exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option("--output", "-o")],
    plan_out: Annotated[
        Path | None, typer.Option("--plan-out", help="Write the plan as JSON.")
    ] = None,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Produce a plan without touching the filesystem."""
    services = _services(state_root)
    result = services.runtime.run(
        RequestContract(
            request_id=uuid4().hex,
            prompt=prompt,
            input_root=str(input_root.resolve()),
            output_root=str(output_root.expanduser()),
        ),
        commit=False,
    )
    if result.plan is None:
        _report(result)
        raise typer.Exit(code=1)
    _render_plan(result.plan)
    if plan_out:
        plan_out.write_text(result.plan.model_dump_json(indent=2))
        console.print(f"[dim]Plan written to {plan_out}[/dim]")


@app.command()
def apply(
    plan_path: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    input_root: Annotated[Path, typer.Option("--input", "-i", exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option("--output", "-o")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Phase two: commit a plan produced earlier by `plan`.

    The plan's recorded source digests are re-verified against the input tree first, so a
    file that changed since planning stops the commit rather than being renamed on stale
    information.
    """
    services = _services(state_root, planner=_no_model(), agents={})
    plan = ChangePlan.model_validate_json(plan_path.read_text())

    drifted = _verify_sources(plan, input_root)
    if drifted:
        _fail(
            f"{len(drifted)} source file(s) changed since this plan was made: "
            f"{', '.join(drifted[:5])}{'...' if len(drifted) > 5 else ''}\n"
            f"Re-run `shakespeare plan` against the current tree."
        )

    _render_plan(plan)
    if not yes and not typer.confirm("Commit this plan?", default=False):
        console.print("[dim]Nothing was written.[/dim]")
        raise typer.Exit(code=0)

    staging = services.state_root / "runs" / plan.run_id / "staging"
    mutation.discard(staging)
    try:
        reversals = mutation.stage_plan(
            plan=plan, input_root=input_root.resolve(), staging_root=staging
        )
        report = mutation.verify_tree(plan=plan, staging_root=staging)
        if not report["ok"]:
            mutation.discard(staging)
            _fail(f"staging does not match the plan: {report}")
        for record in reversals:
            services.audit.record_mutation(
                run_id=plan.run_id,
                target_ref=str(record.payload.get("target", "")),
                operation=record.operation,
                reversal=record,
            )
        record = mutation.commit(
            staging_root=staging, output_root=output_root.expanduser().resolve()
        )
    except mutation.MutationError as exc:
        mutation.discard(staging)
        _fail(str(exc))
        return

    services.audit.record_mutation(
        run_id=plan.run_id,
        target_ref=str(output_root),
        operation="commit",
        reversal=record,
    )
    console.print(
        Panel(f"Committed to [bold]{output_root}[/bold]\nrun {plan.run_id}", border_style="green")
    )


def _verify_sources(plan: ChangePlan, input_root: Path) -> list[str]:
    """Source paths whose contents no longer match the digests recorded in the plan."""
    from .operators.filesystem import digest_file

    drifted: list[str] = []
    for entry in plan.entries:
        recorded = entry.digests.get("source") or getattr(entry, "source_sha256", None)
        if not recorded:
            continue
        path = input_root / entry.source_ref
        if not path.is_file() or digest_file(path) != recorded:
            drifted.append(entry.source_ref)
    return drifted


@app.command()
def replay(
    run_id: Annotated[str, typer.Argument(help="Run to replay.")],
    input_root: Annotated[Path, typer.Option("--input", "-i", exists=True, file_okay=False)],
    output_root: Annotated[
        Path | None, typer.Option("--output", "-o", help="Commit the replay here too.")
    ] = None,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Re-execute a recorded run with zero model calls, and check it reproduces.

    Only the planner and the domain agents are swapped for journal-backed ones; the same
    verifier, executor and obligations run. A replay that reproduces the original plan is
    therefore evidence that the recorded compositions really do determine the result.
    """
    from .replay import ReplayError, assert_same_workflow, journal_components

    inspect = _services(state_root, planner=_no_model(), agents={})
    original = inspect.audit.recorded_plan(run_id)
    if original is None:
        _fail(f"run {run_id} recorded no plan to reproduce")
        return

    try:
        planner, agents, workflow_id, recorded_digest = journal_components(
            inspect.audit, run_id
        )
        assert_same_workflow(recorded_digest, inspect.workflows.get(workflow_id).digest())
    except (ReplayError, KeyError) as exc:
        _fail(str(exc))
        return

    services = _services(state_root, planner=planner, agents=agents)
    request = RequestContract(
        request_id=f"replay-{run_id}",
        prompt=f"replay of {run_id}",
        input_root=str(input_root.resolve()),
        output_root=str((output_root or Path("/dev/null/unused")).expanduser()),
    )
    try:
        result = services.runtime.run(request, commit=output_root is not None)
    except ReplayError as exc:
        _fail(str(exc))
        return

    if result.plan is None:
        _report(result)
        raise typer.Exit(code=1)

    assert planner.model_calls == 0, "replay must make no model call"
    _compare(original, result.plan)
    if output_root is not None:
        _report(result)


def _compare(original: ChangePlan, replayed: ChangePlan) -> None:
    """Compare decisions, ignoring the identifiers that must differ between runs."""

    def decisions(plan: ChangePlan) -> list[tuple[str, str, str | None]]:
        return sorted(
            (entry.source_ref, str(entry.action), getattr(entry, "target_relpath", None))
            for entry in plan.entries
        )

    before, after = decisions(original), decisions(replayed)
    if before == after:
        console.print(
            Panel(
                f"Reproduced exactly · {len(after)} entries · zero model calls",
                border_style="green",
            )
        )
        return

    table = Table(title="Replay diverged from the recorded run")
    table.add_column("source")
    table.add_column("recorded")
    table.add_column("replayed")
    recorded = {item[0]: item for item in before}
    produced = {item[0]: item for item in after}
    for source in sorted(set(recorded) | set(produced)):
        first, second = recorded.get(source), produced.get(source)
        if first != second:
            table.add_row(
                source,
                f"{first[1]} → {first[2]}" if first else "—",
                f"{second[1]} → {second[2]}" if second else "—",
            )
    err.print(table)
    raise typer.Exit(code=1)


@app.command()
def undo(
    run_id: Annotated[str, typer.Argument(help="Run to reverse.")],
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Reverse a committed run using its journaled reversal records."""
    from sqlalchemy import select

    from .audit import schema

    # Reversal replays journaled facts and makes no model call, so it must not require a
    # model profile to be configured.
    services = _services(state_root, planner=_no_model(), agents={})
    with services.audit.engine.begin() as connection:
        commit_row = connection.execute(
            select(schema.commits).where(schema.commits.c.run_id == run_id)
        ).mappings().first()
        records = connection.execute(
            select(schema.mutations)
            .where(schema.mutations.c.run_id == run_id)
            .where(schema.mutations.c.operation == "commit")
        ).mappings().all()

    if commit_row is None or not records:
        _fail(f"run {run_id} has no committed output to reverse")
        return

    for row in records:
        mutation.reverse(ReversalRecord.model_validate(json.loads(row["reversal"])))
    services.audit.record_reversal(
        commit_id=commit_row["commit_id"], detail={"run_id": run_id, "reversed": len(records)}
    )
    console.print(f"[green]Reversed run {run_id}[/green] ({commit_row['output_root']})")


# --------------------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------------------


@workflows_app.command("list")
def workflows_list() -> None:
    services = _services(planner=_no_model(), agents={})
    table = Table(title="Registered workflows")
    table.add_column("id", style="bold")
    table.add_column("version")
    table.add_column("goals")
    for workflow_id in services.workflows.ids():
        registered = services.workflows.get(workflow_id)
        table.add_row(
            workflow_id,
            registered.spec.version,
            ", ".join(goal.id for goal in registered.spec.goals),
        )
    console.print(table)


@workflows_app.command("validate")
def workflows_validate() -> None:
    """Check every goal graph: dependencies, capabilities, gates and required evidence."""
    services = _services(planner=_no_model(), agents={})
    for workflow_id in services.workflows.ids():
        registered = services.workflows.get(workflow_id)
        console.print(f"[green]✓[/green] {workflow_id}")
        for goal in registered.spec.graph.goals:
            after = f"  after {', '.join(goal.depends_on)}" if goal.depends_on else ""
            console.print(
                f"    [bold]{goal.id:18}[/bold] {str(goal.gate.kind):14}"
                f" requires {', '.join(goal.gate.requires) or '-'}"
                f"  via {', '.join(goal.capabilities)}{after}"
            )
    if not services.workflows.ids():
        console.print("[yellow]No workflows registered.[/yellow]")


@app.command("capabilities")
def capabilities_list() -> None:
    """Registered capabilities: what each is for, and what evidence it can produce."""
    services = _services(planner=_no_model(), agents={})
    table = Table(title="Registered capabilities")
    table.add_column("id", style="bold")
    table.add_column("standing goal")
    table.add_column("produces")
    table.add_column("rounds")
    for capability_id in services.capabilities.ids():
        spec = services.capabilities.get(capability_id)
        table.add_row(
            capability_id,
            spec.standing_goal[:48],
            ", ".join(spec.produces),
            str(spec.max_rounds),
        )
    console.print(table)


@app.command("operators")
def operators_list() -> None:
    from .operators.builtin import RUNTIME_ONLY

    services = _services(planner=_no_model(), agents={})
    table = Table(title="Registered operators")
    table.add_column("name", style="bold")
    table.add_column("family")
    table.add_column("risk")
    table.add_column("access")
    for spec in sorted(services.operators.specs(), key=lambda s: s.name):
        access = "runtime only" if spec.name in RUNTIME_ONLY else "composable"
        table.add_row(spec.name, str(spec.family), str(spec.risk), access)
    console.print(table)


@journal_app.command("show")
def journal_show(run_id: str) -> None:
    services = _services(planner=_no_model(), agents={})
    console.print_json(json.dumps(services.audit.costs(run_id), indent=2))


@journal_app.command("dag")
def journal_dag(run_id: str, stage: str) -> None:
    """Render the DAG of what actually ran, including failed attempts."""
    services = _services(planner=_no_model(), agents={})
    dag = services.audit.dag(run_id, stage)
    if not dag["attempts"]:
        _fail(f"no attempts recorded for {stage} in run {run_id}")
    for attempt in dag["attempts"]:
        verdict = attempt["verdict"] or {}
        console.print(
            f"[bold]attempt {attempt['attempt']['attempt_no']}[/bold]"
            f" · {verdict.get('decision', '?')}"
        )
        for node in attempt["nodes"]:
            mark = "[green]✓[/green]" if node["succeeded"] else "[red]✗[/red]"
            console.print(f"  {mark} {node['operator']} ({node['operator_version']})")
        for edge in attempt["edges"]:
            source = edge["from_invocation"].split(":")[-1]
            target = edge["to_invocation"].split(":")[-1]
            console.print(f"    [dim]{source} → {target}[/dim]")


@journal_app.command("costs")
def journal_costs(run_id: str) -> None:
    services = _services(planner=_no_model(), agents={})
    console.print_json(json.dumps(services.audit.costs(run_id), indent=2))


# --------------------------------------------------------------------------------------
# Operator admission
# --------------------------------------------------------------------------------------


@requests_app.command("list")
def requests_list(
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Operator requests awaiting a decision."""
    services = _services(state_root, planner=_no_model(), agents={})
    pending = services.audit.pending_admissions()
    if not pending:
        console.print("[dim]No operator requests are waiting.[/dim]")
        return
    table = Table(title="Pending operator requests")
    table.add_column("report", style="dim")
    table.add_column("operator", style="bold")
    table.add_column("family")
    table.add_column("kind")
    table.add_column("risk")
    for item in pending:
        colour = {"low": "green", "medium": "yellow", "high": "red"}[item["computed_risk"]]
        table.add_row(
            item["report_id"][:12],
            item["name"],
            item["family"],
            item["kind"],
            f"[{colour}]{item['computed_risk']}[/{colour}]",
        )
    console.print(table)


@requests_app.command("review")
def requests_review(
    report_id: Annotated[str, typer.Argument()],
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Show everything a reviewer needs: the package, the risk, the findings, the tests."""
    item = _pending(report_id, state_root)
    console.print(
        Panel(
            f"[bold]{item['name']}[/bold]  ({item['family']}, {item['kind']} request)\n"
            f"risk       {item['computed_risk']}\n"
            f"digest     {item['package_digest'][:16]}\n"
            f"reproducible {item['reproducible']}\n\n"
            f"{item['request'].get('rationale', '')}",
            title=f"report {item['report_id'][:12]}",
            border_style="yellow",
        )
    )
    if item["findings"]:
        table = Table(title="Findings")
        table.add_column("severity")
        table.add_column("code")
        table.add_column("message")
        for finding in item["findings"]:
            table.add_row(finding["severity"], finding["code"], finding["message"])
        console.print(table)
    console.print_json(json.dumps(item["test_results"], indent=2))
    if item["kind"] == "behaviour":
        err.print(
            "[yellow]This is a behaviour request: it needs a runner operation that does "
            "not exist. No approval here can satisfy it — a person must add the vetted "
            "function to shakespeare/runners.py.[/yellow]"
        )


@requests_app.command("approve")
def requests_approve(
    report_id: Annotated[str, typer.Argument()],
    rationale: Annotated[str, typer.Option("--why")] = "approved after review",
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Admit a requested operator."""
    _decide(report_id, AdmissionChoice.APPROVE, rationale, state_root)


@requests_app.command("deny")
def requests_deny(
    report_id: Annotated[str, typer.Argument()],
    rationale: Annotated[str, typer.Option("--why")] = "denied after review",
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Refuse a requested operator. It never enters the registry."""
    _decide(report_id, AdmissionChoice.DENY, rationale, state_root)


def _pending(report_id: str, state_root: Path | None) -> dict[str, Any]:
    services = _services(state_root, planner=_no_model(), agents={})
    matches = [
        item
        for item in services.audit.pending_admissions()
        if item["report_id"].startswith(report_id)
    ]
    if not matches:
        _fail(f"no pending request matches {report_id!r}")
    if len(matches) > 1:
        _fail(f"{report_id!r} is ambiguous: {[m['report_id'][:12] for m in matches]}")
    return matches[0]


def _decide(
    report_id: str, choice: AdmissionChoice, rationale: str, state_root: Path | None
) -> None:
    item = _pending(report_id, state_root)
    if item["kind"] == "behaviour" and choice is AdmissionChoice.APPROVE:
        _fail(
            "a behaviour request cannot be approved: it needs a runner operation that "
            "does not exist. Add the vetted function to shakespeare/runners.py instead."
        )
    services = _services(state_root, planner=_no_model(), agents={})
    decision = AdmissionDecision(
        decision_id=uuid4().hex,
        report_id=str(item["report_id"]),
        decided_by=DecidedBy.HUMAN,
        choice=choice,
        rationale=rationale,
    )
    services.audit.record_admission_decision(decision)
    verb = "Approved" if choice is AdmissionChoice.APPROVE else "Denied"
    colour = "green" if choice is AdmissionChoice.APPROVE else "yellow"
    console.print(f"[{colour}]{verb}[/{colour}] {item['name']} ({item['report_id'][:12]})")


# --------------------------------------------------------------------------------------
# Prompt artifacts
# --------------------------------------------------------------------------------------


@prompts_app.command("list")
def prompts_list(
    prompt_root: Annotated[Path | None, typer.Option("--prompts", hidden=True)] = None,
) -> None:
    """Every prompt artifact, and which version each capability pins."""
    from .prompts import PromptStore

    services = _services(planner=_no_model(), agents={})
    store = PromptStore(prompt_root)
    pinned = {
        capability_id: ("capability", services.capabilities.get(capability_id).prompt_version)
        for capability_id in services.capabilities.ids()
    }
    table = Table(title="Prompt artifacts")
    table.add_column("signature", style="bold")
    table.add_column("versions")
    table.add_column("pinned by")
    planner_signatures = {
        "planner.route",
        "planner.select_goal",
        "planner.select_capability",
        "planner.judge_gate",
    }
    for signature in sorted(set(pinned) | planner_signatures):
        versions = store.versions(signature)
        if not versions:
            continue
        stage, version = pinned.get(signature, ("planner", "1.0.0"))
        marked = ", ".join(f"[green]{v}[/green]" if v == version else v for v in versions)
        table.add_row(signature, marked, f"{stage} @ {version}")
    console.print(table)


@prompts_app.command("compile")
def prompts_compile(signature: Annotated[str, typer.Argument()]) -> None:
    """Optimize a prompt offline with DSPy."""
    from .optimize.compile import OptimizeError, require_dspy

    try:
        require_dspy()
    except OptimizeError as exc:
        _fail(str(exc))
    _fail(
        f"compiling {signature} needs a training set. Build one from the audit log with "
        f"real runs, or seed golden fixtures first; see shakespeare/optimize/."
    )


@prompts_app.command("promote")
def prompts_promote(
    signature: Annotated[str, typer.Argument()],
    candidate: Annotated[str, typer.Option("--candidate", help="Candidate version.")],
    candidate_score: Annotated[float, typer.Option("--score")],
    incumbent: Annotated[str | None, typer.Option("--incumbent")] = None,
    incumbent_score: Annotated[float | None, typer.Option("--incumbent-score")] = None,
    regressed: Annotated[
        list[str] | None, typer.Option("--regressed", help="Golden fixture that regressed.")
    ] = None,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
    prompt_root: Annotated[Path | None, typer.Option("--prompts", hidden=True)] = None,
) -> None:
    """Assess a compiled prompt against the promotion gate."""
    from .optimize import PromotionGate, PromotionOutcome
    from .prompts import PromptStore

    store = PromptStore(prompt_root)
    services = _services(state_root, planner=_no_model(), agents={})
    run = OptimizationRun(
        optimization_id=uuid4().hex,
        signature_id=signature,
        optimizer="manual",
        eval_set_digest="0" * 64,
        incumbent_version=incumbent,
        incumbent_score=incumbent_score,
        candidate_version=candidate,
        candidate_score=candidate_score,
        fixture_regressions=tuple(regressed or ()),
    )
    services.audit.record_optimization(run)
    decision, outcome = PromotionGate().decide(
        run,
        incumbent=store.load(signature, incumbent) if incumbent else None,
        candidate=store.load(signature, candidate),
    )
    services.audit.record_promotion(decision)

    colour = {
        PromotionOutcome.AUTO_PROMOTE: "green",
        PromotionOutcome.HUMAN_REVIEW: "yellow",
        PromotionOutcome.REJECT: "red",
    }[outcome]
    console.print(Panel(f"{outcome}\n{decision.rationale}", border_style=colour))
    if outcome is PromotionOutcome.AUTO_PROMOTE:
        console.print(
            f"[dim]Pin it by setting prompt_version: \"{candidate}\" in the stage "
            f"package that uses {signature}.[/dim]"
        )


# --------------------------------------------------------------------------------------
# Canary runs
# --------------------------------------------------------------------------------------


@canary_app.command("list")
def canary_list(
    root: Annotated[Path | None, typer.Option("--canaries", hidden=True)] = None,
) -> None:
    """Golden cases available for drift detection."""
    from .canary import load_cases

    cases = load_cases(root)
    if not cases:
        console.print(
            "[dim]No canary cases. Create _canaries/<name>/case.yml with a prompt and an "
            "inputs/ tree, then run `canary record <name>`.[/dim]"
        )
        return
    table = Table(title="Canary cases")
    table.add_column("name", style="bold")
    table.add_column("entries")
    table.add_column("prompt")
    for case in cases:
        table.add_row(
            case.name,
            str(len(case.expected)) if case.has_expectation else "[yellow]unrecorded[/yellow]",
            case.prompt[:60],
        )
    console.print(table)


@canary_app.command("run")
def canary_run(
    name: Annotated[str | None, typer.Argument(help="One case, or all of them.")] = None,
    root: Annotated[Path | None, typer.Option("--canaries", hidden=True)] = None,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Re-run golden cases and report any drift.

    This uses the real model on purpose: the point is to notice when the same prompt over
    the same files stops producing the same answer — a promoted prompt, a new operator
    version, or a provider changing silently behind an alias.
    """
    import tempfile

    from .canary import load_cases, run_case

    cases = [case for case in load_cases(root) if name is None or case.name == name]
    if not cases:
        _fail(f"no canary case named {name!r}" if name else "no canary cases are defined")

    services = _services(state_root)
    drifted = []
    with tempfile.TemporaryDirectory() as scratch:
        for case in cases:
            if not case.has_expectation:
                err.print(f"[yellow]skipped[/yellow] {case.name}: nothing recorded yet")
                continue
            result = run_case(
                services.runtime, case, output_root=Path(scratch) / case.name
            )
            if not result.drifted:
                console.print(f"[green]✓[/green] {case.name}  ({len(result.produced)} entries)")
                continue
            drifted.append(result)
            if result.error:
                err.print(f"[red]✗[/red] {case.name}: {result.error}")
                continue
            table = Table(title=f"{case.name} drifted")
            table.add_column("source")
            table.add_column("expected")
            table.add_column("produced")
            for row in result.diff():
                table.add_row(*row)
            err.print(table)

    if drifted:
        raise typer.Exit(code=1)


@canary_app.command("record")
def canary_record(
    name: Annotated[str, typer.Argument()],
    root: Annotated[Path | None, typer.Option("--canaries", hidden=True)] = None,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Capture a case's current decisions as its expectation.

    Deliberately explicit: a canary that re-recorded itself on every run would never
    detect anything.
    """
    import tempfile

    from .canary import load_cases, record, run_case

    matches = [case for case in load_cases(root) if case.name == name]
    if not matches:
        _fail(f"no canary case named {name!r}")
    case = matches[0]

    services = _services(state_root)
    with tempfile.TemporaryDirectory() as scratch:
        request_plan = run_case(services.runtime, case, output_root=Path(scratch) / name)
    if request_plan.error:
        _fail(f"{name} did not produce a plan: {request_plan.error}")

    from .contracts import ChangeAction, ChangeEntry, ChangePlan

    plan = ChangePlan(
        run_id="recorded",
        workflow_id="",
        workflow_digest="",
        decision_digest="",
        entries=tuple(
            ChangeEntry.model_validate(
                {
                    "item_id": source,
                    "source_ref": source,
                    "action": ChangeAction(action),
                    **({"target_relpath": target} if target else {}),
                }
            )
            for source, action, target in request_plan.produced
        ),
    )
    path = record(case, plan, root)
    console.print(f"[green]Recorded[/green] {len(request_plan.produced)} entries to {path}")


@app.command()
def metrics() -> None:
    """Agent-ops SLIs, computed from the audit log."""
    services = _services(planner=_no_model(), agents={})
    console.print_json(json.dumps(snapshot(services.audit), indent=2))


# --------------------------------------------------------------------------------------
# Scaffolding
# --------------------------------------------------------------------------------------


@app.command("new-operator")
def new_operator(
    family: Annotated[str, typer.Option("--family")],
    name: Annotated[str, typer.Option("--name")],
    operation: Annotated[str, typer.Option("--operation", help="A vetted runner operation.")],
    features: Annotated[
        str, typer.Option("--features", help="Comma-separated configuration slots.")
    ] = "",
    destination: Annotated[Path, typer.Option("--to")] = Path("_operators"),
) -> None:
    """Render an operator package from its family template."""
    from .contracts import OperatorFamily
    from .registry import FAMILY_RUNNERS
    from .runners import allowlist

    try:
        resolved = OperatorFamily(family)
    except ValueError:
        _fail(f"unknown family: {family}. Choose from {[f.value for f in OperatorFamily]}")
        return
    if operation not in allowlist(resolved):
        _fail(
            f"{operation!r} is not a vetted operation for {family}. "
            f"Vetted: {sorted(allowlist(resolved))}. Adding a new one is a human change "
            f"to shakespeare/runners.py."
        )
    from .families import FamilyError, check_features

    requested = frozenset(item.strip() for item in features.split(",") if item.strip())
    try:
        check_features(resolved, requested)
    except FamilyError as exc:
        _fail(str(exc))

    _copier(
        Path("_operator_templates"),
        destination / name.replace(".", "_"),
        {
            "operator_name": name,
            "operator_summary": f"{name} ({family})",
            "operator_family": family,
            "entrypoint": FAMILY_RUNNERS[resolved],
            "runner_operation": operation,
            "features_json": json.dumps(sorted(requested)),
        },
    )


@app.command("new-stage")
def new_stage(
    name: Annotated[str, typer.Option("--name")],
    version: Annotated[str, typer.Option("--version")] = "1.0.0",
) -> None:
    """Scaffold a stage package."""
    _copier(
        Path("_scaffolds/stage_package"),
        Path("_stages") / name / version,
        {"stage_name": name, "stage_version": version},
    )


@app.command("new-workflow")
def new_workflow(workflow_id: Annotated[str, typer.Option("--id")]) -> None:
    """Scaffold a workflow package."""
    _copier(
        Path("_scaffolds/workflow_package"),
        Path("_workflows") / workflow_id,
        {"workflow_id": workflow_id},
    )


def _copier(template: Path, destination: Path, data: dict[str, str]) -> None:
    """Render through Copier's Python API rather than its console script.

    Shelling out would depend on `copier` being on PATH, which it is not when the CLI
    runs from a virtualenv that was not activated.
    """
    import copier

    if not template.is_dir():
        _fail(f"template not found: {template} (run from the repository root)")
    if destination.exists():
        _fail(f"destination already exists: {destination}")
    try:
        copier.run_copy(
            str(template), str(destination), data=data, defaults=True, quiet=True, unsafe=False
        )
    except Exception as exc:  # noqa: BLE001 - report template errors, do not traceback
        _fail(f"copier failed: {type(exc).__name__}: {exc}")
    console.print(f"[green]Rendered[/green] {destination}")


@app.command()
def gc(
    older_than_days: Annotated[int, typer.Option("--older-than-days")] = 30,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Purge workspace content. The audit log is never touched."""
    import time

    root = (state_root or default_state_root()) / "runs"
    if not root.is_dir():
        console.print("[dim]No workspaces to purge.[/dim]")
        return
    cutoff = time.time() - older_than_days * 86_400
    purged = 0
    for workspace in root.iterdir():
        if workspace.is_dir() and workspace.stat().st_mtime < cutoff:
            mutation.discard(workspace)
            purged += 1
    console.print(f"Purged {purged} workspace(s). The audit log is unchanged.")


def _no_model() -> Planner:
    """Inspection commands must not require a model profile to be configured."""
    return FakePlanner()


if __name__ == "__main__":
    app()
