"""Command line interface.

Every command goes through `bootstrap.build_runtime`, so the CLI has no privileged path
of its own and a GUI can be layered on later without a second one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .capabilities import CapabilityRegistryError
from .components.filesystem_mutation import mutation
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
from .model_access import GatewayError
from .planning.planner import FakePlanner, Planner
from .runtime.audit.metrics import snapshot
from .services import Services, build_runtime, default_state_root
from .workflows import WorkflowRegistryError

if TYPE_CHECKING:
    from .measurements import Proposal

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
measurements_app = typer.Typer(
    help="What runs measured, and which declared constant it supports.",
    no_args_is_help=True,
)
app.add_typer(workflows_app, name="workflows")
app.add_typer(journal_app, name="journal")
app.add_typer(requests_app, name="requests")
app.add_typer(prompts_app, name="prompts")
app.add_typer(canary_app, name="canary")
app.add_typer(measurements_app, name="measurements")

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


@requests_app.command("gaps")
def requests_gaps(
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Requests nothing registered could serve, and what each would take to serve.

    A workflow is a saved process. This is the list of processes nobody has saved yet,
    ordered by how often they have been asked for — which is the order worth building in.
    """
    from rich.table import Table

    gaps = _services(state_root).audit.capability_gaps()
    if not gaps:
        console.print("[dim]No unmet requests recorded.[/dim]")
        return
    table = Table(title="Unmet requests")
    table.add_column("asked", justify="right")
    table.add_column("prompt")
    table.add_column("would need")
    for gap in gaps:
        table.add_row(
            str(gap["asked"]),
            gap["prompt_digest"][:12],
            "\n".join(f"· {item}" for item in gap["requires"]) or gap["rationale"],
        )
    console.print(table)


@app.command()
def calibrate(
    prompt: Annotated[str, typer.Option("--prompt", "-p")],
    input_root: Annotated[Path, typer.Option("--input", "-i", exists=True, file_okay=False)],
    truth: Annotated[
        Path,
        typer.Option(
            "--truth",
            exists=True,
            dir_okay=False,
            help="JSON or YAML mapping each input relpath to its true field values.",
        ),
    ],
    precision: Annotated[
        float, typer.Option("--precision", min=0.0, max=1.0, help="Accuracy a floor must reach.")
    ] = 0.99,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Measure what a reported confidence has actually been worth.

    Plans without touching the filesystem, then compares every field the run claimed
    against what is true. The floor it suggests is the lowest one the evidence supports:
    every point above that quarantines files a person then renames by hand.
    """
    import yaml

    from . import calibration

    expected = (
        json.loads(truth.read_text())
        if truth.suffix == ".json"
        else yaml.safe_load(truth.read_text())
    )
    services = _services(state_root)
    result = services.runtime.run(
        RequestContract(
            request_id=uuid4().hex,
            prompt=prompt,
            input_root=str(input_root.resolve()),
            output_root=str(input_root.parent / "calibration-not-written"),
        ),
        commit=False,
    )
    rows = _claims_of(result)
    if not rows:
        _fail("the run reported no field values, so there is nothing to measure")
    per_field = calibration.observe(rows, expected)
    report = calibration.report(per_field, targets=(precision,))
    _render_calibration(report, precision)

    # Kept, not just printed. A floor derived from one sitting describes one corpus; the
    # question of what a claimed confidence is worth has stayed open across two ADRs
    # precisely because every measurement of it was discarded when the process exited.
    recorded = _remember_claims(services, result, per_field)
    if recorded:
        console.print(
            f"[dim]Recorded {recorded} claims. "
            f"`shakespeare measurements propose` reads them alongside earlier runs.[/dim]"
        )

    # A calibration measured over part of a corpus is a calibration of the easy part.
    # Saying so is the difference between a number and a misleading number.
    measured = len({row.get("relpath") for row in rows if row.get("relpath") in expected})
    if measured < len(expected):
        console.print(
            f"[yellow]Measured {measured} of {len(expected)} inputs — the run did not "
            f"resolve the rest, so this describes a subset, not the corpus.[/yellow]"
        )
    if result.outcome != "planned":
        _report(result)
        raise typer.Exit(code=1)


def _remember_claims(
    services: Services, result: object, per_field: dict[str, list[tuple[float, bool]]]
) -> int:
    """Record each claimed confidence against whether the value beside it was right."""
    from .contracts import Measurement, MeasurementKind

    run_id = getattr(result, "run_id", "")
    models = services.audit.resolved_models(run_id)
    if not run_id or not models:
        # No model identity means no measurement: a claim is evidence about the model
        # that made it, and a run whose model is unknown cannot contribute to a floor.
        return 0
    measurements = [
        Measurement(
            kind=MeasurementKind.CONFIDENCE,
            subject=field,
            resolved_model=models[0],
            value=confidence,
            outcome=correct,
        )
        for field, claims in sorted(per_field.items())
        for confidence, correct in claims
    ]
    return services.audit.record_measurements(run_id=run_id, measurements=measurements)


def _claims_of(result: object) -> list[dict[str, object]]:
    """Every rendered item's claimed values, keyed by the relpath the truth file uses."""
    context: dict[str, Any] = {}
    for attempt in getattr(result, "attempts", ()):
        context.update(attempt.outcome.context)
    relpath = {
        item["item_id"]: item["relpath"]
        for item in context.get("items") or []
        if isinstance(item, dict) and "relpath" in item
    }
    return [
        {**row, "relpath": relpath.get(row.get("item_id"), "")}
        for row in context.get("results") or []
        if isinstance(row, dict)
    ]


def _render_calibration(report: object, precision: float) -> None:
    from rich.table import Table

    bands = Table(title="Reported confidence against what was true")
    for column in ("band", "claims", "claimed", "observed", "gap"):
        bands.add_column(column, justify="right" if column != "band" else "left")
    for band in report.bands:  # type: ignore[attr-defined]
        bands.add_row(
            f"{band.lower:.1f}–{band.upper:.1f}",
            str(band.count),
            f"{band.claimed:.2f}",
            f"{band.observed:.2f}",
            f"{band.gap:+.2f}",
        )
    console.print(bands)

    per_field = Table(title="Accuracy by field")
    per_field.add_column("field")
    per_field.add_column("claims", justify="right")
    per_field.add_column("correct", justify="right")
    for name, (count, accuracy) in report.fields.items():  # type: ignore[attr-defined]
        per_field.add_row(name, str(count), f"{accuracy:.1%}")
    console.print(per_field)

    floor = next(iter(report.floors.values()))  # type: ignore[attr-defined]
    console.print(
        f"[bold]{report.observations}[/bold] claims · "  # type: ignore[attr-defined]
        f"Brier [bold]{report.brier:.4f}[/bold] · "  # type: ignore[attr-defined]
        f"mean error [bold]{report.expected_error:.3f}[/bold] · "  # type: ignore[attr-defined]
        f"{'overconfident' if report.overconfident else 'not overconfident'}"  # type: ignore[attr-defined]
    )
    if floor is None:
        console.print(
            f"[yellow]No floor reaches {precision:.0%} on this evidence — the claims do "
            f"not separate right from wrong.[/yellow]"
        )
    else:
        console.print(f"Lowest floor reaching {precision:.0%} accuracy: [bold]{floor}[/bold]")


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
    from .components.readonly_scan.inspection import digest_file

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
    from .runtime.replay import ReplayError, assert_same_workflow, journal_components

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

    from .runtime.audit import schema

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


@app.command("operator")
def operator_show(name: str) -> None:
    """Everything about one operator, in one place.

    An operator's definition is necessarily spread over several files: the spec, its
    argument model, its produced keys, its marshalling and its logic. A reader should not
    have to visit all of them to learn what to expect, so this assembles the whole picture
    — including the family contract, which is what actually bounds the operator's
    behaviour.
    """
    from .components import families
    from .components.arguments import OUTPUT_KEYS, argument_summary
    from .components.catalog import RUNTIME_ONLY

    services = _services(planner=_no_model(), agents={})
    if name not in services.operators:
        known = sorted(spec.name for spec in services.operators.specs())
        _fail(f"no operator named {name!r}. Registered: {', '.join(known)}")
    spec = services.operators.get(name).spec

    console.print(f"[bold]{spec.name}[/bold] {spec.version} — {spec.description}")
    facts = Table(show_header=False, box=None, pad_edge=False)
    facts.add_column(style="dim")
    facts.add_column()
    facts.add_row("family", str(spec.family))
    facts.add_row("runner", spec.entrypoint)
    facts.add_row("operation", ", ".join(sorted(spec.features)) or "—")
    facts.add_row("risk", str(spec.risk))
    facts.add_row("access", "runtime only" if name in RUNTIME_ONLY else "composable")
    facts.add_row("idempotent", "yes" if spec.idempotent else "no")
    facts.add_row("timeout", f"{spec.timeout_seconds:g}s")
    facts.add_row("side effects", ", ".join(spec.side_effects) or "none")
    console.print(facts)

    summary = argument_summary(name)
    if summary:
        arguments = Table(title="Arguments", title_justify="left")
        arguments.add_column("name", style="bold")
        arguments.add_column("required")
        arguments.add_column("where it comes from")
        for kind in ("required", "optional"):
            for entry in summary.get(kind, []):
                arguments.add_row(
                    entry["name"], "yes" if kind == "required" else "", entry.get("note", "")
                )
        console.print(arguments)
    produced = OUTPUT_KEYS.get(name)
    console.print(
        f"[dim]produces:[/dim] {', '.join(produced) if produced else 'nothing an agent binds'}"
    )

    # The family card is the operator's real contract: lifecycle, risks and failure modes
    # are declared once per family and inherited, not restated per operator.
    _, card = families.load_all()[spec.family]
    contract = Table(title=f"{spec.family} contract", title_justify="left", show_header=False)
    contract.add_column(style="dim")
    contract.add_column(overflow="fold")
    for field in ("lifecycle", "side_effects", "risks", "failure_modes", "resource_limits"):
        contract.add_row(field.replace("_", " "), getattr(card, field).strip())
    console.print(contract)


@app.command("operators")
def operators_list() -> None:
    from .components.catalog import RUNTIME_ONLY

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
    from .prompt_store import PromptStore

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
    from .tuning.compile import OptimizeError, require_dspy

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
    from .prompt_store import PromptStore
    from .tuning import PromotionGate, PromotionOutcome

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
# Measurements
# --------------------------------------------------------------------------------------


@measurements_app.command("list")
def measurements_list(
    kind: Annotated[
        str, typer.Option("--kind", help="schedule_cost or confidence.")
    ] = "schedule_cost",
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """What has been measured, and how much of it there is."""
    from .contracts import MeasurementKind

    try:
        selected = MeasurementKind(kind)
    except ValueError:
        _fail(f"unknown kind {kind!r}; use one of {[str(k) for k in MeasurementKind]}")
    services = _services(state_root, planner=_no_model(), agents={})
    rows = services.audit.measured_subjects(selected)
    if not rows:
        console.print(
            f"[yellow]No {kind} measurements recorded yet.[/yellow]\n"
            f"[dim]Runs record them as they go; a fake-model run measures nothing.[/dim]"
        )
        return
    table = Table(title=f"Measured · {kind}")
    table.add_column("subject", style="bold")
    table.add_column("model")
    table.add_column("observations", justify="right")
    table.add_column("runs", justify="right")
    for row in rows:
        table.add_row(
            row["subject"], row["resolved_model"], str(row["observations"]), str(row["runs"])
        )
    console.print(table)


@measurements_app.command("propose")
def measurements_propose(
    subject: Annotated[
        str | None,
        typer.Argument(help="Capability id. Omit to assess every measured capability."),
    ] = None,
    precision: Annotated[
        float,
        typer.Option(
            "--precision", min=0.0, max=1.0, help="Accuracy a confidence floor must reach."
        ),
    ] = 0.99,
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """Say which declared constant the recorded evidence supports.

    Prints a proposal and changes nothing. A measured constant reaches a run by being
    written into the manifest or config that declares it, so the change is a versioned
    edit visible in git rather than state accumulating in a database nobody reviews.
    """
    from . import measurements
    from .contracts import MeasurementKind

    services = _services(state_root, planner=_no_model(), agents={})
    proposals: list[tuple[measurements.Proposal, str]] = []

    # Keyed on what is declared now. Evidence recorded under an older capability version
    # or an older pinned prompt describes something else, so it is set aside rather than
    # silently averaged into a number somebody is about to write down.
    seen: set[tuple[str, str]] = set()
    for row in services.audit.measured_subjects(MeasurementKind.SCHEDULE_COST):
        capability_id = row["subject"].split("@")[0]
        if subject and capability_id != subject:
            continue
        try:
            capability = services.capabilities.get(capability_id)
        except Exception:
            console.print(
                f"[dim]{row['subject']}: measured, but no capability declares it now.[/dim]"
            )
            continue
        if (capability.ref, row["resolved_model"]) in seen:
            continue
        seen.add((capability.ref, row["resolved_model"]))
        evidence = measurements.applicable(
            [
                measured
                for measured in services.audit.measurements(
                    kind=MeasurementKind.SCHEDULE_COST, resolved_model=row["resolved_model"]
                )
                # This capability's own history. Another capability's cost is not
                # weaker evidence about this one; it is evidence about something else.
                if measured["subject"].split("@")[0] == capability_id
            ],
            subject=capability.ref,
            prompt_version=capability.prompt_version,
        )
        if not evidence.rows:
            # A hollow panel with no subject and no model reads as a broken command
            # rather than as an answer. The answer is that the evidence is about
            # something this capability no longer is.
            console.print(
                f"[dim]{capability.ref}: nothing measured under what is declared now "
                f"({evidence.summary or 'no observations'}).[/dim]"
                if evidence.set_aside
                else f"[dim]{capability.ref}: nothing measured yet.[/dim]"
            )
            continue
        proposals.append(
            (
                measurements.cost_proposal(evidence.rows, incumbent=capability.cost_per_item),
                f"cost_per_item in shakespeare/capabilities/{capability_id}/capability.yml",
            )
        )
        if evidence.set_aside:
            console.print(
                f"[dim]{capability.ref}: set aside {evidence.summary} — "
                f"a different version or prompt is a different thing to measure.[/dim]"
            )

    if not subject:
        claims = services.audit.measurements(kind=MeasurementKind.CONFIDENCE)
        if claims:
            for model in sorted({row["resolved_model"] for row in claims}):
                proposals.append(
                    (
                        measurements.floor_proposal(
                            [row for row in claims if row["resolved_model"] == model],
                            incumbent=_declared_floor(),
                            precision=precision,
                        ),
                        "floor in configs/confidence/*.yaml",
                    )
                )

    if not proposals:
        console.print(
            "[yellow]Nothing measured yet.[/yellow]\n"
            "[dim]Cost is recorded by any real run; confidence by `shakespeare calibrate`.[/dim]"
        )
        return
    for proposal, where in proposals:
        _render_proposal(proposal, where)


@measurements_app.command("shapes")
def measurements_shapes(
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """How each shape has fared when the planner actually had a choice.

    The planner picks between capabilities from their *declared* per-item cost and the
    size of the corpus. This is the other half: what happened after it picked. A report
    only — feeding it back into the choice would make a run depend on state its journal
    does not pin.
    """
    from . import measurements
    from .contracts import MeasurementKind

    services = _services(state_root, planner=_no_model(), agents={})
    rows = services.audit.measurements(kind=MeasurementKind.SHAPE_CHOICE)
    if not rows:
        console.print(
            "[yellow]No shape choices recorded yet.[/yellow]\n"
            "[dim]Only goals that several capabilities could answer are recorded: "
            "a goal with one candidate was not chosen for.[/dim]"
        )
        return
    runs = [row["run_id"] for row in rows]
    found = measurements.shapes(
        rows, costs=services.audit.run_costs(runs), endings=services.audit.run_endings(runs)
    )
    table = Table(title="Shapes the planner chose, and what followed")
    table.add_column("capability", style="bold")
    table.add_column("chosen", justify="right")
    table.add_column("goal satisfied", justify="right")
    table.add_column("corpus", justify="right")
    table.add_column("runs ended")
    table.add_column("median run cost", justify="right")
    for shape in found:
        low, high = shape.corpus
        table.add_row(
            shape.subject,
            str(shape.chosen),
            f"{shape.satisfied}/{shape.chosen} ({shape.rate:.0%})",
            f"{low}" if low == high else f"{low}–{high}",
            " ".join(f"{name}:{count}" for name, count in sorted(shape.endings.items())),
            f"${shape.median_run_cost:.4f}" if shape.median_run_cost is not None else "—",
        )
    console.print(table)
    console.print(
        "[dim]Run cost is the whole run's, not this goal's — a run can be expensive for "
        "reasons that have nothing to do with the shape chosen.[/dim]"
    )


@measurements_app.command("recovery")
def measurements_recovery(
    state_root: Annotated[Path | None, typer.Option("--state", hidden=True)] = None,
) -> None:
    """How often a goal that failed was worth another attempt.

    Needs no measurement of its own: whether a retry ever recovered is already a fact of
    the audit log. `max_goal_attempts` is a chosen number, and this is what would replace
    the choosing.
    """
    from . import measurements

    services = _services(state_root, planner=_no_model(), agents={})
    rows = measurements.recovery(services.audit.attempts_by_goal())
    if not rows:
        console.print("[yellow]No attempts recorded yet.[/yellow]")
        return
    table = Table(title="Attempts, and whether a retry was worth making")
    table.add_column("goal", style="bold")
    table.add_column("attempts by number")
    table.add_column("deepest recovery", justify="right")
    table.add_column("spent past it", justify="right")
    for row in rows:
        spread = " ".join(
            f"{number}:{met}/{reached}"
            for number, (reached, met) in sorted(row.by_attempt.items())
        )
        deepest = row.deepest_recovery
        table.add_row(
            row.goal_id,
            spread,
            str(deepest) if deepest else "[red]never[/red]",
            f"[yellow]{row.wasted}[/yellow]" if row.wasted else "0",
        )
    console.print(table)
    console.print(
        "[dim]met/reached per attempt number. An attempt number that never recovered "
        "anywhere is budget every run is free to spend.[/dim]"
    )


def _declared_floor() -> float | None:
    """The floor the default config group declares, or None if it cannot be read."""
    import yaml

    path = Path(__file__).resolve().parent / "conventions" / "confidence" / "balanced.yaml"
    if not path.is_file():
        return None
    try:
        return float((yaml.safe_load(path.read_text()) or {})["floor"])
    except Exception:
        return None


def _render_proposal(proposal: Proposal, where: str) -> None:
    from . import measurements

    colour = {
        measurements.Verdict.SUPPORTED: "green",
        measurements.Verdict.REVIEW: "yellow",
        measurements.Verdict.INSUFFICIENT: "dim",
    }[proposal.verdict]
    lines = [
        f"[bold]{proposal.subject}[/bold]  ·  {proposal.resolved_model or 'no model'}",
        f"declared {proposal.incumbent if proposal.incumbent is not None else '—'}"
        f"   measured {proposal.candidate if proposal.candidate is not None else '—'}"
        + (f"   ({proposal.change:.2f}x)" if proposal.change else ""),
        f"[dim]{proposal.observations} observations across {proposal.runs} runs[/dim]",
        "",
        proposal.rationale,
    ]
    if proposal.detail:
        lines.append(
            "[dim]" + "  ".join(f"{k}={v}" for k, v in proposal.detail.items()) + "[/dim]"
        )
    if proposal.verdict is measurements.Verdict.SUPPORTED:
        lines.append(f"\n[dim]Pin it by setting {where}.[/dim]")
    console.print(Panel("\n".join(lines), border_style=colour, title=str(proposal.verdict)))


# --------------------------------------------------------------------------------------
# Canary runs
# --------------------------------------------------------------------------------------


@canary_app.command("list")
def canary_list(
    root: Annotated[Path | None, typer.Option("--canaries", hidden=True)] = None,
) -> None:
    """Golden cases available for drift detection."""
    from .drift import load_cases

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

    from .drift import load_cases, run_case

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

    from .drift import load_cases, record, run_case

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
    from .components.registry import FAMILY_RUNNERS
    from .components.runners import allowlist
    from .contracts import OperatorFamily

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
    from .components.families import FamilyError, check_features

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
