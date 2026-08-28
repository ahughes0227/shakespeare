"""Choosing the shape of the work, and saying when no shape fits.

ADR 0004. For seven live runs the system computed the number that proved the model was in
the data path — items times cost against the response ceiling — used it privately to size
a batch, and showed it to nobody. The planner learned the corpus size only after routing
became irrevocable, and no return type could carry an objection.

These tests are built to force that conclusion rather than wait for it: a corpus sized so
that answering in the response is arithmetically impossible, and a run where nothing fits
at all.
"""

from __future__ import annotations

from pathlib import Path

from system.capabilities import CapabilityRegistry, CapabilityRunner
from system.capabilities.runner import Organization
from system.components.catalog import build_registry
from system.contracts import BudgetEnvelope, Invocation, RouteDecision
from system.planning.planner import ScriptedGoalPlanner
from system.runtime.artifacts import ArtifactStore
from system.runtime.executor import Budget, Executor
from system.runtime.verifier import Verifier

from harness import build, org, seed_invoices, values_for

CEILING = 16384


def corpus(count: int) -> list[dict[str, str]]:
    return [{"item_id": f"i{n}", "relpath": f"q/{n}.pdf"} for n in range(count)]


class TestTheEvidenceIsDecisive:
    """The facts reach the choice, and they are enough to settle it by arithmetic."""

    def test_the_planner_is_told_what_there_is_to_do(self, tmp_path: Path) -> None:
        planner = ScriptedGoalPlanner(route=RouteDecision(workflow_id="rename_files"))
        runtime, request, audit, _ = build(tmp_path, planner=planner)
        runtime.run(request, commit=False)
        assert planner.seen_evidence, "a choice was made, so evidence was shown"
        assert "items" in planner.seen_evidence["items"]
        assert planner.seen_evidence["response_ceiling_tokens"] == CEILING
        audit.close()

    def test_the_candidates_declare_what_settles_it(self) -> None:
        registry = CapabilityRegistry()
        answering = registry.get("resolve")
        durable = registry.get("transcribe")
        assert answering.cost_per_item and durable.cost_per_item
        assert "record.append" not in answering.catalog
        assert "record.append" in durable.catalog

    def test_sixty_documents_do_not_fit_one_response(self) -> None:
        """The number that was computed and thrown away for seven runs."""
        cost = CapabilityRegistry().get("resolve").cost_per_item
        assert cost is not None
        assert 60 * cost > CEILING, "answering in the response is arithmetically impossible"

    def test_a_handful_of_documents_do_fit(self) -> None:
        """Which is why the in-response capability is kept rather than replaced."""
        cost = CapabilityRegistry().get("resolve").cost_per_item
        assert cost is not None
        assert 5 * cost < CEILING


class TestAnImpedimentEndsTheRun:
    def test_a_planner_that_finds_no_workable_shape_escalates(self, tmp_path: Path) -> None:
        planner = ScriptedGoalPlanner(
            route=RouteDecision(workflow_id="rename_files"),
            impediments={"named": "no candidate can carry 60 documents in one response"},
        )
        runtime, request, audit, _ = build(tmp_path, planner=planner)
        result = runtime.run(request, commit=False)
        assert result.outcome == "escalated", "not aborted: no retry would have helped"
        assert "no candidate can carry" in result.detail
        audit.close()

    def test_a_capability_can_raise_one_too(self, tmp_path: Path) -> None:
        """The capability sees the mechanism; the planner sees the shape. Either may object."""
        from system.capabilities.agent import FakeCapabilityAgent

        agents = {"*": FakeCapabilityAgent()}
        agents["*"].queue(
            "survey",
            org(intent="cannot", publishes=None).model_copy(
                update={"impediment": "these are images, and nothing here reads images"}
            ),
        )
        runtime, request, audit, _ = build(tmp_path, agents=agents)
        result = runtime.run(request, commit=False)
        assert result.outcome == "escalated"
        assert "nothing here reads images" in result.detail
        audit.close()

    def test_an_escalation_is_not_an_abort_in_the_audit_log(self, tmp_path: Path) -> None:
        """A person reads one of these; the other might have been a bad roll."""
        planner = ScriptedGoalPlanner(
            route=RouteDecision(workflow_id="rename_files"),
            impediments={"named": "wrong shape"},
        )
        runtime, request, audit, _ = build(tmp_path, planner=planner)
        result = runtime.run(request, commit=False)
        from sqlalchemy import text

        with audit.engine.connect() as connection:
            recorded = list(
                connection.execute(text("select outcome, error_code from run_outcomes"))
            )
        assert recorded == [("escalated", "impediment")]
        assert result.error_code == "impediment"
        audit.close()

    def test_nothing_is_committed_when_a_run_escalates(self, tmp_path: Path) -> None:
        planner = ScriptedGoalPlanner(
            route=RouteDecision(workflow_id="rename_files"),
            impediments={"named": "wrong shape"},
        )
        runtime, request, audit, _ = build(tmp_path, planner=planner)
        runtime.run(request, commit=True)
        assert not Path(request.output_root).exists()
        audit.close()


class TestTheDurableShapeWorks:
    """Reading fills a table; naming reads it. The model never sees a filename."""

    def _runner(self, tmp_path: Path, agent):
        operators = build_registry()
        return CapabilityRunner(
            executor=Executor(operators, Verifier(operators)),
            agents={"*": agent},
            artifacts=ArtifactStore(root=tmp_path / "artifacts", run_id="r"),
            capacity=CEILING,
        )

    def test_names_are_rendered_from_the_table_in_one_call(self, tmp_path: Path) -> None:
        source = seed_invoices(tmp_path / "in")
        rows = values_for(source)
        spec = _frozen()

        class Transcriber:
            """Stores what it reads, then renders everything from storage."""

            def __init__(self) -> None:
                self.renders = 0

            def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
                batch = context.get("items") or []
                ids = {item["item_id"] for item in batch}
                store = (
                    Invocation(
                        invocation_id="w",
                        operator="record.append",
                        parameters={"rows": [r for r in rows if r["item_id"] in ids]},
                    ),
                    Invocation(invocation_id="r", operator="record.read", parameters={}),
                )
                if context.get("batch_remaining"):
                    return (
                        Organization(
                            invocations=store,
                            intent="store this batch",
                            sufficient=True,
                            publishes=None,
                        ),
                        None,
                    )
                # The last batch renders every name in the run, from the table.
                self.renders += 1
                return (
                    Organization(
                        invocations=(
                            *store,
                            Invocation(
                                invocation_id="n",
                                operator="name.render",
                                parameters={"spec": spec},
                                bindings={"items": "r.records"},
                            ),
                        ),
                        intent="store the last batch and render every name from the table",
                        sufficient=True,
                        publishes="ResolvedNames",
                    ),
                    None,
                )

        agent = Transcriber()
        capability = CapabilityRegistry().get("transcribe")
        outcome = self._runner(tmp_path, agent).run(
            capability=capability,
            request="read them into records and name them",
            context={"items": [{"item_id": r["item_id"]} for r in rows], "root": str(source)},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="200"), items=len(rows)),
            workspace=tmp_path / "work",
        )
        assert len(outcome.context["candidates"]) == len(rows)
        assert agent.renders == 1, "every name in the run came from one deterministic call"

    def test_the_table_outlives_the_response_that_filled_it(self, tmp_path: Path) -> None:
        """A batch that fails costs itself, not the run's progress."""
        from system.components.record_store import storage as records

        workspace = tmp_path / "work"
        records.append(
            workspace=workspace,
            table="items",
            rows=({"item_id": "a", "values": {"vendor": "ACME"}},),
        )
        assert records.read(workspace=workspace, table="items")["stored"] == 1


def _frozen() -> dict:
    """The convention as the runtime freezes it, not as the harness writes it."""
    from system.components.runners import pure_transform

    from harness import SPEC

    return pure_transform({"operation": "freeze_spec", "spec": SPEC}, Path("."))["spec"]


class TestTheFrozenConventionIsBinding:
    """Freezing a convention made it evidence. It did not make it binding.

    A live run committed sixty files named `2024-04-01, Umbrella Health, ...` under a
    convention that says `%Y%m`, with every gate green: the capability passed name.render
    its own `template` and `fields` instead of the frozen spec, and the renderer obliged.
    """

    @staticmethod
    def _check(payload: dict) -> object:
        from system.runtime.checks import check_convention_followed

        return check_convention_followed("convention_followed", payload)

    def _row(self, rendered: str) -> dict:
        return {
            "item_id": "a",
            "rendered": rendered,
            "extension": ".pdf",
            "values": {"invoice_date": "2024-04-01", "vendor": "Umbrella Health"},
            "confidences": {"invoice_date": 0.99, "vendor": 0.99},
        }

    def _spec(self) -> dict:
        from system.components.runners import pure_transform

        return pure_transform(
            {
                "operation": "freeze_spec",
                "spec": {
                    "template": "{invoice_date}, {vendor}",
                    "fields": [
                        {"name": "invoice_date", "kind": "date", "format": "%Y%m"},
                        {"name": "vendor"},
                    ],
                    "policy": {"separator": ", "},
                },
            },
            Path("."),
        )["spec"]

    def test_a_name_the_convention_produces_passes(self) -> None:
        result = self._check(
            {"spec": self._spec(), "results": [self._row("202404, Umbrella Health.pdf")]}
        )
        assert result.passed

    def test_a_name_from_another_convention_is_caught(self) -> None:
        """The exact failure: the date rendered raw instead of as %Y%m."""
        result = self._check(
            {
                "spec": self._spec(),
                "results": [self._row("2024-04-01, Umbrella Health.pdf")],
            }
        )
        assert not result.passed
        assert result.detail["divergent"][0]["convention"] == "202404, Umbrella Health.pdf"

    def test_a_quarantined_item_is_not_judged(self) -> None:
        """It has no name, so there is no name to disagree with."""
        row = {**self._row("x"), "rendered": None}
        result = self._check({"spec": self._spec(), "results": [row]})
        assert result.passed, "nothing diverged"
        assert result.detail["checked"] == 0

    def test_a_corpus_nothing_can_read_is_all_quarantine_and_not_a_failure(self) -> None:
        """A live run on documents with no text layer aborted here, having behaved right.

        Every file unreadable means every file quarantined, which renders no names. That
        is the safe failure working, and this check has nothing to object to. Coverage is
        `resolution_accounted`'s question, not this one.
        """
        unread = [{**self._row("x"), "rendered": None, "values": {}} for _ in range(5)]
        assert self._check({"spec": self._spec(), "results": unread}).passed

    def test_no_frozen_spec_is_a_failure_rather_than_a_pass(self) -> None:
        """A check that cannot run has not been satisfied."""
        assert not self._check({"results": [self._row("anything.pdf")]}).passed

    def test_the_named_goal_actually_runs_it(self) -> None:
        from system.capabilities import CapabilityRegistry
        from system.components.catalog import build_registry
        from system.workflows import WorkflowRegistry

        registry = WorkflowRegistry(
            capabilities=CapabilityRegistry(), operators=build_registry()
        )
        goal = registry.get("rename_files").spec.graph.goal("named")
        assert "convention_followed" in goal.gate.checks


class TestAnUnmetRequestIsRemembered:
    """A workflow is a saved process, so the list of unsaved ones is worth keeping.

    Three live requests — summarise, spreadsheet, translate — were refused correctly and
    the router's analysis of what each would take was rendered to a terminal and lost with
    the process. An operator a capability lacks has had a backlog since admission was
    written; a process nobody has saved had nowhere to be recorded at all.
    """

    def _refuse(self, tmp_path: Path, requires: tuple[str, ...] = ()):
        planner = ScriptedGoalPlanner(
            route=RouteDecision(
                workflow_id="",
                supported=False,
                rationale="the only workflow renames files",
                requires=requires,
            )
        )
        runtime, request, audit, _ = build(tmp_path, planner=planner)
        return runtime.run(request, commit=False), audit

    def test_an_unmet_request_is_not_an_invalid_composition(self, tmp_path: Path) -> None:
        """A gap is a thing to go and fill; a malformed composition is a mistake."""
        result, audit = self._refuse(tmp_path)
        assert result.outcome == "unsupported"
        assert result.error_code == "unsupported"
        audit.close()

    def test_what_it_would_take_is_kept(self, tmp_path: Path) -> None:
        needed = ("read a document into structured fields", "write a spreadsheet")
        result, audit = self._refuse(tmp_path, needed)
        gaps = audit.capability_gaps()
        assert len(gaps) == 1
        assert tuple(gaps[0]["requires"]) == needed
        assert gaps[0]["run_id"] == result.run_id
        audit.close()

    def test_the_prompt_itself_is_never_stored(self, tmp_path: Path) -> None:
        """It is the user's content. A digest is enough to recognise it again."""
        _, audit = self._refuse(tmp_path, ("write a spreadsheet",))
        gap = audit.capability_gaps()[0]
        assert "invoice" not in str(gap).lower()
        assert len(gap["prompt_digest"]) >= 32
        audit.close()

    def test_the_same_request_asked_twice_is_counted(self, tmp_path: Path) -> None:
        """How often a process has been wanted is the order worth building in."""
        planner = ScriptedGoalPlanner(
            route=RouteDecision(workflow_id="", supported=False, rationale="no")
        )
        runtime, request, audit, _ = build(tmp_path, planner=planner)
        runtime.run(request, commit=False)
        runtime.run(request, commit=False)
        assert {gap["asked"] for gap in audit.capability_gaps()} == {2}
        audit.close()
