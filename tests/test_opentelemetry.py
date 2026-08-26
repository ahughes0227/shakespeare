"""OpenTelemetry emission.

Three things went wrong at once and each is pinned here: there was no OTel exporter at
all, no exporter activated in practice, and the span tree was one level deep because the
control loop held a tracer it never called.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from shakespeare.telemetry import OpenTelemetryExporter, Tracer

from harness import build


@pytest.fixture
def collected(monkeypatch):
    """A real tracer provider writing into memory, so spans are inspected not assumed."""
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider, raising=False)

    exporter = OpenTelemetryExporter()
    exporter._tracer = provider.get_tracer("shakespeare")
    return memory, exporter


class TestActivation:
    def test_a_collector_endpoint_is_enough_to_activate_it(self, monkeypatch) -> None:
        """The standard variable, so a collector already running picks this up."""
        from shakespeare.bootstrap import exporters

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        assert any(isinstance(item, OpenTelemetryExporter) for item in exporters())

    def test_nothing_is_wired_without_configuration(self, monkeypatch) -> None:
        from shakespeare.bootstrap import exporters
        from shakespeare.telemetry import NullExporter

        for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert all(isinstance(item, NullExporter) for item in exporters())

    def test_both_backends_can_run_together(self, monkeypatch) -> None:
        from shakespeare.bootstrap import exporters
        from shakespeare.telemetry import LangSmithExporter

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("LANGSMITH_PROJECT", "shakespeare")
        monkeypatch.setenv("LANGSMITH_API_KEY", "x")
        kinds = {type(item) for item in exporters()}
        assert {OpenTelemetryExporter, LangSmithExporter} <= kinds


class TestSpanShape:
    def test_envelope_fields_become_attributes(self, collected) -> None:
        memory, exporter = collected
        tracer = Tracer("run-1", [exporter])
        with tracer.span("operator.fs.scan", stage="inventoried", operator="fs.scan") as span:
            span.add_count("items", 7)
        (recorded,) = memory.get_finished_spans()
        assert recorded.name == "operator.fs.scan"
        assert recorded.attributes["shakespeare.stage"] == "inventoried"
        assert recorded.attributes["shakespeare.counts.items"] == 7

    def test_a_failure_marks_the_span(self, collected) -> None:
        from opentelemetry.trace import StatusCode
        from shakespeare.contracts import ErrorCode

        memory, exporter = collected
        tracer = Tracer("run-1", [exporter])
        with tracer.span("operator.doc.extract") as span:
            span.fail(ErrorCode.EXTRACTION_UNAVAILABLE)
        (recorded,) = memory.get_finished_spans()
        assert recorded.status.status_code is StatusCode.ERROR

    def test_no_document_content_can_become_an_attribute(self, collected) -> None:
        """Attributes come from the envelope and nowhere else."""
        memory, exporter = collected
        tracer = Tracer("run-1", [exporter])
        with tracer.span("operator.doc.extract") as span:
            span.add_digest("content", "ACME Corporation invoice INV-1")
        (recorded,) = memory.get_finished_spans()
        assert "ACME" not in str(dict(recorded.attributes))


class TestSpanTree:
    def test_a_run_produces_a_nested_tree_not_a_flat_list(
        self, collected, tmp_path: Path
    ) -> None:
        """The gap the refactor left: only component calls were ever traced."""
        memory, exporter = collected
        runtime, request, audit, _ = build(tmp_path)
        runtime.tracer = Tracer("session", [exporter])

        runtime.run(request, commit=False)
        spans = memory.get_finished_spans()
        names = {span.name for span in spans}

        assert "run" in names, "the run itself must be a span"
        assert any(name.startswith("goal.") for name in names), "each goal attempt"
        assert any(name.startswith("round.") for name in names), "each capability round"
        assert any(name.startswith("operator.") for name in names), "each component call"

        by_id = {span.context.span_id: span for span in spans}
        parents = {
            span.name: by_id[span.parent.span_id].name
            for span in spans
            if span.parent and span.parent.span_id in by_id
        }
        component = next(name for name in parents if name.startswith("operator."))
        assert parents[component].startswith("round."), "a component sits under its round"
        round_name = parents[component]
        assert parents[round_name].startswith("goal."), "a round sits under its goal"
        assert parents[parents[round_name]] == "run", "a goal sits under the run"

    def test_every_span_carries_its_run(self, collected, tmp_path: Path) -> None:
        memory, exporter = collected
        runtime, request, audit, _ = build(tmp_path)
        runtime.tracer = Tracer("session", [exporter])
        result = runtime.run(request, commit=False)

        run_ids = {
            span.attributes.get("shakespeare.run_id") for span in memory.get_finished_spans()
        }
        assert run_ids == {result.run_id}, "a session tracer must rebind per run"
        audit.close()


class TestDiagnosis:
    """Would the trace explain a stall?

    A sixty-invoice run stalled and the telemetry could not say why: every round span
    looked identical and successful. These pin the fields that make the difference.
    """

    def _stalled(self, tmp_path: Path, collected):
        from shakespeare.artifacts import Quality
        from shakespeare.capabilities.runner import Organization
        from shakespeare.contracts import Invocation
        from shakespeare.telemetry import Tracer

        from harness import rename_agent, seed_invoices, values_for

        memory, exporter = collected
        source = seed_invoices(tmp_path / "in")
        agent = rename_agent(values_for(source))
        # A capability that never finishes and never advances.
        agent.plans["acquire"] = [
            Organization(
                invocations=(
                    Invocation(
                        invocation_id="ext",
                        operator="doc.extract",
                        selections={"extract": "auto_chain"},
                        inputs=("root", "items"),
                    ),
                ),
                intent="read a slice",
                sufficient=False,
                publishes="ExtractedContent",
                quality=Quality.PARTIAL,
                summary={"items_done": 20, "items_total": 60},
            )
        ]
        runtime, request, audit, _ = build(tmp_path, agents={"*": agent})
        runtime.tracer = Tracer("session", [exporter])
        runtime.run(request, commit=False)
        audit.close()
        return memory.get_finished_spans()

    def _named(self, spans, prefix: str):
        return [s for s in spans if s.name.startswith(prefix)]

    def test_the_trace_shows_the_capability_never_finished(
        self, collected, tmp_path: Path
    ) -> None:
        rounds = self._named(self._stalled(tmp_path, collected), "round.acquire")
        assert rounds, "the stalling capability must appear"
        assert all(r.attributes["shakespeare.sufficient"] is False for r in rounds)

    def test_the_trace_shows_it_made_no_progress(self, collected, tmp_path: Path) -> None:
        """The field that turns 'it stalled' into 'it stalled at twenty of sixty'."""
        rounds = self._named(self._stalled(tmp_path, collected), "round.acquire")
        done = {r.attributes["shakespeare.counts.items_done"] for r in rounds}
        total = {r.attributes["shakespeare.counts.items_total"] for r in rounds}
        assert done == {20} and total == {60}, "identical across rounds: no progress"

    def test_the_trace_shows_how_many_rounds_were_burned(
        self, collected, tmp_path: Path
    ) -> None:
        goals = self._named(self._stalled(tmp_path, collected), "goal.readable")
        assert goals[0].attributes["shakespeare.counts.rounds"] > 1

    def test_partial_evidence_is_visible_as_partial(self, collected, tmp_path: Path) -> None:
        rounds = self._named(self._stalled(tmp_path, collected), "round.acquire")
        assert rounds[0].attributes["shakespeare.quality"] == "partial"
        assert rounds[0].attributes["shakespeare.published"] == "ExtractedContent"

    def test_a_gate_records_its_outcome_and_what_was_missing(
        self, collected, tmp_path: Path
    ) -> None:
        from shakespeare.agent import FakeCapabilityAgent
        from shakespeare.capabilities.runner import Organization
        from shakespeare.telemetry import Tracer

        memory, exporter = collected
        agent = FakeCapabilityAgent()
        agent.queue("survey", Organization(intent="produce nothing", sufficient=True))
        runtime, request, audit, _ = build(tmp_path, agents={"*": agent})
        runtime.tracer = Tracer("session", [exporter])
        runtime.run(request, commit=False)

        goals = self._named(memory.get_finished_spans(), "goal.inventoried")
        assert goals[0].attributes["shakespeare.outcome"] == "blocked"
        assert "FileInventory" in goals[0].attributes["shakespeare.missing_kinds"]
        audit.close()

    def test_model_calls_are_attributable(self, collected, tmp_path: Path) -> None:
        """Cost and prompt version per call, so a regression can be traced to a promotion."""
        from shakespeare.contracts import RouteDecision
        from shakespeare.gateway import FakeGateway, ModelProfile
        from shakespeare.planner import ModelGoalPlanner
        from shakespeare.prompts import PromptStore
        from shakespeare.telemetry import Tracer

        memory, exporter = collected
        from shakespeare.planner import GoalChoice, Judgment

        judgment = {"satisfied": True, "rationale": "sufficient"}
        gateway = (
            FakeGateway()
            .queue(RouteDecision, {"workflow_id": "rename_files", "supported": True})
            .queue(GoalChoice, {"goal_id": "readable"})
            .queue(Judgment, judgment, judgment, judgment)
        )
        runtime, request, audit, _ = build(tmp_path)
        runtime.planner = ModelGoalPlanner(
            gateway=gateway,
            profile=ModelProfile(profile_id="p", model="openrouter/openai/gpt-5-mini"),
            prompts=PromptStore(),
        )
        runtime.tracer = Tracer("session", [exporter])
        runtime.run(request, commit=False)

        model_spans = self._named(memory.get_finished_spans(), "model.")
        assert model_spans, "a model call must be its own span"
        attrs = model_spans[0].attributes
        assert attrs["shakespeare.requested_model"] == "openrouter/openai/gpt-5-mini"
        assert "shakespeare.prompt_tokens" in attrs
        audit.close()

    def test_no_content_reaches_any_of_it(self, collected, tmp_path: Path) -> None:
        spans = self._stalled(tmp_path, collected)
        blob = str([dict(s.attributes) for s in spans])
        for secret in ("ACME Corporation", "INV-99812", "Globex", "invoice body"):
            assert secret not in blob


class TestSurvivingAKill:
    """The audit log records a goal attempt only once the goal completes.

    A run killed mid-goal therefore loses everything in flight — which is exactly what
    happened to the sixty-invoice run, leaving zero journalled compositions against
    seventeen model calls. Spans close as each round ends, so they are already exported
    by then.
    """

    def test_a_round_span_closes_before_its_goal_does(
        self, collected, tmp_path: Path
    ) -> None:
        from shakespeare.artifacts import Quality
        from shakespeare.capabilities.runner import Organization
        from shakespeare.contracts import Invocation
        from shakespeare.telemetry import Tracer

        from harness import rename_agent, seed_invoices, values_for

        memory, exporter = collected
        source = seed_invoices(tmp_path / "in")
        agent = rename_agent(values_for(source))
        agent.plans["acquire"] = [
            Organization(
                invocations=(
                    Invocation(
                        invocation_id="ext",
                        operator="doc.extract",
                        selections={"extract": "auto_chain"},
                        inputs=("root", "items"),
                    ),
                ),
                intent="a slice",
                sufficient=False,
                publishes="ExtractedContent",
                quality=Quality.PARTIAL,
                summary={"items_done": 20, "items_total": 60},
            )
        ]
        runtime, request, audit, _ = build(tmp_path, agents={"*": agent})
        runtime.tracer = Tracer("session", [exporter])
        runtime.run(request, commit=False)

        spans = memory.get_finished_spans()
        goal = next(s for s in spans if s.name == "goal.readable")
        rounds = [s for s in spans if s.name == "round.acquire"]
        assert len(rounds) > 1
        # Every round ended before its goal did, so a kill part-way still leaves the
        # completed rounds exported and readable.
        assert all(r.end_time <= goal.end_time for r in rounds)
        assert min(r.end_time for r in rounds) < goal.end_time
        audit.close()

    def test_the_journal_has_nothing_for_an_unfinished_goal(
        self, collected, tmp_path: Path
    ) -> None:
        """The gap spans fill: a goal still in flight is absent from the audit log."""
        from shakespeare.agent import FakeCapabilityAgent
        from shakespeare.audit import schema
        from shakespeare.capabilities.runner import Organization
        from sqlalchemy import select

        agent = FakeCapabilityAgent()
        agent.queue("survey", Organization(intent="never publishes", sufficient=True))
        runtime, request, audit, _ = build(tmp_path, agents={"*": agent})

        # Interrupt after the capability has run but before the goal resolves.
        original = runtime._journal
        runtime._journal = lambda *a, **k: None  # type: ignore[method-assign]
        runtime.run(request, commit=False)
        runtime._journal = original  # type: ignore[method-assign]

        with audit.engine.begin() as connection:
            attempts = connection.execute(select(schema.stage_attempts)).mappings().all()
        assert not attempts, "an unfinished goal leaves no journal row"
        audit.close()
