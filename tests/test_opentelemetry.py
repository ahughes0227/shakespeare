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
