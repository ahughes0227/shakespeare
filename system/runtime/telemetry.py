"""Span emission with an architectural redaction boundary.

The inputs to this program are customer documents.  Extracted text, field values and
content-derived filenames must never reach a hosted backend, so redaction is not a
configuration setting that can be forgotten — it is the shape of the API.

`Tracer.span()` accepts only the primitives of a `TelemetryEnvelope`: ids, digests,
versions, counts, timings, costs and closed error codes.  There is no parameter through
which a caller could pass document content, so there is nothing for a masking hook to
miss.  LangSmith's own `hide_inputs`/`hide_outputs` masking is enabled as defence in
depth, not as the control.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

from ..contracts import ErrorCode, TelemetryEnvelope, content_digest


def digest_of(value: Any) -> str:
    """Convert content to a digest at the call site, so content never travels further."""
    return content_digest(value)


class Exporter(Protocol):
    def export(self, envelope: TelemetryEnvelope) -> None: ...


class NullExporter:
    """Default.  Nothing leaves the machine unless an exporter is deliberately wired."""

    def export(self, envelope: TelemetryEnvelope) -> None:
        return None


class RecordingExporter:
    """Captures everything that would ship, for the redaction test."""

    def __init__(self) -> None:
        self.envelopes: list[TelemetryEnvelope] = []

    def export(self, envelope: TelemetryEnvelope) -> None:
        self.envelopes.append(envelope)

    def serialized(self) -> str:
        """Every byte that would be transmitted, for content assertions."""
        return "\n".join(item.model_dump_json() for item in self.envelopes)


class OpenTelemetryExporter:
    """Emit envelopes as OTLP spans.

    Nesting comes from OpenTelemetry's own context propagation rather than from anything
    in the envelope: `Tracer.span` opens a real span, so a component call is a child of
    its round, which is a child of its capability, and so on up to the run.

    The redaction guarantee is unchanged. Span attributes are set from
    `TelemetryEnvelope` fields and nothing else, so there is no path by which document
    content could become an attribute.
    """

    def __init__(self, service_name: str = "shakespeare") -> None:
        self.service_name = service_name
        self._tracer: Any = None

    def tracer(self) -> Any:
        if self._tracer is None:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = trace.get_tracer_provider()
            if not isinstance(provider, TracerProvider):
                provider = TracerProvider(
                    resource=Resource.create({"service.name": self.service_name})
                )
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
                trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("shakespeare")
        return self._tracer

    def export(self, envelope: TelemetryEnvelope) -> None:
        """Only reached for an envelope emitted outside a live span."""
        span = self.tracer().start_span(envelope.span)
        _apply(span, envelope)
        span.end()

    def shutdown(self) -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.force_flush()


def _apply(span: Any, envelope: TelemetryEnvelope) -> None:
    """Set span attributes from the envelope, and only from the envelope."""
    for key, value in envelope.model_dump(mode="json", exclude_none=True).items():
        if key == "span":
            continue
        if isinstance(value, dict):
            for inner, item in value.items():
                span.set_attribute(f"shakespeare.{key}.{inner}", item)
        else:
            span.set_attribute(f"shakespeare.{key}", value)
    if envelope.error_code is not None:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, str(envelope.error_code)))


class LangSmithExporter:
    """Ships envelopes to LangSmith as span metadata.

    Only `TelemetryEnvelope` fields are sent.  The LangSmith client is additionally
    configured with masking hooks; if that configuration is ever lost, the envelope shape
    still guarantees no content can be transmitted.
    """

    def __init__(self, project: str, client: Any | None = None) -> None:
        self.project = project
        self._client = client

    def _lazy_client(self) -> Any:
        if self._client is None:
            from langsmith import Client

            # Defence in depth.  Confirm hook names against the installed SDK version.
            self._client = Client(hide_inputs=True, hide_outputs=True)
        return self._client

    def export(self, envelope: TelemetryEnvelope) -> None:
        client = self._lazy_client()
        client.create_run(
            name=envelope.span,
            run_type="chain",
            project_name=self.project,
            extra={"metadata": envelope.model_dump(mode="json", exclude_none=True)},
            inputs={},
            outputs={},
        )


class Tracer:
    """Emits envelopes for one run.

    Note the absence of a free-form attribute parameter.  That absence is the redaction
    control; do not add one.
    """

    def __init__(self, run_id: str, exporters: Sequence[Exporter] = ()) -> None:
        self.run_id = run_id
        self._exporters: tuple[Exporter, ...] = tuple(exporters) or (NullExporter(),)
        self._otel = next(
            (item for item in self._exporters if isinstance(item, OpenTelemetryExporter)),
            None,
        )

    def rebind(self, run_id: str) -> Tracer:
        """A tracer for one run, sharing the configured exporters."""
        return Tracer(run_id, self._exporters)

    def emit(self, envelope: TelemetryEnvelope) -> None:
        for exporter in self._exporters:
            exporter.export(envelope)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        stage: str | None = None,
        attempt: int | None = None,
        domain: str | None = None,
        operator: str | None = None,
        operator_version: str | None = None,
        prompt_version: str | None = None,
        requested_model: str | None = None,
        resolved_model: str | None = None,
        provider: str | None = None,
        digests: dict[str, str] | None = None,
        counts: dict[str, int] | None = None,
        cost_usd: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> Iterator[SpanState]:
        started = time.monotonic()
        state = SpanState()
        # A real OTel span when one is configured, so parent/child nesting comes from
        # OpenTelemetry's own context rather than being reconstructed afterwards: a
        # component call is a child of its round, which is a child of its capability.
        live = self._otel.tracer().start_as_current_span(name) if self._otel else None
        span = live.__enter__() if live is not None else None
        try:
            yield state
        finally:
            envelope = TelemetryEnvelope(
                run_id=self.run_id,
                span=name,
                stage=stage,
                attempt=attempt,
                domain=domain,
                operator=operator,
                operator_version=operator_version,
                prompt_version=prompt_version,
                requested_model=requested_model,
                resolved_model=resolved_model,
                provider=provider,
                sufficient=state.sufficient,
                published=state.published,
                quality=state.quality,
                outcome=state.outcome,
                digests={**(digests or {}), **state.digests},
                counts={**(counts or {}), **state.counts},
                failed_checks=state.failed_checks,
                missing_kinds=state.missing_kinds,
                duration_ms=(time.monotonic() - started) * 1000,
                cost_usd=cost_usd if cost_usd is not None else state.cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error_code=state.error_code,
            )
            if span is not None:
                _apply(span, envelope)
            if live is not None:
                live.__exit__(None, None, None)
            for exporter in self._exporters:
                # The OTel exporter has already recorded this as a live span; emitting it
                # again would double-count it.
                if exporter is self._otel:
                    continue
                exporter.export(envelope)


class SpanState:
    """Mutable handle for a span in flight.  Digests and counts only, by design."""

    def __init__(self) -> None:
        self.digests: dict[str, str] = {}
        self.counts: dict[str, int] = {}
        self.cost_usd: float | None = None
        self.error_code: ErrorCode | None = None
        self.sufficient: bool | None = None
        self.published: str | None = None
        self.quality: str | None = None
        self.outcome: str | None = None
        self.failed_checks: tuple[str, ...] = ()
        self.missing_kinds: tuple[str, ...] = ()

    def record(
        self,
        *,
        sufficient: bool | None = None,
        published: str | None = None,
        quality: str | None = None,
        outcome: str | None = None,
        failed_checks: tuple[str, ...] = (),
        missing_kinds: tuple[str, ...] = (),
    ) -> None:
        if sufficient is not None:
            self.sufficient = sufficient
        self.published = published or self.published
        self.quality = quality or self.quality
        self.outcome = outcome or self.outcome
        self.failed_checks = failed_checks or self.failed_checks
        self.missing_kinds = missing_kinds or self.missing_kinds

    def add_counts(self, values: dict[str, Any]) -> None:
        """Numeric values only.

        A capability's summary is model-supplied and typed as Any, so a string could
        arrive in it. Taking only numbers keeps the no-content guarantee true by
        construction rather than by trusting the prompt.
        """
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.counts[key] = int(value)

    def add_digest(self, key: str, value: Any) -> None:
        self.digests[key] = digest_of(value)

    def add_count(self, key: str, value: int) -> None:
        self.counts[key] = self.counts.get(key, 0) + value

    def fail(self, code: ErrorCode) -> None:
        self.error_code = code
