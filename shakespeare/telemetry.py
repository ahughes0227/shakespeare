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

from .contracts import ErrorCode, TelemetryEnvelope, content_digest


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
    ) -> Iterator[SpanState]:
        started = time.monotonic()
        state = SpanState()
        try:
            yield state
        finally:
            self.emit(
                TelemetryEnvelope(
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
                    digests={**(digests or {}), **state.digests},
                    counts={**(counts or {}), **state.counts},
                    duration_ms=(time.monotonic() - started) * 1000,
                    cost_usd=cost_usd if cost_usd is not None else state.cost_usd,
                    error_code=state.error_code,
                )
            )


class SpanState:
    """Mutable handle for a span in flight.  Digests and counts only, by design."""

    def __init__(self) -> None:
        self.digests: dict[str, str] = {}
        self.counts: dict[str, int] = {}
        self.cost_usd: float | None = None
        self.error_code: ErrorCode | None = None

    def add_digest(self, key: str, value: Any) -> None:
        self.digests[key] = digest_of(value)

    def add_count(self, key: str, value: int) -> None:
        self.counts[key] = self.counts.get(key, 0) + value

    def fail(self, code: ErrorCode) -> None:
        self.error_code = code
