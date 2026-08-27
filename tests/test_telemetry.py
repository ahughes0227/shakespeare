"""The highest-consequence test in the suite.

The inputs are customer invoices.  If document content can reach an exporter, it reaches
a hosted backend.  These tests assert the boundary holds by construction rather than by
configuration.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError
from shakespeare.contracts import ErrorCode, TelemetryEnvelope
from shakespeare.runtime.telemetry import RecordingExporter, Tracer

#: Strings standing in for real customer data in the fixture set.
SENSITIVE = (
    "ACME Corporation",
    "INV-99812",
    "PO-44117",
    "Remittance to account 12345678",
    "202401, ACME Corporation, INV-99812, PO-44117.pdf",
)


@pytest.fixture
def recorder() -> RecordingExporter:
    return RecordingExporter()


class TestNoContentEverShips:
    def test_digested_content_does_not_appear_in_the_payload(
        self, recorder: RecordingExporter
    ) -> None:
        tracer = Tracer("run-1", [recorder])
        with tracer.span("stage.extract", stage="extract", attempt=1) as span:
            for index, secret in enumerate(SENSITIVE):
                span.add_digest(f"item-{index}", secret)
            span.add_count("items", len(SENSITIVE))

        shipped = recorder.serialized()
        for secret in SENSITIVE:
            assert secret not in shipped, f"content leaked into telemetry: {secret!r}"
        assert '"items":5' in shipped

    def test_error_codes_are_closed_enum_values(self, recorder: RecordingExporter) -> None:
        tracer = Tracer("run-1", [recorder])
        with tracer.span("op.extract") as span:
            span.fail(ErrorCode.EXTRACTION_UNAVAILABLE)
        assert recorder.envelopes[0].error_code is ErrorCode.EXTRACTION_UNAVAILABLE

    def test_span_has_no_free_form_attribute_parameter(self) -> None:
        """The absence of a free-form parameter is the redaction control.

        If someone adds `**attributes` or a `payload` argument to `Tracer.span`, content
        becomes shippable again and every other guarantee here is void.
        """
        signature = inspect.signature(Tracer.span)
        for parameter in signature.parameters.values():
            assert parameter.kind is not inspect.Parameter.VAR_KEYWORD, (
                "Tracer.span must not accept **kwargs: it would reopen the redaction hole"
            )
        allowed = set(TelemetryEnvelope.model_fields) | {"self", "name", "digests", "counts"}
        for parameter in signature.parameters:
            assert parameter in allowed, f"unexpected span parameter: {parameter}"

    def test_envelope_rejects_raw_text_in_digests(self) -> None:
        with pytest.raises(ValidationError):
            TelemetryEnvelope(run_id="r", span="s", digests={"item": "ACME Corporation"})


class TestSpanShape:
    def test_records_duration_and_dimensions(self, recorder: RecordingExporter) -> None:
        tracer = Tracer("run-1", [recorder])
        with tracer.span(
            "model.call",
            domain="field_resolution",
            prompt_version="1.2.0",
            requested_model="openrouter/openai/gpt-5-mini",
            resolved_model="openai/gpt-5-mini",
            provider="openrouter",
            cost_usd=0.004,
        ):
            pass
        envelope = recorder.envelopes[0]
        assert envelope.duration_ms is not None and envelope.duration_ms >= 0
        # Requested vs resolved is what makes silent provider drift attributable.
        assert envelope.requested_model != envelope.resolved_model
        assert envelope.prompt_version == "1.2.0"

    def test_default_tracer_exports_nothing(self) -> None:
        tracer = Tracer("run-1")
        with tracer.span("noop"):
            pass  # No exporter configured: nothing may leave the machine.
