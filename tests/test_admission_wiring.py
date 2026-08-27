"""Operator requests reaching admission from inside a real run.

The admission service was fully tested standalone but unreachable: nothing in the run
path could submit a request, so the feature could not actually be exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.admission import AdmissionService
from shakespeare.capabilities.runner import Organization
from shakespeare.components.builtin import build_registry
from shakespeare.contracts import (
    Composition,
    Invocation,
    OperatorAsk,
    OperatorFamily,
    RequestKind,
)
from shakespeare.runtime.verifier import Denial, Verifier

from harness import INVOICES, build, rename_agent, seed_invoices, values_for
from test_admission import StubRenderer, passing_tests


def ask(
    name: str = "text.titlecase",
    *,
    family: OperatorFamily = OperatorFamily.PURE_TRANSFORM,
    kind: RequestKind = RequestKind.VARIANT,
    dependencies: tuple[str, ...] = (),
    side_effects: tuple[str, ...] = (),
) -> OperatorAsk:
    return OperatorAsk(
        kind=kind,
        family=family,
        name=name,
        features=frozenset({"normalize"}),
        dependencies=dependencies,
        declared_side_effects=side_effects,
        rationale="vendor names arrive in shouty caps and need title casing",
    )


def _with_ask(tmp_path: Path, request_ask: OperatorAsk, *, wire: bool = True):
    # Seed first: item ids are content-addressed, so the agent must be built against the
    # bytes the run will actually see.
    source = seed_invoices(tmp_path / "in", INVOICES)
    agent = rename_agent(values_for(source, INVOICES))
    # The survey capability asks for a component it does not have.
    agent.plans["survey"] = [
        Organization(
            invocations=(
                Invocation(invocation_id="scan", operator="fs.scan", inputs=("root",)),
            ),
            intent="walk the tree",
            sufficient=True,
            publishes="FileInventory",
            ask=request_ask,
        )
    ]
    runtime, request, audit, _ = build(tmp_path, agents={"*": agent})
    if wire:
        runtime.admission = AdmissionService(
            registry=runtime.operators,
            audit=audit,
            workspace=tmp_path / "candidates",
            renderer=StubRenderer(),
            test_runner=passing_tests,
        )
    return runtime, request, audit


class TestAutoAdmissionInsideARun:
    def test_a_clean_variant_is_admitted_and_granted_to_its_domain(
        self, tmp_path: Path
    ) -> None:
        runtime, request, audit = _with_ask(tmp_path, ask())
        result = runtime.run(request)
        assert result.outcome == "committed", result.detail
        assert "text.titlecase" in runtime.operators
        assert runtime.grants["survey"] == {"text.titlecase"}
        audit.close()

    def test_the_grant_is_scoped_to_the_requesting_domain(self, tmp_path: Path) -> None:
        """Admitting an operator for one domain must not widen another's surface."""
        runtime, request, audit = _with_ask(tmp_path, ask())
        runtime.run(request)
        assert "compose" not in runtime.grants
        audit.close()

    def test_the_full_provenance_is_queryable(self, tmp_path: Path) -> None:
        from shakespeare.runtime.audit import schema
        from sqlalchemy import select

        runtime, request, audit = _with_ask(tmp_path, ask())
        runtime.run(request)
        with audit.engine.begin() as connection:
            for table in (
                schema.operator_requests,
                schema.admission_reports,
                schema.admission_decisions,
                schema.operator_registrations,
            ):
                assert connection.execute(select(table)).mappings().all(), table.name
        audit.close()


class TestEscalationInsideARun:
    @pytest.mark.parametrize(
        ("kwargs", "why"),
        [
            ({"dependencies": ("requests",)}, "a new dependency is medium risk"),
            ({"side_effects": ("write:/tmp",)}, "a declared side effect is high risk"),
            ({"kind": RequestKind.BEHAVIOUR}, "no runner operation exists for it"),
        ],
        ids=["dependency", "side-effect", "behaviour"],
    )
    def test_an_escalated_request_is_not_granted(
        self, tmp_path: Path, kwargs: dict, why: str
    ) -> None:
        runtime, request, audit = _with_ask(tmp_path, ask(**kwargs))
        result = runtime.run(request)
        assert result.outcome == "committed", "escalation must not derail the run"
        assert runtime.grants == {}, why
        assert "text.titlecase" not in runtime.operators
        audit.close()

    def test_an_escalated_request_waits_for_a_person(self, tmp_path: Path) -> None:
        runtime, request, audit = _with_ask(tmp_path, ask(dependencies=("requests",)))
        runtime.run(request)
        pending = audit.pending_admissions()
        assert [item["name"] for item in pending] == ["text.titlecase"]
        assert pending[0]["computed_risk"] == "medium"
        audit.close()


class TestWithoutAnAdmissionService:
    def test_the_request_is_recorded_but_never_granted(self, tmp_path: Path) -> None:
        """The right default for an unattended run: note the gap, grant nothing."""
        runtime, request, audit = _with_ask(tmp_path, ask(), wire=False)
        result = runtime.run(request)
        assert result.outcome == "committed", result.detail
        assert runtime.grants == {}
        assert "text.titlecase" not in runtime.operators

        from shakespeare.runtime.audit import schema
        from sqlalchemy import select

        with audit.engine.begin() as connection:
            recorded = connection.execute(select(schema.operator_requests)).mappings().all()
        assert [row["name"] for row in recorded] == ["text.titlecase"]
        audit.close()


class TestGrantBounds:
    def test_a_grant_lets_the_verifier_accept_an_admitted_operator(self) -> None:
        from shakespeare.capabilities import CapabilityRegistry
        from shakespeare.capabilities.runner import _as_domain

        verifier = Verifier(build_registry())
        domain = _as_domain(CapabilityRegistry().get("survey"))
        composition = Composition(
            domain_id="survey",
            invocations=(Invocation(invocation_id="t", operator="text.normalize"),),
        )
        with pytest.raises(Denial, match="outside the catalog"):
            verifier.verify_composition(composition, domain)
        verifier.verify_composition(composition, domain, granted=frozenset({"text.normalize"}))

    def test_a_grant_cannot_unlock_a_mutation_operator(self) -> None:
        """Even a granted operator is refused if it writes: agents plan, the runtime commits."""
        from shakespeare.capabilities import CapabilityRegistry
        from shakespeare.capabilities.runner import _as_domain

        verifier = Verifier(build_registry())
        domain = _as_domain(CapabilityRegistry().get("survey"))
        composition = Composition(
            domain_id="survey",
            invocations=(Invocation(invocation_id="c", operator="fs.commit"),),
        )
        with pytest.raises(Denial, match="reserved to the runtime"):
            verifier.verify_composition(composition, domain, granted=frozenset({"fs.commit"}))

    def test_an_ask_cannot_assert_authority(self) -> None:
        """A request describes a need; it cannot claim a risk, a version or an entrypoint."""
        forbidden = {"risk", "version", "entrypoint", "admission_id", "package_digest"}
        assert not forbidden & set(OperatorAsk.model_fields)
