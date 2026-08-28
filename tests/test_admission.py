"""Operator admission: computed risk, a narrow auto-admit path, and human escalation."""

from __future__ import annotations

from pathlib import Path

import pytest
from system.components.admission import (
    TEST_TIERS,
    AdmissionError,
    AdmissionService,
    _requested_operation,
    compute_risk,
    digest_tree,
)
from system.components.catalog import build_registry
from system.components.registry import FAMILY_RUNNERS
from system.contracts import (
    AdmissionChoice,
    AdmissionDisposition,
    DecidedBy,
    OperatorCandidate,
    OperatorFamily,
    OperatorRequest,
    OperatorSpec,
    RequestKind,
    RiskLevel,
)
from system.runtime.audit import AuditStore


class StubRenderer:
    """Renders a deterministic package without shelling out to Copier."""

    def __init__(self, *, reproducible: bool = True) -> None:
        self.reproducible = reproducible
        self._calls = 0

    def render(self, request: OperatorRequest, destination: Path) -> str:
        # Same validation the real renderer performs; a more permissive stub would hide
        # the refusal these tests exist to check.
        _requested_operation(request)
        self._calls += 1
        destination.mkdir(parents=True, exist_ok=True)
        salt = "" if self.reproducible else str(self._calls)
        (destination / "operator.json").write_text(f'{{"name": "{request.name}"{salt}}}')
        return digest_tree(destination)


def passing_tests(_: Path) -> dict[str, bool]:
    return dict.fromkeys(TEST_TIERS, True)


def failing_tests(_: Path) -> dict[str, bool]:
    return {**dict.fromkeys(TEST_TIERS, True), "containment": False}


def request_for(
    family: OperatorFamily = OperatorFamily.PURE_TRANSFORM,
    *,
    kind: RequestKind = RequestKind.VARIANT,
    features: frozenset[str] = frozenset({"normalize"}),
    dependencies: tuple[str, ...] = (),
    side_effects: tuple[str, ...] = (),
    name: str = "text.tidy",
) -> OperatorRequest:
    return OperatorRequest(
        request_id="req-1",
        run_id="run-1",
        domain_id="field_resolution",
        kind=kind,
        family=family,
        name=name,
        features=features,
        dependencies=dependencies,
        declared_side_effects=side_effects,
        rationale="normalise vendor names before rendering",
    )


@pytest.fixture
def service(tmp_path: Path):
    audit = AuditStore(tmp_path / "audit.sqlite3")
    yield AdmissionService(
        registry=build_registry(),
        audit=audit,
        workspace=tmp_path / "candidates",
        renderer=StubRenderer(),
        test_runner=passing_tests,
    )
    audit.close()


class TestComputedRisk:
    @pytest.mark.parametrize(
        ("side_effects", "dependencies", "expected"),
        [
            ((), (), RiskLevel.LOW),
            ((), ("numpy",), RiskLevel.MEDIUM),
            (("write:output_root",), (), RiskLevel.HIGH),
            (("write:output_root",), ("numpy",), RiskLevel.HIGH),
        ],
    )
    def test_risk_follows_from_what_it_touches(
        self, side_effects: tuple[str, ...], dependencies: tuple[str, ...], expected: RiskLevel
    ) -> None:
        candidate = OperatorCandidate(
            candidate_id="c",
            request_id="r",
            spec=OperatorSpec(
                name="x",
                version="1.0.0",
                description="d",
                family=OperatorFamily.PURE_TRANSFORM,
                entrypoint=FAMILY_RUNNERS[OperatorFamily.PURE_TRANSFORM],
            ),
            package_digest="d",
            dependencies=dependencies,
            declared_side_effects=side_effects,
        )
        assert compute_risk(candidate) is expected


class TestAutoAdmission:
    def test_clean_pure_transform_variant_auto_admits(self, service: AdmissionService) -> None:
        report, _ = service.evaluate(request_for())
        assert report.disposition is AdmissionDisposition.AUTO_ADMIT
        assert report.computed_risk is RiskLevel.LOW
        assert report.findings == ()

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"kind": RequestKind.BEHAVIOUR}, "needs an operation that does not exist"),
            ({"dependencies": ("requests",)}, "brings an untrusted dependency"),
            ({"side_effects": ("write:/tmp",)}, "declares a side effect"),
            ({"name": "name.render"}, "collides with a registered operator"),
        ],
        ids=lambda value: str(value)[:40],
    )
    def test_escalates_to_human_review(
        self, service: AdmissionService, kwargs: dict[str, object], reason: str
    ) -> None:
        report, _ = service.evaluate(request_for(**kwargs))  # type: ignore[arg-type]
        assert report.disposition is AdmissionDisposition.HUMAN_REVIEW, reason
        assert report.findings

    def test_failing_test_tier_escalates(self, service: AdmissionService) -> None:
        service.test_runner = failing_tests
        report, _ = service.evaluate(request_for())
        assert report.disposition is AdmissionDisposition.HUMAN_REVIEW
        assert any(f.code == "tests_failed" for f in report.findings)

    def test_irreproducible_render_escalates(self, service: AdmissionService) -> None:
        """The audited digest is only meaningful if rendering is a pure function."""
        service.renderer = StubRenderer(reproducible=False)
        report, _ = service.evaluate(request_for())
        assert not report.reproducible
        assert any(f.code == "not_reproducible" for f in report.findings)

    def test_mutation_family_can_never_auto_admit(self, service: AdmissionService) -> None:
        report, _ = service.evaluate(
            request_for(
                OperatorFamily.FILESYSTEM_MUTATION,
                features=frozenset({"stage_write"}),
                name="fs.sneak",
            )
        )
        assert report.disposition is AdmissionDisposition.HUMAN_REVIEW
        assert any(f.code == "mutation_requires_human" for f in report.findings)

    def test_a_request_naming_no_vetted_operation_is_refused(
        self, service: AdmissionService
    ) -> None:
        report, _ = service.evaluate(request_for(features=frozenset({"exec_shell"})))
        assert report.disposition is AdmissionDisposition.HUMAN_REVIEW
        assert any(f.code == "render_failed" for f in report.findings)


class TestDecisions:
    def test_planner_may_approve_an_auto_admit_candidate(
        self, service: AdmissionService
    ) -> None:
        report, candidate = service.evaluate(request_for())
        service.decide(
            report, candidate, decided_by=DecidedBy.PLANNER, choice=AdmissionChoice.APPROVE
        )
        assert "text.tidy" in service.registry

    def test_planner_may_not_approve_an_escalated_candidate(
        self, service: AdmissionService
    ) -> None:
        report, candidate = service.evaluate(request_for(dependencies=("requests",)))
        with pytest.raises(AdmissionError, match="needs human review"):
            service.decide(
                report, candidate, decided_by=DecidedBy.PLANNER, choice=AdmissionChoice.APPROVE
            )
        assert "text.tidy" not in service.registry

    def test_human_may_approve_an_escalated_candidate(self, service: AdmissionService) -> None:
        report, candidate = service.evaluate(request_for(dependencies=("requests",)))
        service.decide(
            report, candidate, decided_by=DecidedBy.HUMAN, choice=AdmissionChoice.APPROVE
        )
        assert "text.tidy" in service.registry

    def test_a_denied_candidate_never_enters_the_registry(
        self, service: AdmissionService
    ) -> None:
        report, candidate = service.evaluate(request_for())
        service.decide(
            report, candidate, decided_by=DecidedBy.HUMAN, choice=AdmissionChoice.DENY
        )
        assert "text.tidy" not in service.registry

    def test_an_admitted_operator_still_uses_its_family_runner(
        self, service: AdmissionService
    ) -> None:
        """Admission cannot introduce a new call target."""
        report, candidate = service.evaluate(request_for())
        service.decide(
            report, candidate, decided_by=DecidedBy.PLANNER, choice=AdmissionChoice.APPROVE
        )
        spec = service.registry.get("text.tidy").spec
        assert spec.entrypoint == FAMILY_RUNNERS[OperatorFamily.PURE_TRANSFORM]
        assert spec.admission_id == report.report_id


class TestProvenance:
    def test_request_report_and_decision_are_all_recorded(
        self, service: AdmissionService
    ) -> None:
        from sqlalchemy import select
        from system.runtime.audit import schema

        report, candidate = service.evaluate(request_for())
        service.decide(
            report, candidate, decided_by=DecidedBy.PLANNER, choice=AdmissionChoice.APPROVE
        )
        with service.audit.engine.begin() as connection:
            for table in (
                schema.operator_requests,
                schema.admission_reports,
                schema.admission_decisions,
                schema.operator_registrations,
            ):
                assert connection.execute(select(table)).mappings().all(), table.name
