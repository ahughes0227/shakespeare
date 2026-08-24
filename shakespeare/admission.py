"""Operator requests and their deterministic admission.

A subagent that lacks an operator may request one.  This is safe only because of the
family design: a rendered package carries no callable, so a request can select vetted
behaviour and configure it but can never author new behaviour.

Risk is computed here, never declared by the requester, and the auto-admit path is
deliberately narrow.  Anything touching the filesystem is high risk by construction and
can only ever be approved by a person.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .audit import AuditStore
from .contracts import (
    AUTO_ADMISSIBLE_FAMILIES,
    AdmissionChoice,
    AdmissionDecision,
    AdmissionDisposition,
    AdmissionFinding,
    AdmissionReport,
    DecidedBy,
    FindingSeverity,
    OperatorCandidate,
    OperatorFamily,
    OperatorRequest,
    OperatorSpec,
    RequestKind,
    RiskLevel,
    content_digest,
)
from .registry import FAMILY_RUNNERS, OperatorRegistry
from .runners import allowlist

#: The four tiers a rendered package must pass before it can be admitted.
TEST_TIERS: tuple[str, ...] = ("contract", "containment", "functional", "regression")


class AdmissionError(RuntimeError):
    pass


def compute_risk(candidate: OperatorCandidate) -> RiskLevel:
    """Risk follows from what the operator touches, not from what its author claims."""
    if candidate.declared_side_effects:
        return RiskLevel.HIGH
    if candidate.dependencies:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


class PackageRenderer(Protocol):
    def render(self, request: OperatorRequest, destination: Path) -> str:
        """Render the package and return its content digest."""


@dataclass
class CopierRenderer:
    template_root: Path = Path("_operator_templates")

    def render(self, request: OperatorRequest, destination: Path) -> str:
        command = [
            "copier",
            "copy",
            "--defaults",
            "--quiet",
            str(self.template_root),
            str(destination),
            "--data",
            f"operator_name={request.name}",
            "--data",
            f"operator_summary={request.rationale[:120]}",
            "--data",
            f"operator_family={request.family}",
            "--data",
            f"entrypoint={FAMILY_RUNNERS[request.family]}",
            "--data",
            f"runner_operation={_requested_operation(request)}",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise AdmissionError(f"copier failed: {result.stderr.strip()}")
        return digest_tree(destination)


def digest_tree(root: Path) -> str:
    """Content digest of a rendered package, excluding Copier's own answers file."""
    return content_digest(
        {
            path.relative_to(root).as_posix(): path.read_bytes().hex()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != ".copier-answers.yml"
        }
    )


def _requested_operation(request: OperatorRequest) -> str:
    operations = request.features & allowlist(request.family)
    if not operations:
        raise AdmissionError(
            f"{request.name} names no vetted operation for {request.family}; "
            f"vetted operations are {sorted(allowlist(request.family))}"
        )
    return sorted(operations)[0]


TestRunner = Callable[[Path], dict[str, bool]]


def run_test_tiers(package: Path) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for tier in TEST_TIERS:
        directory = package / "tests" / tier
        if not directory.is_dir():
            results[tier] = False
            continue
        outcome = subprocess.run(
            ["python", "-m", "pytest", "-q", str(directory)], capture_output=True, text=True
        )
        results[tier] = outcome.returncode == 0
    return results


@dataclass
class AdmissionService:
    """Deterministic admission.  The planner may approve only what this marks AUTO_ADMIT."""

    registry: OperatorRegistry
    audit: AuditStore
    workspace: Path
    renderer: PackageRenderer = field(default_factory=CopierRenderer)
    test_runner: TestRunner = run_test_tiers
    trusted_dependencies: frozenset[str] = frozenset()

    def evaluate(self, request: OperatorRequest) -> tuple[AdmissionReport, OperatorCandidate]:
        self.audit.record_operator_request(request)
        findings: list[AdmissionFinding] = []

        if request.kind is RequestKind.BEHAVIOUR:
            # No model can satisfy this and no runtime approval path exists for it: a new
            # runner operation is a human change to shakespeare/runners.py.
            findings.append(
                AdmissionFinding(
                    code="behaviour_requires_human_implementation",
                    message=(
                        "this request needs a runner operation that does not exist; "
                        "it is backlog for a person, not an admission decision"
                    ),
                )
            )

        if request.name in self.registry:
            findings.append(
                AdmissionFinding(
                    code="name_already_registered",
                    message=f"{request.name} is already a registered operator",
                )
            )

        untrusted = sorted(set(request.dependencies) - self.trusted_dependencies)
        if untrusted:
            findings.append(
                AdmissionFinding(
                    code="untrusted_dependency",
                    message=f"dependencies are outside the trust policy: {untrusted}",
                )
            )

        destination = self.workspace / f"candidate-{uuid4().hex}"
        try:
            digest = self.renderer.render(request, destination)
            # Rendering twice and comparing is what proves the package is a pure function
            # of its answers, and therefore that the audited digest means something.
            second = self.renderer.render(request, destination.with_suffix(".verify"))
            reproducible = digest == second
        except AdmissionError as exc:
            findings.append(AdmissionFinding(code="render_failed", message=str(exc)))
            digest, reproducible = "", False

        if not reproducible:
            findings.append(
                AdmissionFinding(
                    code="not_reproducible",
                    message="a second render did not reproduce the candidate digest",
                )
            )

        test_results = self.test_runner(destination) if digest else dict.fromkeys(TEST_TIERS, False)
        if not all(test_results.get(tier, False) for tier in TEST_TIERS):
            findings.append(
                AdmissionFinding(
                    code="tests_failed", message="all four family test tiers must pass"
                )
            )

        candidate = OperatorCandidate(
            candidate_id=uuid4().hex,
            request_id=request.request_id,
            spec=OperatorSpec(
                name=request.name,
                version="1.0.0",
                description=request.rationale,
                family=request.family,
                entrypoint=FAMILY_RUNNERS[request.family],
                features=request.features,
                side_effects=request.declared_side_effects,
                package_digest=digest or "unrendered",
            ),
            package_digest=digest or "unrendered",
            dependencies=request.dependencies,
            declared_side_effects=request.declared_side_effects,
        )

        risk = compute_risk(candidate)
        if risk is not RiskLevel.LOW:
            cause = "declared side effects" if candidate.declared_side_effects else "dependencies"
            # Escalating without saying why leaves a reviewer guessing and leaves the
            # audit log unable to explain the decision later.
            findings.append(
                AdmissionFinding(
                    code=f"risk_{risk}",
                    message=(
                        f"computed {risk} risk from "
                        f"{cause}"
                        f"; only low-risk declarative variants may be auto-approved"
                    ),
                    severity=FindingSeverity.WARNING,
                )
            )
        auto = (
            not [f for f in findings if f.severity is FindingSeverity.ERROR]
            and risk is RiskLevel.LOW
            and request.family in AUTO_ADMISSIBLE_FAMILIES
            and reproducible
        )
        if request.family is OperatorFamily.FILESYSTEM_MUTATION:
            findings.append(
                AdmissionFinding(
                    code="mutation_requires_human",
                    message="an operator that writes can never be auto-approved",
                    severity=FindingSeverity.WARNING,
                )
            )
            auto = False

        report = AdmissionReport(
            report_id=uuid4().hex,
            candidate_id=candidate.candidate_id,
            computed_risk=risk,
            disposition=(
                AdmissionDisposition.AUTO_ADMIT if auto else AdmissionDisposition.HUMAN_REVIEW
            ),
            findings=tuple(findings),
            test_results=test_results,
            reproducible=reproducible,
        )
        self.audit.record_admission_report(
            report, request_id=request.request_id, package_digest=candidate.package_digest
        )
        return report, candidate

    def decide(
        self,
        report: AdmissionReport,
        candidate: OperatorCandidate,
        *,
        decided_by: DecidedBy,
        choice: AdmissionChoice,
        rationale: str = "",
    ) -> AdmissionDecision:
        """Record a decision and, on approval, register the operator."""
        if (
            decided_by is DecidedBy.PLANNER
            and report.disposition is not AdmissionDisposition.AUTO_ADMIT
            and choice is AdmissionChoice.APPROVE
        ):
            raise AdmissionError(
                f"the planner may only approve an AUTO_ADMIT candidate; "
                f"{candidate.spec.name} is {report.computed_risk} risk and needs human review"
            )

        decision = AdmissionDecision(
            decision_id=uuid4().hex,
            report_id=report.report_id,
            decided_by=decided_by,
            choice=choice,
            rationale=rationale,
        )
        self.audit.record_admission_decision(decision)

        if choice is AdmissionChoice.APPROVE:
            admitted = candidate.spec.model_copy(update={"admission_id": report.report_id})
            self.registry.register(admitted)
            self.audit.record_operator_registration(admitted)
        return decision
