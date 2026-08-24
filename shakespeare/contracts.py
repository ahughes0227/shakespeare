"""Typed contracts for every boundary in the runtime.

Every model here is frozen and forbids unknown fields.  A contract is the surface the
programmer fixes; models fill values inside it and can never widen it.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_json(value: Any) -> str:
    """Stable JSON used for every digest.  Sorted keys, no incidental whitespace."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def digest(self) -> str:
        return content_digest(self)


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


RISK_ORDER: tuple[RiskLevel, ...] = (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)


class OperatorFamily(StrEnum):
    READONLY_SCAN = "readonly_scan"
    CONTENT_EXTRACT = "content_extract"
    PURE_TRANSFORM = "pure_transform"
    FILESYSTEM_MUTATION = "filesystem_mutation"


#: The only families a requested operator may be auto-admitted into.
AUTO_ADMISSIBLE_FAMILIES: frozenset[OperatorFamily] = frozenset(
    {OperatorFamily.PURE_TRANSFORM, OperatorFamily.READONLY_SCAN}
)


class ErrorCode(StrEnum):
    """Closed failure taxonomy.  Free-text errors are never recorded: the SLIs in
    `audit.metrics` aggregate over exactly these values."""

    MODEL_TRANSIENT = "model_transient"
    MODEL_PERMANENT = "model_permanent"
    COMPOSITION_INVALID = "composition_invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"
    OPERATOR_FAILED = "operator_failed"
    OBLIGATION_FAILED = "obligation_failed"
    EXTRACTION_UNAVAILABLE = "extraction_unavailable"
    ADMISSION_DENIED = "admission_denied"
    APPROVAL_TIMEOUT = "approval_timeout"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    COMMIT_VERIFICATION_FAILED = "commit_verification_failed"


class ChangeAction(StrEnum):
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    UNRESOLVED = "unresolved"


class StageDecision(StrEnum):
    ACCEPT = "accept"
    RERUN = "rerun"
    ABORT = "abort"


class FindingSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class RequestKind(StrEnum):
    #: New configuration over an existing trusted-runner operation.  Admissible.
    VARIANT = "variant"
    #: Needs a runner operation that does not exist.  Human backlog; never admissible.
    BEHAVIOUR = "behaviour"


class AdmissionDisposition(StrEnum):
    AUTO_ADMIT = "auto_admit"
    HUMAN_REVIEW = "human_review"


class AdmissionChoice(StrEnum):
    APPROVE = "approve"
    DENY = "deny"


class DecidedBy(StrEnum):
    PLANNER = "planner"
    HUMAN = "human"
    AUTO = "auto"


# --------------------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------------------

_ALLOWANCE = re.compile(r"^\s*(\d+)\s*(?:\+\s*(\d+)\s*\*\s*n\s*)?$")


class Allowance(Contract):
    """A per-item budget allowance, written `base` or `base + per_item*n`.

    File counts are unbounded, so allowances resolve against the item count at stage start
    rather than being fixed constants.
    """

    base: int = Field(ge=0)
    per_item: int = Field(default=0, ge=0)

    @classmethod
    def parse(cls, value: Allowance | int | str) -> Allowance:
        if isinstance(value, Allowance):
            return value
        if isinstance(value, int):
            return cls(base=value)
        match = _ALLOWANCE.match(value)
        if match is None:
            raise ValueError(f"allowance must be 'base' or 'base + per_item*n': {value!r}")
        return cls(base=int(match.group(1)), per_item=int(match.group(2) or 0))

    def resolve(self, items: int) -> int:
        return self.base + self.per_item * max(0, items)


class BudgetEnvelope(Contract):
    model_invocations: Allowance = Allowance(base=8)
    total_tokens: Allowance = Allowance(base=200_000)
    operator_calls: Allowance = Allowance(base=20, per_item=4)
    wall_time_seconds: int = Field(default=1800, gt=0)
    max_cost_usd: float = Field(default=10.0, ge=0)

    @field_validator("model_invocations", "total_tokens", "operator_calls", mode="before")
    @classmethod
    def _parse(cls, value: object) -> object:
        if isinstance(value, (str, int)):
            return Allowance.parse(value)
        return value


class BudgetUsage(Contract):
    model_invocations: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    operator_calls: int = Field(default=0, ge=0)
    wall_time_seconds: float = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)


# --------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------


class OperatorSpec(Contract):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    family: OperatorFamily
    entrypoint: str = Field(min_length=1)
    features: frozenset[str] = frozenset()
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    side_effects: tuple[str, ...] = ()
    risk: RiskLevel = RiskLevel.LOW
    idempotent: bool = True
    timeout_seconds: float = Field(default=300, gt=0, le=86_400)
    template_revision: str = "builtin-1"
    package_digest: str = "builtin"
    admission_id: str = "bootstrap"


class Invocation(Contract):
    """One operator call.  A node in the stage DAG."""

    invocation_id: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    #: Hydra `group=choice` selections, e.g. {"extract": "pdf_text"}.
    selections: dict[str, str] = Field(default_factory=dict)
    #: Bounded typed parameters.  Never interpolated.
    parameters: dict[str, Any] = Field(default_factory=dict)
    #: Stage input names, or `invocation_id`s of earlier invocations in this composition.
    inputs: tuple[str, ...] = ()
    #: Rename resolved keys: {argument_name: source_key}.  Lets an agent wire one
    #: operator's output onto the next operator's parameter without either having to
    #: agree on vocabulary in advance.
    bindings: dict[str, str] = Field(default_factory=dict)

    @field_validator("bindings")
    @classmethod
    def _identifier_keys(cls, value: dict[str, str]) -> dict[str, str]:
        for target, source in value.items():
            if not target.isidentifier():
                raise ValueError(f"binding target must be an identifier: {target}")
            if not all(part.isidentifier() for part in source.split(".")):
                raise ValueError(f"binding source must be a dotted identifier: {source}")
        return value


class OperatorAsk(Contract):
    """A subagent's request for an operator it does not have.

    Note what it cannot set: no ids, no version, no entrypoint, no risk. Those are
    stamped by the runtime or computed by admission, so a request describes a need and
    never asserts an authority.
    """

    kind: RequestKind
    family: OperatorFamily
    name: str = Field(min_length=1)
    features: frozenset[str] = frozenset()
    dependencies: tuple[str, ...] = ()
    declared_side_effects: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)


class Composition(Contract):
    """The entire output surface of a domain subagent."""

    domain_id: str = Field(min_length=1)
    invocations: tuple[Invocation, ...]
    rationale: str = ""
    #: An operator the subagent lacked. Evaluated after this composition runs, so an
    #: admitted operator is usable from the next attempt onward — never mid-composition.
    ask: OperatorAsk | None = None

    @model_validator(mode="after")
    def _validate(self) -> Composition:
        seen: set[str] = set()
        for invocation in self.invocations:
            if invocation.invocation_id in seen:
                raise ValueError(f"duplicate invocation id: {invocation.invocation_id}")
            for reference in invocation.inputs:
                # A reference is either a stage input name or an earlier invocation.
                # Forward references would make the DAG cyclic.
                if reference in seen:
                    continue
                if reference in {item.invocation_id for item in self.invocations}:
                    raise ValueError(
                        f"invocation {invocation.invocation_id} references a later"
                        f" invocation: {reference}"
                    )
            seen.add(invocation.invocation_id)
        return self

    def edges(self) -> tuple[tuple[str, str], ...]:
        """Data dependencies, as (from_invocation, to_invocation) pairs."""
        ids = {item.invocation_id for item in self.invocations}
        return tuple(
            (reference, invocation.invocation_id)
            for invocation in self.invocations
            for reference in invocation.inputs
            if reference in ids
        )


# --------------------------------------------------------------------------------------
# Domains and stages
# --------------------------------------------------------------------------------------


class DomainSpec(Contract):
    id: str = Field(min_length=1)
    #: The immutable bound.  A planner goal must sit inside this; it can never widen it.
    scope: str = Field(min_length=1)
    skippable: bool = True
    catalog: frozenset[str] = Field(min_length=1)
    config_groups: frozenset[str] = frozenset()
    prompt_version: str = "1.0.0"


class DomainGoal(Contract):
    """The planner's per-run goal for an activated domain.

    Note what is absent: no catalog, no config groups, no budget.  Whatever the goal text
    says, the executable surface comes from `DomainSpec` alone.
    """

    domain_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    success_criterion: str = Field(min_length=1)
    obligation_refs: tuple[str, ...] = ()


class SkipDecision(Contract):
    domain_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class StagePlan(Contract):
    activated: tuple[DomainGoal, ...] = ()
    skipped: tuple[SkipDecision, ...] = ()

    @model_validator(mode="after")
    def _disjoint(self) -> StagePlan:
        activated = {item.domain_id for item in self.activated}
        skipped = {item.domain_id for item in self.skipped}
        if activated & skipped:
            raise ValueError(f"domains both activated and skipped: {sorted(activated & skipped)}")
        return self


class Obligation(Contract):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    #: Name of the deterministic checker operator.  Never an agent.
    checker: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ObligationResult(Contract):
    obligation_id: str
    passed: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class StageVerdict(Contract):
    met: bool
    unmet: tuple[str, ...] = ()
    decision: StageDecision
    revised_goals: tuple[DomainGoal, ...] = ()
    rationale: str = ""


class StageSpec(Contract):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    input_contract: str = Field(min_length=1)
    output_contract: str = Field(min_length=1)
    max_attempts: int = Field(default=2, ge=1, le=10)
    domains: tuple[DomainSpec, ...] = Field(min_length=1)
    obligations: tuple[str, ...] = ()
    budget: BudgetEnvelope = BudgetEnvelope()
    side_effects: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _unique_domains(self) -> StageSpec:
        ids = [item.id for item in self.domains]
        if len(ids) != len(set(ids)):
            raise ValueError("domain ids must be unique within a stage")
        return self

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def domain(self, domain_id: str) -> DomainSpec:
        for item in self.domains:
            if item.id == domain_id:
                return item
        raise KeyError(f"unknown domain: {self.name}.{domain_id}")


TEN_FIELD_CARD: tuple[str, ...] = (
    "purpose",
    "lifecycle",
    "contracts",
    "allowed_configuration",
    "side_effects",
    "risks",
    "failure_modes",
    "resource_limits",
    "examples",
    "provenance",
)


class SemanticCard(Contract):
    """The ten-field description shared by families, stages and workflows.

    It is the only thing an upstream reader — the planner, or a human — is given about a
    unit, which is why every field is mandatory.
    """

    purpose: str = Field(min_length=1)
    lifecycle: str = Field(min_length=1)
    contracts: str = Field(min_length=1)
    allowed_configuration: str = Field(min_length=1)
    side_effects: str = Field(min_length=1)
    risks: str = Field(min_length=1)
    failure_modes: str = Field(min_length=1)
    resource_limits: str = Field(min_length=1)
    examples: str = Field(min_length=1)
    provenance: str = Field(min_length=1)


# --------------------------------------------------------------------------------------
# Workflows
# --------------------------------------------------------------------------------------


_STAGE_REF = re.compile(r"^[a-z][a-z0-9_]*@\d+\.\d+\.\d+$")


class WorkflowSpec(Contract):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    #: Ordered `stage@version` refs.  Programmer-authored; the planner never writes these.
    spine: tuple[str, ...] = Field(min_length=1)
    #: Stage name after which the atomic commit runs.
    commit_after: str = Field(min_length=1)
    entry_contract: str = "RequestContract"

    @field_validator("spine")
    @classmethod
    def _refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for ref in value:
            if not _STAGE_REF.match(ref):
                raise ValueError(f"spine entry must be 'name@major.minor.patch': {ref!r}")
        if len({ref.split("@")[0] for ref in value}) != len(value):
            raise ValueError("a stage may appear at most once in a spine")
        return value

    @model_validator(mode="after")
    def _commit_in_spine(self) -> WorkflowSpec:
        names = [ref.split("@")[0] for ref in self.spine]
        if self.commit_after not in names:
            raise ValueError(f"commit_after names a stage not in the spine: {self.commit_after}")
        return self


class RouteDecision(Contract):
    """The planner's entire output surface when selecting a workflow."""

    workflow_id: str
    rationale: str = ""
    supported: bool = True


class RequestContract(Contract):
    request_id: str
    prompt: str = Field(min_length=1)
    input_root: str
    output_root: str


# --------------------------------------------------------------------------------------
# Change plans
# --------------------------------------------------------------------------------------


class ChangeEntry(Contract):
    """Base plan entry.  The runtime touches only these fields, which is what keeps
    accounting, preview, commit and undo generic across workflows.

    Extras are allowed, deliberately: a workflow subclasses this to add its own fields
    (a rename adds `target_relpath`), and a plan round-tripped through JSON must carry
    them back without the runtime having to know which subclass produced it.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    item_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    action: ChangeAction
    reason: str = ""
    digests: dict[str, str] = Field(default_factory=dict)


class ChangePlan(Contract):
    run_id: str
    workflow_id: str
    workflow_digest: str
    decision_digest: str
    operator_versions: dict[str, str] = Field(default_factory=dict)
    #: SerializeAsAny, because workflows subclass ChangeEntry.  Without it Pydantic dumps
    #: against the declared base type and silently drops subclass fields such as
    #: `target_relpath`, producing a plan whose entries have no destination.
    entries: tuple[SerializeAsAny[ChangeEntry], ...] = ()

    def fingerprint(self) -> str:
        """Identity of the decisions, independent of which run made them.

        `digest()` covers run_id, so two runs of the same request never share it. The
        idempotency receipt needs the opposite: the same decisions to the same place must
        be recognisable as the same plan.
        """
        return content_digest(
            {
                "workflow_id": self.workflow_id,
                "entries": sorted(
                    (
                        entry.source_ref,
                        str(entry.action),
                        getattr(entry, "target_relpath", None),
                    )
                    for entry in self.entries
                ),
            }
        )

    def balanced(self, scanned: int) -> bool:
        return len(self.entries) == scanned and len({e.item_id for e in self.entries}) == scanned

    def count(self, action: ChangeAction) -> int:
        return sum(1 for entry in self.entries if entry.action == action)


class ReversalRecord(Contract):
    mutation_id: str
    operation: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ApplyReceipt(Contract):
    run_id: str
    item_id: str
    action: ChangeAction
    target_ref: str | None = None
    reversal: ReversalRecord | None = None
    error_code: ErrorCode | None = None


# --------------------------------------------------------------------------------------
# Operator requests and admission
# --------------------------------------------------------------------------------------


class OperatorRequest(Contract):
    request_id: str
    run_id: str
    domain_id: str
    kind: RequestKind
    family: OperatorFamily
    name: str = Field(min_length=1)
    features: frozenset[str] = frozenset()
    dependencies: tuple[str, ...] = ()
    declared_side_effects: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)


class OperatorCandidate(Contract):
    candidate_id: str
    request_id: str
    spec: OperatorSpec
    package_digest: str
    dependencies: tuple[str, ...] = ()
    declared_side_effects: tuple[str, ...] = ()


class AdmissionFinding(Contract):
    code: str
    message: str
    severity: FindingSeverity = FindingSeverity.ERROR


class AdmissionReport(Contract):
    report_id: str
    candidate_id: str
    computed_risk: RiskLevel
    disposition: AdmissionDisposition
    findings: tuple[AdmissionFinding, ...] = ()
    test_results: dict[str, bool] = Field(default_factory=dict)
    reproducible: bool = False


class AdmissionDecision(Contract):
    decision_id: str
    report_id: str
    decided_by: DecidedBy
    choice: AdmissionChoice
    rationale: str = ""


# --------------------------------------------------------------------------------------
# Prompt artifacts
# --------------------------------------------------------------------------------------


class PromptArtifact(Contract):
    """A prompt compiled offline by DSPy, pinned by version.

    The version is part of the workflow digest, so promoting a prompt is a visible,
    versioned change and `replay` resolves the prompt a run actually used.
    """

    signature_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    demonstrations: tuple[dict[str, Any], ...] = ()
    compiled_from: dict[str, Any] = Field(default_factory=dict)


class OptimizationRun(Contract):
    optimization_id: str
    signature_id: str
    optimizer: str
    eval_set_digest: str
    incumbent_version: str | None
    incumbent_score: float | None
    candidate_version: str
    candidate_score: float
    fixture_regressions: tuple[str, ...] = ()


class PromotionDecision(Contract):
    decision_id: str
    optimization_id: str
    decided_by: DecidedBy
    choice: AdmissionChoice
    rationale: str = ""


# --------------------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------------------


class TelemetryEnvelope(Contract):
    """The only shape permitted to reach an exporter.

    There are no free-form value fields by construction: attributes are ids, digests,
    versions, counts, timings, costs and closed error codes.  Raw document content is
    never handed to the tracer, so there is nothing for a masking hook to miss.
    """

    run_id: str
    span: str = Field(min_length=1)
    stage: str | None = None
    attempt: int | None = None
    domain: str | None = None
    operator: str | None = None
    operator_version: str | None = None
    prompt_version: str | None = None
    requested_model: str | None = None
    resolved_model: str | None = None
    provider: str | None = None
    digests: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    duration_ms: float | None = None
    cost_usd: float | None = None
    error_code: ErrorCode | None = None

    @field_validator("digests")
    @classmethod
    def _digest_shape(cls, value: dict[str, str]) -> dict[str, str]:
        for key, digest in value.items():
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"digests must be sha256 hex, not content: {key}")
        return value
