"""Registration of the built-in operator set.

Few operators, each broad: one per verb, with the backend chosen by configuration.
`rename.py`-style generality is the point — `doc.extract` handles every media type
through its `backend` selection rather than through a dozen near-identical operators.
"""

from __future__ import annotations

from ..contracts import OperatorFamily, OperatorSpec, RiskLevel
from .arguments import INPUT_MODELS
from .registry import FAMILY_RUNNERS, OperatorRegistry

#: Operators that write.  Granted to no domain: agents plan, the runtime commits.
RUNTIME_ONLY: frozenset[str] = frozenset({"fs.stage", "fs.commit", "fs.reverse", "fs.discard"})


def _spec(
    name: str,
    family: OperatorFamily,
    description: str,
    *,
    operation: str,
    composable: bool = True,
    features: frozenset[str] = frozenset(),
    side_effects: tuple[str, ...] = (),
    risk: RiskLevel = RiskLevel.LOW,
    idempotent: bool = True,
    timeout_seconds: float = 300,
) -> OperatorSpec:
    """One operator, declared in one block, or not registered at all.

    An operator's definition is spread over several files by necessity — the spec here,
    its argument model and produced keys in `contracts`, its marshalling in `runners`, its
    logic in a domain module. That is tolerable while every piece is guaranteed to arrive;
    it is not while a piece can be forgotten. So the pieces are checked here, at import,
    rather than trusted: an operator missing its argument model or its declared outputs
    fails the moment the registry is built, naming what is absent.
    """
    from .arguments import INPUT_MODELS, OUTPUT_KEYS
    from .families import allowed_features
    from .runners import allowlist

    if operation not in allowlist(family):
        raise ValueError(
            f"{name}: {family.value} has no vetted operation {operation!r}; "
            f"vetted operations are {sorted(allowlist(family))}"
        )
    if operation not in allowed_features(family):
        raise ValueError(
            f"{name}: {family.value}'s manifest does not declare {operation!r}, so no "
            f"request could ever name it"
        )
    if composable:
        missing = [
            what
            for what, present in (
                ("an argument model in INPUT_MODELS", name in INPUT_MODELS),
                ("declared outputs in OUTPUT_KEYS", name in OUTPUT_KEYS),
            )
            if not present
        ]
        if missing:
            raise ValueError(f"{name} is registered without {' and '.join(missing)}")

    return OperatorSpec(
        name=name,
        version="1.0.0",
        description=description,
        family=family,
        entrypoint=FAMILY_RUNNERS[family],
        features=frozenset({operation}) | features,
        side_effects=side_effects,
        risk=risk,
        idempotent=idempotent,
        timeout_seconds=timeout_seconds,
    )


#: name -> (spec, runner operation).  The operation is what the runner dispatches.
BUILTIN: dict[str, tuple[OperatorSpec, str]] = {
    spec.name: (spec, operation)
    for spec, operation in (
        (
            _spec(
                "fs.scan",
                OperatorFamily.READONLY_SCAN,
                "Walk an input tree and inventory every file deterministically.",
                operation="walk",
                features=frozenset({"stat", "digest", "mime_detect", "stable_sort"}),
            ),
            "walk",
        ),
        (
            _spec(
                "fs.dirs",
                OperatorFamily.READONLY_SCAN,
                "List the directory structure of a tree.",
                operation="directories",
            ),
            "directories",
        ),
        (
            _spec(
                "fs.verify",
                OperatorFamily.READONLY_SCAN,
                "Re-scan a staged tree and compare it against a plan.",
                operation="verify_staging",
                features=frozenset({"digest"}),
            ),
            "verify_staging",
        ),
        (
            _spec(
                "doc.extract",
                OperatorFamily.CONTENT_EXTRACT,
                "Extract text and provenance spans; backend selected by configuration.",
                operation="extract",
                features=frozenset({"fallback_chain", "page_limit", "char_limit"}),
                timeout_seconds=900,
            ),
            "extract",
        ),
        (
            _spec(
                "text.normalize",
                OperatorFamily.PURE_TRANSFORM,
                "Normalise whitespace, case and aliases in extracted values.",
                operation="normalize",
            ),
            "normalize",
        ),
        (
            _spec(
                "spec.freeze",
                OperatorFamily.PURE_TRANSFORM,
                "Validate a proposed naming convention and freeze it under a digest.",
                operation="freeze_spec",
            ),
            "freeze_spec",
        ),
        (
            _spec(
                "name.render",
                OperatorFamily.PURE_TRANSFORM,
                "Render filenames from a frozen spec and field values.",
                operation="render_template",
            ),
            "render_template",
        ),
        (
            _spec(
                "name.collide",
                OperatorFamily.PURE_TRANSFORM,
                "Resolve duplicate targets deterministically across a whole set.",
                operation="collision_resolve",
            ),
            "collision_resolve",
        ),
        (
            _spec(
                "batch.window",
                OperatorFamily.PURE_TRANSFORM,
                "Take the next slice of work not yet done, so a large set can be "
                "processed in windows.",
                operation="next_window",
            ),
            "next_window",
        ),
        (
            _spec(
                "schedule.plan",
                OperatorFamily.PURE_TRANSFORM,
                "Size the next batch of work from what earlier batches actually cost, "
                "or report that the whole set already fits.",
                operation="plan_batch",
            ),
            "plan_batch",
        ),
        (
            _spec(
                "record.append",
                OperatorFamily.RECORD_STORE,
                "Store one row per item in the run's record table, replacing any row "
                "already held for that item.",
                operation="record_append",
            ),
            "record_append",
        ),
        (
            _spec(
                "record.read",
                OperatorFamily.RECORD_STORE,
                "Read back every stored row, so later work runs from the table rather "
                "than from a model response.",
                operation="record_read",
            ),
            "record_read",
        ),
        (
            _spec(
                "plan.assemble",
                OperatorFamily.PURE_TRANSFORM,
                "Assemble a ChangePlan and enforce balanced accounting.",
                operation="plan_assemble",
            ),
            "plan_assemble",
        ),
        (
            _spec(
                "check.assert",
                OperatorFamily.PURE_TRANSFORM,
                "Run a named deterministic obligation check.",
                operation="obligation_check",
            ),
            "obligation_check",
        ),
        (
            _spec(
                "fs.stage",
                OperatorFamily.FILESYSTEM_MUTATION,
                "Materialise a plan into a staging tree by copying.",
                composable=False,
                operation="stage_write",
                features=frozenset({"mirror_tree", "journal_reverse"}),
                side_effects=("write:staging_root",),
                risk=RiskLevel.HIGH,
                timeout_seconds=3600,
            ),
            "stage_write",
        ),
        (
            _spec(
                "fs.commit",
                OperatorFamily.FILESYSTEM_MUTATION,
                "Atomically move a verified staging tree into the output root.",
                composable=False,
                operation="atomic_move",
                features=frozenset({"atomic_move", "idempotency_receipt"}),
                side_effects=("write:output_root",),
                risk=RiskLevel.HIGH,
                idempotent=False,
            ),
            "atomic_move",
        ),
        (
            _spec(
                "fs.reverse",
                OperatorFamily.FILESYSTEM_MUTATION,
                "Reverse one journaled mutation.",
                composable=False,
                operation="journal_reverse",
                features=frozenset({"journal_reverse"}),
                side_effects=("write:output_root",),
                risk=RiskLevel.HIGH,
                idempotent=False,
            ),
            "journal_reverse",
        ),
        (
            _spec(
                "fs.discard",
                OperatorFamily.FILESYSTEM_MUTATION,
                "Discard an uncommitted staging tree.",
                composable=False,
                operation="discard",
                side_effects=("write:staging_root",),
                risk=RiskLevel.HIGH,
            ),
            "discard",
        ),
    )
}


def operation_of(name: str) -> str:
    return BUILTIN[name][1]


def build_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    for spec, _ in BUILTIN.values():
        registry.register(spec, input_model=INPUT_MODELS.get(spec.name))
    return registry
