"""Input contracts for the built-in operators.

Every operator declared its family, its risk and its side effects, but not what arguments
it takes. A domain subagent was therefore told *which* operators it may call and left to
guess *how* — so a live model invented `path`, `dir`, `out_dir` and `output_format`, and
the failures surfaced as opaque operator errors rather than contract violations.

These models are `extra="ignore"` on purpose: the executor splats prior outputs and the
composed config into the argument mapping, so an operator sees far more than it declares.
What matters is that everything it *requires* is present and well-typed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OperatorInput(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ScanInput(OperatorInput):
    root: str = Field(
        description="Directory to walk. Bind from a stage input; never a literal path."
    )
    depth_limit: int = Field(default=32, ge=1, le=64)
    include_hidden: bool = False


class DirectoriesInput(OperatorInput):
    root: str = Field(description="Directory whose structure to list. Bind from a stage input.")


class VerifyStagingInput(OperatorInput):
    plan: dict[str, Any] = Field(description="The ChangePlan to verify against.")
    staging_root: str


class ExtractInput(OperatorInput):
    root: str = Field(description="Directory the items are relative to.")
    items: list[dict[str, Any]] = Field(
        description="Inventory entries, each with item_id, relpath and media_type."
    )


class NormalizeInput(OperatorInput):
    values: dict[str, Any] = Field(description="Field name to raw value.")


class FreezeSpecInput(OperatorInput):
    spec: dict[str, Any] = Field(
        description="Naming spec: template, fields, policy, collision_policy."
    )


class RenderInput(OperatorInput):
    """Either a frozen `spec`, or `template` plus `fields`.

    Items may be supplied explicitly, or omitted entirely — in which case they are derived
    from the inventory, which is how a sequential convention costs no transcription.
    """

    spec: dict[str, Any] | None = None
    template: str | None = None
    fields: list[dict[str, Any]] | None = None
    items: list[dict[str, Any]] | None = None
    scanned: list[dict[str, Any]] | None = None


class CollisionInput(OperatorInput):
    candidates: list[dict[str, Any]] = Field(
        description="Rendered names: item_id, directory, name. Bind from name.render."
    )
    unrendered: list[dict[str, Any]] | None = None


class PlanAssembleInput(OperatorInput):
    run_id: str
    workflow_id: str
    workflow_digest: str
    decision_digest: str
    scanned: list[dict[str, Any]] = Field(description="The inventory. Bind from fs.scan.")
    skipped: list[dict[str, Any]] | None = Field(
        default=None,
        description="Unreadable paths from fs.scan. Bind them so they appear in the plan.",
    )
    planned: list[dict[str, Any]] | None = Field(
        default=None,
        description="Resolved names. Must be bound from name.collide, never written by hand.",
    )


class WindowInput(OperatorInput):
    items: list[dict[str, Any]] | None = Field(
        default=None, description="The full set of work. Bind from fs.scan."
    )
    scanned: list[dict[str, Any]] | None = None
    completed: list[Any] | None = Field(
        default=None,
        description="Work already done in earlier windows. Bind from the accumulated "
        "results; ids or records both work.",
    )


class BatchPlanInput(OperatorInput):
    remaining: list[dict[str, Any]] | None = Field(
        default=None, description="Work not yet handed over."
    )
    items: list[dict[str, Any]] | None = None
    capacity: int = Field(description="Output tokens one response may use.")
    cost_per_item: int = Field(description="Starting estimate of one item's cost.")
    observations: list[dict[str, Any]] | None = Field(
        default=None, description="What earlier batches actually cost."
    )
    weights: list[int] | None = Field(
        default=None, description="How much material each remaining item carries."
    )


class RecordAppendInput(OperatorInput):
    rows: list[dict[str, Any]] = Field(
        description="One row per item. Each must carry the key column."
    )
    table: str = Field(default="items", description="Which table to write.")
    key: str = Field(default="item_id", description="Column identifying an item.")


class RecordReadInput(OperatorInput):
    table: str = Field(default="items", description="Which table to read.")


class ObligationCheckInput(OperatorInput):
    obligation_id: str
    check: str
    payload: dict[str, Any] = Field(default_factory=dict)


#: operator name -> input contract. Only the composable operators need one; the
#: runtime-only mutation operators are never called by an agent.
INPUT_MODELS: dict[str, type[OperatorInput]] = {
    "fs.scan": ScanInput,
    "fs.dirs": DirectoriesInput,
    "fs.verify": VerifyStagingInput,
    "doc.extract": ExtractInput,
    "text.normalize": NormalizeInput,
    "spec.freeze": FreezeSpecInput,
    "name.render": RenderInput,
    "name.collide": CollisionInput,
    "batch.window": WindowInput,
    "schedule.plan": BatchPlanInput,
    "record.append": RecordAppendInput,
    "record.read": RecordReadInput,
    "plan.assemble": PlanAssembleInput,
    "check.assert": ObligationCheckInput,
}


#: What each operator puts into the argument mapping for later invocations. Declaring
#: inputs without outputs left an agent able to call an operator but unable to wire one
#: into the next — it had to guess the key to bind from, and guessed wrong.
OUTPUT_KEYS: dict[str, list[str]] = {
    "fs.scan": ["items", "skipped", "count"],
    "fs.dirs": ["directories"],
    "fs.verify": ["ok", "missing", "mismatched", "staged_files", "planned_entries"],
    "doc.extract": ["extractions", "usable", "unavailable"],
    "text.normalize": ["values"],
    "spec.freeze": ["spec", "digest"],
    "name.render": ["results", "candidates", "unrendered"],
    "name.collide": ["resolutions"],
    "batch.window": ["window", "window_size", "remaining", "completed_count", "exhausted"],
    "schedule.plan": [
        "needed",
        "batch",
        "batch_size",
        "batch_weight",
        "remaining_count",
        "estimate",
    ],
    "plan.assemble": ["plan", "entries", "scanned"],
    "record.append": ["table", "stored", "added", "replaced", "path"],
    "record.read": ["records", "stored", "table"],
    "check.assert": ["obligation_id", "passed", "detail"],
}


def argument_summary(name: str) -> dict[str, Any]:
    """What a subagent needs to call an operator correctly.

    Deliberately not the raw JSON schema: a subagent needs the argument names, which are
    required, and the one-line note on where a value should come from.
    """
    model = INPUT_MODELS.get(name)
    if model is None:
        return {}
    required: list[dict[str, Any]] = []
    optional: list[dict[str, Any]] = []
    for field, info in model.model_fields.items():
        entry: dict[str, Any] = {"name": field}
        if info.description:
            entry["note"] = info.description
        (required if info.is_required() else optional).append(entry)
    return {
        "required": required,
        "optional": optional,
        # Naming an earlier invocation in `inputs` splats these keys into the arguments;
        # `bindings` renames one onto a different argument.
        "produces": OUTPUT_KEYS.get(name, []),
    }
