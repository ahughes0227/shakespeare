"""name.render — render filenames from a frozen spec and field values.

Every operator module holds the same five things in this order: what it is, what it takes,
what it produces, and how it runs. The behaviour it calls lives in this family's own logic
modules; nothing here does the work itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts import OperatorFamily, RiskLevel
from ..arguments import OperatorInput, config_value
from . import naming

NAME = "name.render"
FAMILY = OperatorFamily.PURE_TRANSFORM
OPERATION = "render_template"
SUMMARY = "Render filenames from a frozen spec and field values."
FEATURES = frozenset({"render_template"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    """Either a frozen `spec`, or `template` plus `fields`.

    Items may be supplied explicitly, or omitted entirely — in which case they are derived
    from the inventory, which is how a sequential convention costs no transcription.
    """

    spec: dict[str, Any] | None = None
    template: str | None = None
    fields: list[dict[str, Any]] | None = None
    items: list[dict[str, Any]] | None = None
    scanned: list[dict[str, Any]] | None = None
    #: Rows straight from the record store. Bind from record.read, whose output key is
    #: `records`: the renderer's items now legitimately come from storage, and a live run
    #: stored all sixty rows, read them back, and rendered nothing because the key it
    #: passed was not one this operator would answer to.
    records: list[dict[str, Any]] | None = None


OUTPUTS = ("results", "candidates", "unrendered")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    # A frozen spec carries the template, fields and policy together, so accepting one
    # directly is what lets a composition bind spec.freeze straight into the renderer.
    spec_payload = arguments.get("spec")
    if spec_payload is not None:
        spec = naming.NamingSpec.model_validate(spec_payload)
        template = spec.template
        fields = spec.fields
        policy = spec.policy
    else:
        template = arguments["template"]
        fields = tuple(naming.FieldDecl.model_validate(item) for item in arguments["fields"])
        policy = naming.policy_from(arguments)

    items = naming.render_items(arguments)
    if not items:
        # Rendering nothing is never the answer, and reporting it as success is how a run
        # stores sixty records, renders none of them, and fails a gate two goals later
        # with no sign of where it went wrong.
        raise ValueError(
            "name.render was given no items. Bind them from record.read (`records`), "
            "fs.scan (`items`), or pass `items` directly"
        )
    floor = config_value(arguments, "confidence", "floor", None)
    if spec_payload is not None and floor is None:
        floor = naming.NamingSpec.model_validate(spec_payload).confidence_floor
    results = [
        naming.render(
            item_id=item["item_id"],
            template=template,
            fields=fields,
            values=item.get("values", {}),
            policy=policy,
            extension=item.get("extension", ""),
            sequence=item.get("sequence", index + 1),
            confidences=item.get("confidences"),
            floor=float(floor) if floor is not None else None,
        )
        for index, item in enumerate(items)
    ]
    directories = {item["item_id"]: item.get("directory", "") for item in items}
    claimed = {item["item_id"]: item for item in items}
    results = [
        item.model_copy(
            update={
                "values": claimed.get(item.item_id, {}).get("values") or {},
                "confidences": claimed.get(item.item_id, {}).get("confidences") or {},
                "extension": claimed.get(item.item_id, {}).get("extension") or "",
            }
        )
        for item in results
    ]
    return {
        "results": [item.model_dump(mode="json") for item in results],
        # Shaped for name.collide, so a composition can bind one straight into the other.
        "candidates": [
            {
                "item_id": item.item_id,
                "directory": directories.get(item.item_id, ""),
                "name": item.rendered,
            }
            for item in results
            if item.rendered is not None
        ],
        "unrendered": [
            {"item_id": item.item_id, "reason": item.reason}
            for item in results
            if item.rendered is None
        ],
    }
