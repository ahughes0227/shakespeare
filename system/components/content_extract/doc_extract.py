"""doc.extract — extract text and provenance spans; backend selected by configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ...contracts import OperatorFamily, RiskLevel
from ..arguments import OperatorInput, config_value
from . import extraction

NAME = "doc.extract"
FAMILY = OperatorFamily.CONTENT_EXTRACT
OPERATION = "extract"
SUMMARY = "Extract text and provenance spans; backend selected by configuration."
FEATURES = frozenset({"char_limit", "extract", "fallback_chain", "page_limit"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 900.0
COMPOSABLE = True


class Input(OperatorInput):
    root: str = Field(description="Directory the items are relative to.")
    items: list[dict[str, Any]] = Field(
        description="Inventory entries, each with item_id, relpath and media_type."
    )


OUTPUTS = ("extractions", "usable", "unavailable")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    options = extraction.ExtractOptions(
        page_limit=int(config_value(arguments, "extract", "page_limit", 20)),
        char_limit=int(config_value(arguments, "extract", "char_limit", 200_000)),
    )
    root = Path(arguments["root"])
    items = tuple(
        extraction.Item(
            item_id=item["item_id"],
            path=root / item["relpath"],
            media_type=item.get("media_type", "application/octet-stream"),
        )
        for item in arguments["items"]
    )
    # `max_workers` is absent from FEATURES and from the `extract` config group on
    # purpose, so no composition can carry it and no subagent can ask for it. It reads
    # from the flat argument path only, which is the seam the unit tests use to pin the
    # serial and parallel routes against each other.
    requested = config_value(arguments, "extract", "max_workers", None)
    results = extraction.extract_many(
        items,
        backend=extraction.Backend(config_value(arguments, "extract", "backend", "auto_chain")),
        options=options,
        max_workers=None if requested is None else int(requested),
    )
    return {
        "extractions": [item.model_dump(mode="json") for item in results],
        "usable": sum(1 for item in results if item.usable),
        "unavailable": sum(1 for item in results if not item.usable),
    }
