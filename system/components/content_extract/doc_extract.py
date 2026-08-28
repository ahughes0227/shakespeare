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
    results = [
        extraction.extract(
            item_id=item["item_id"],
            path=Path(arguments["root"]) / item["relpath"],
            media_type=item.get("media_type", "application/octet-stream"),
            backend=extraction.Backend(config_value(arguments, "extract", "backend", "auto_chain")),
            options=options,
        )
        for item in arguments["items"]
    ]
    return {
        "extractions": [item.model_dump(mode="json") for item in results],
        "usable": sum(1 for item in results if item.usable),
        "unavailable": sum(1 for item in results if not item.usable),
    }
