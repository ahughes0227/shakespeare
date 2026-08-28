"""spec.freeze — validate a proposed naming convention and freeze it under a digest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from ...contracts import OperatorFamily, RiskLevel
from ..arguments import OperatorInput, RunnerError, arguments_shape_hint
from . import naming

NAME = "spec.freeze"
FAMILY = OperatorFamily.PURE_TRANSFORM
OPERATION = "freeze_spec"
SUMMARY = "Validate a proposed naming convention and freeze it under a digest."
FEATURES = frozenset({"freeze_spec"})
SIDE_EFFECTS = ()
RISK = RiskLevel.LOW
IDEMPOTENT = True
TIMEOUT_SECONDS = 300.0
COMPOSABLE = True


class Input(OperatorInput):
    spec: dict[str, Any] = Field(
        description="Naming spec: template, fields, policy, collision_policy."
    )


OUTPUTS = ("spec", "digest")


def run(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    payload = arguments["spec"]
    try:
        parsed = naming.NamingSpec.model_validate(payload)
    except ValidationError as exc:
        # A spec is executable, not documentary, so commentary keys are rejected — but
        # the message has to name them or the next attempt guesses again.
        extras = sorted(
            ".".join(str(part) for part in item["loc"])
            for item in exc.errors()
            if item["type"] == "extra_forbidden"
        )
        if extras:
            raise RunnerError(
                f"the naming spec carries keys it does not support: {extras}. "
                f"A spec holds only template, fields, policy, collision_policy and "
                f"confidence_floor; each field holds only name, kind, format, required, "
                f"confidence_floor and max_length. Put nothing else in it."
            ) from exc
        # Name the field and the shape it wants. A raw pydantic dump tells a model that
        # something is wrong without telling it what to write instead.
        problems = [
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            f"{arguments_shape_hint('.'.join(str(part) for part in item['loc']))}"
            for item in exc.errors()[:6]
        ]
        raise RunnerError("the naming spec is invalid - " + "; ".join(problems)) from exc
    spec, digest = naming.freeze_spec(parsed)
    return {"spec": spec.model_dump(mode="json"), "digest": digest}
