"""What an operator is given, and how to read it.

The base every argument model derives from, plus the two helpers an operator needs to read
its arguments. The models themselves live with their operators — one file per operator, in
its family — so that opening an operator shows what it takes without a second file.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class RunnerError(RuntimeError):
    pass

class OperatorInput(BaseModel):
    model_config = ConfigDict(extra="ignore")


def config_value(arguments: dict[str, Any], group: str, key: str, default: Any = None) -> Any:
    """Read a value from the composed Hydra config, allowing a direct override.

    The executor passes the whole composed mapping under `config`; a caller may still
    pass a flat key, which is what keeps operators unit-testable without a composition.
    """
    if key in arguments:
        return arguments[key]
    group_config = (arguments.get("config") or {}).get(group) or {}
    return group_config.get(key, default)


_SPEC_SHAPES: dict[str, str] = {
    "policy.max_length": (
        " (a single whole-filename cap, e.g. 200. For a per-field cap put max_length on "
        "the field itself)"
    ),
    "policy.replacement": " (a single character, e.g. \"-\")",
    "policy.separator": " (a string placed between fields, e.g. \", \")",
    "policy.case": " (one of preserve, lower, upper, title)",
    "policy.aliases": " (a flat mapping of name to name)",
}


def arguments_shape_hint(location: str) -> str:
    return _SPEC_SHAPES.get(location, "")
