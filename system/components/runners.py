"""Trusted family runners.

Generated operator packages are declarative and contain no callable.  All behaviour
reaches the runtime through exactly one runner per family, and each runner dispatches a
named `operation` against a closed allowlist of vetted functions.

This is what makes it safe for a subagent to request an operator: a request can select
vetted behaviour and configure it, but adding a *new* operation requires a human to edit
the allowlists below and pass the family test tiers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..contracts import OperatorFamily
from .arguments import RunnerError
from .catalog import MODULES

Operation = Callable[[dict[str, Any], Path], dict[str, Any]]


def _dispatch(
    arguments: dict[str, Any], workspace: Path, allowed: dict[str, Operation]
) -> dict[str, Any]:
    payload = dict(arguments)
    operation = payload.pop("operation", None)
    if operation not in allowed:
        raise RunnerError(
            f"unsupported operation: {operation!r}. "
            f"Vetted operations for this family: {sorted(allowed)}"
        )
    return allowed[operation](payload, workspace)


# --------------------------------------------------------------------------------------
# readonly_scan
# -------------------------------------------------------------------------------------


#: family -> operation -> the module's `run`. Read from the operator files rather than
#: listed here, so a new operator is dispatchable the moment its file exists and an
#: operation cannot be listed that nothing implements.
_ALLOWLISTS: dict[OperatorFamily, dict[str, Operation]] = {
    family: {
        module.OPERATION: module.run
        for module in MODULES.values()
        if module.FAMILY is family
    }
    for family in OperatorFamily
}


def allowlist(family: OperatorFamily | str) -> frozenset[str]:
    """The vetted operations for a family.  Used by the template's functional test tier."""
    return frozenset(_ALLOWLISTS[OperatorFamily(family)])


def readonly_scan(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return _dispatch(arguments, workspace, _ALLOWLISTS[OperatorFamily.READONLY_SCAN])


def content_extract(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return _dispatch(arguments, workspace, _ALLOWLISTS[OperatorFamily.CONTENT_EXTRACT])


def pure_transform(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return _dispatch(arguments, workspace, _ALLOWLISTS[OperatorFamily.PURE_TRANSFORM])


def record_store(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return _dispatch(arguments, workspace, _ALLOWLISTS[OperatorFamily.RECORD_STORE])


def filesystem_mutation(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return _dispatch(arguments, workspace, _ALLOWLISTS[OperatorFamily.FILESYSTEM_MUTATION])
