"""Allowlisted Hydra composition.

A domain subagent selects `group=choice` and supplies bounded parameters.  It never
writes config *syntax*: interpolation, instantiation targets, and override operators are
rejected before Hydra sees them, so a selection cannot become code execution or a path
traversal.

Port of the guard in Lassie/services/research_operators.py:375.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

#: Hydra syntax a caller must never be able to inject.  `_target_` instantiates a Python
#: object; `${...}` interpolates; `~` and `+` are override operators that add or delete
#: keys outside the catalog.
FORBIDDEN_TOKENS: tuple[str, ...] = ("_target_", "${", "~", "+", "..")
_SAFE_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_CHOICE = re.compile(r"^[a-z][a-z0-9_]*$")

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


class CompositionError(ValueError):
    pass


@lru_cache(maxsize=1)
def catalog(config_root: str | None = None) -> dict[str, frozenset[str]]:
    """Enumerate the closed catalog from disk.

    Derived from the directory layout rather than hardcoded, so adding a choice is a
    file, and *only* a file that is actually present can be selected.
    """
    root = Path(config_root) if config_root else CONFIG_ROOT
    if not root.is_dir():
        raise CompositionError(f"config root does not exist: {root}")
    return {
        group.name: frozenset(item.stem for item in group.glob("*.yaml"))
        for group in sorted(root.iterdir())
        if group.is_dir()
    }


#: Rejected at every depth.  `_target_` would instantiate a Python object if the value
#: ever reached OmegaConf, `${` interpolates, and `..` is path traversal.
_ALWAYS_FORBIDDEN: tuple[str, ...] = ("_target_", "${", "..")
#: Rejected only in a top-level scalar, which is the shape a value would have if it were
#: being used as configuration.  `~` and `+` are Hydra override operators and matter only
#: in an override string; parameters never become override strings, and a vendor name may
#: legitimately contain either.
_SCALAR_FORBIDDEN: tuple[str, ...] = ("~", "+", "/", "\\")


#: Rejected as a key at any depth: these are the forms that could be interpreted as
#: instructions rather than data if a mapping ever reached OmegaConf.
_DIRECTIVE_KEYS: tuple[str, ...] = ("_target_", "_partial_", "_args_", "_convert_", "_recursive_")


def _reject_unsafe(key: str, value: Any, *, depth: int = 0) -> None:
    if depth == 0:
        # A top-level key is an argument name, so it must look like one.
        if key.startswith("_") or not _SAFE_KEY.match(key):
            raise CompositionError(f"unsafe parameter key: {key!r}")
    elif key in _DIRECTIVE_KEYS:
        # Deeper keys are data. An alias map is keyed by vendor names, a values mapping by
        # whatever the document says — arbitrary strings are correct there. Only Hydra's
        # own directive keys are refused.
        raise CompositionError(f"unsafe parameter key: {key!r}")
    if isinstance(value, str):
        for token in _ALWAYS_FORBIDDEN:
            if token in value:
                raise CompositionError(f"unsafe parameter value for {key}: contains {token!r}")
        if depth == 0:
            for token in _SCALAR_FORBIDDEN:
                if token in value:
                    raise CompositionError(
                        f"unsafe parameter value for {key}: contains {token!r}"
                    )
    elif isinstance(value, dict):
        # Nested payloads are data an operator consumes, not configuration. A relative
        # directory such as "2024/q1" is ordinary data here; the actual write is guarded
        # separately by mutation._guard, which resolves the path and checks containment.
        for inner_key, inner_value in value.items():
            _reject_unsafe(str(inner_key), inner_value, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_unsafe(key, item, depth=depth + 1)
    elif not isinstance(value, (int, float, bool, type(None))):
        raise CompositionError(f"unsupported parameter type for {key}: {type(value).__name__}")


def validate_selections(
    selections: dict[str, str], *, allowed_groups: frozenset[str] | None = None,
    config_root: str | None = None,
) -> None:
    known = catalog(config_root)
    for group, choice in selections.items():
        if group not in known:
            raise CompositionError(f"unknown config group: {group}")
        if allowed_groups is not None and group not in allowed_groups:
            raise CompositionError(f"config group not granted to this domain: {group}")
        if not _SAFE_CHOICE.match(choice):
            raise CompositionError(f"unsafe choice syntax: {group}={choice}")
        if choice not in known[group]:
            raise CompositionError(
                f"unknown choice {group}={choice}; catalog offers {sorted(known[group])}"
            )


def validate_parameters(parameters: dict[str, Any]) -> None:
    for key, value in parameters.items():
        _reject_unsafe(key, value)


def compose(
    selections: dict[str, str],
    parameters: dict[str, Any] | None = None,
    *,
    allowed_groups: frozenset[str] | None = None,
    config_root: str | None = None,
) -> dict[str, Any]:
    """Resolve a validated selection through Hydra into a plain mapping.

    Validation happens *before* Hydra sees anything.  Overrides are rebuilt from values
    that have already been checked against the catalog, so the override string Hydra
    parses can only ever be `group=choice` drawn from files that exist on disk.
    """
    parameters = parameters or {}
    validate_selections(selections, allowed_groups=allowed_groups, config_root=config_root)
    validate_parameters(parameters)

    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from omegaconf import OmegaConf

    root = Path(config_root) if config_root else CONFIG_ROOT
    overrides = [f"{group}={choice}" for group, choice in sorted(selections.items())]

    with initialize_config_dir(version_base="1.3", config_dir=str(root)):
        configured = hydra_compose(config_name="config", overrides=overrides)

    # resolve=True is safe only because interpolation was rejected during validation.
    container = OmegaConf.to_container(configured, resolve=True)
    if not isinstance(container, dict):
        raise CompositionError("Hydra composition did not produce a mapping")

    resolved: dict[str, Any] = {str(key): value for key, value in container.items()}
    resolved["parameters"] = parameters
    resolved["selections"] = dict(selections)
    return resolved
