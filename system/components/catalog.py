"""Every operator there is, discovered from the files that define them.

An operator used to be declared in four places — a spec here, an argument model and its
produced keys in one module, its marshalling in another — and a piece could be forgotten.
Now each operator is one file in its family, and this walks those files. The registry, the
argument models, the produced keys and the family dispatch tables are all read from the
same modules, so they cannot disagree with each other or drift from what actually runs.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Any

from ..contracts import OperatorFamily, OperatorSpec
from .registry import FAMILY_RUNNERS, OperatorRegistry

#: Operators the runtime calls itself. They write, so no composition may name one.
RUNTIME_ONLY: frozenset[str] = frozenset({"fs.stage", "fs.commit", "fs.reverse", "fs.discard"})


def _operator_modules() -> list[ModuleType]:
    """Every module in a family package that declares an operator.

    A file is an operator when it says its own NAME. The family's logic modules do not,
    which is how `naming.py` and `name_render.py` sit side by side without ambiguity.
    """
    found: list[ModuleType] = []
    for family in OperatorFamily:
        package = importlib.import_module(f"{__package__}.{family.value}")
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{info.name}")
            if getattr(module, "NAME", None):
                found.append(module)
    return sorted(found, key=lambda module: module.NAME)


MODULES: dict[str, ModuleType] = {module.NAME: module for module in _operator_modules()}


def _spec(module: ModuleType) -> OperatorSpec:
    if module.OPERATION not in module.FEATURES:
        raise ValueError(f"{module.NAME} does not name its own operation in FEATURES")
    return OperatorSpec(
        name=module.NAME,
        version="1.0.0",
        description=module.SUMMARY,
        family=module.FAMILY,
        entrypoint=FAMILY_RUNNERS[module.FAMILY],
        features=frozenset(module.FEATURES),
        side_effects=tuple(module.SIDE_EFFECTS),
        risk=module.RISK,
        idempotent=module.IDEMPOTENT,
        timeout_seconds=module.TIMEOUT_SECONDS,
    )


#: name -> (spec, runner operation). The operation is what the family runner dispatches.
BUILTIN: dict[str, tuple[OperatorSpec, str]] = {
    name: (_spec(module), module.OPERATION) for name, module in MODULES.items()
}

#: name -> argument model, for the operators an agent may compose.
INPUT_MODELS: dict[str, Any] = {
    name: module.Input for name, module in MODULES.items() if module.Input is not None
}

#: name -> what it puts into the argument mapping for later invocations. Declaring inputs
#: without outputs left an agent able to call an operator but unable to wire one into the
#: next: it had to guess the key to bind from, and guessed wrong.
OUTPUT_KEYS: dict[str, list[str]] = {
    name: list(module.OUTPUTS) for name, module in MODULES.items() if module.OUTPUTS
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


def operation_of(name: str) -> str:
    """Which vetted operation an operator dispatches to. The executor passes it through."""
    return BUILTIN[name][1]


def build_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    for name, (spec, _) in BUILTIN.items():
        registry.register(spec, input_model=INPUT_MODELS.get(name))
    return registry
