"""Closed operator registry with trusted-runner entrypoint pinning.

A registered operator is a manifest, not code.  Behaviour always reaches the runtime
through its family's single trusted runner, so an operator that arrived by any other
route — including one a model asked for — cannot introduce a new call target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .contracts import OperatorFamily, OperatorSpec

#: The one entrypoint per family.  Registration rejects anything else, which is what makes
#: a declarative operator package unable to smuggle in executable behaviour.
FAMILY_RUNNERS: dict[OperatorFamily, str] = {
    OperatorFamily.READONLY_SCAN: "shakespeare.runners:readonly_scan",
    OperatorFamily.CONTENT_EXTRACT: "shakespeare.runners:content_extract",
    OperatorFamily.PURE_TRANSFORM: "shakespeare.runners:pure_transform",
    OperatorFamily.FILESYSTEM_MUTATION: "shakespeare.runners:filesystem_mutation",
}


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RegisteredOperator:
    spec: OperatorSpec
    input_model: type[BaseModel] | None = None
    output_model: type[BaseModel] | None = None

    def validate_input(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.input_model is None:
            return value
        return self.input_model.model_validate(value).model_dump(mode="json")

    def validate_output(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.output_model is None:
            return value
        return self.output_model.model_validate(value).model_dump(mode="json")


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, RegisteredOperator] = {}

    def register(
        self,
        spec: OperatorSpec,
        *,
        input_model: type[BaseModel] | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> None:
        if spec.name in self._operators:
            raise RegistryError(f"operator already registered: {spec.name}")

        expected = FAMILY_RUNNERS[spec.family]
        if spec.entrypoint != expected:
            raise RegistryError(
                f"{spec.name} does not use its family's trusted runner:"
                f" expected {expected}, got {spec.entrypoint}"
            )

        if spec.family is not OperatorFamily.FILESYSTEM_MUTATION and spec.side_effects:
            # Only the mutation family may declare write side effects; otherwise the
            # write-containment guarantee would depend on an author's discipline.
            writes = [item for item in spec.side_effects if item.startswith("write")]
            if writes:
                raise RegistryError(
                    f"{spec.name} declares write side effects but is not a"
                    f" filesystem_mutation operator: {writes}"
                )

        input_schema = input_model.model_json_schema() if input_model else {}
        output_schema = output_model.model_json_schema() if output_model else {}
        if spec.input_schema and spec.input_schema != input_schema:
            raise RegistryError(f"{spec.name}: declared input_schema does not match its model")
        if spec.output_schema and spec.output_schema != output_schema:
            raise RegistryError(f"{spec.name}: declared output_schema does not match its model")

        resolved = spec.model_copy(
            update={"input_schema": input_schema, "output_schema": output_schema}
        )
        self._operators[spec.name] = RegisteredOperator(resolved, input_model, output_model)

    def get(self, name: str) -> RegisteredOperator:
        try:
            return self._operators[name]
        except KeyError as exc:
            raise RegistryError(f"unknown operator: {name}") from exc

    def __contains__(self, name: object) -> bool:
        return name in self._operators

    def names(self) -> frozenset[str]:
        return frozenset(self._operators)

    def specs(self) -> tuple[OperatorSpec, ...]:
        return tuple(item.spec for item in self._operators.values())

    def family_of(self, name: str) -> OperatorFamily:
        return self.get(name).spec.family
