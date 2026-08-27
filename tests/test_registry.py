from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError
from shakespeare.components.registry import FAMILY_RUNNERS, OperatorRegistry, RegistryError
from shakespeare.contracts import OperatorFamily, OperatorSpec


def spec(name: str, family: OperatorFamily = OperatorFamily.PURE_TRANSFORM, **kwargs: object):
    return OperatorSpec(
        name=name,
        version="1.0.0",
        description="test operator",
        family=family,
        entrypoint=FAMILY_RUNNERS[family],
        **kwargs,
    )


class Input(BaseModel):
    value: int


class Output(BaseModel):
    result: int


class TestEntrypointPinning:
    def test_every_family_has_a_trusted_runner(self) -> None:
        assert set(FAMILY_RUNNERS) == set(OperatorFamily)

    def test_rejects_entrypoint_that_is_not_the_family_runner(self) -> None:
        registry = OperatorRegistry()
        rogue = spec("rogue").model_copy(update={"entrypoint": "shakespeare.evil:run"})
        with pytest.raises(RegistryError, match="trusted runner"):
            registry.register(rogue)

    def test_rejects_another_familys_runner(self) -> None:
        registry = OperatorRegistry()
        crossed = spec("crossed").model_copy(
            update={"entrypoint": FAMILY_RUNNERS[OperatorFamily.FILESYSTEM_MUTATION]}
        )
        with pytest.raises(RegistryError, match="trusted runner"):
            registry.register(crossed)


class TestWriteContainment:
    def test_non_mutation_family_may_not_declare_writes(self) -> None:
        registry = OperatorRegistry()
        sneaky = spec("sneaky").model_copy(update={"side_effects": ("write:/tmp/x",)})
        with pytest.raises(RegistryError, match="write side effects"):
            registry.register(sneaky)

    def test_mutation_family_may_declare_writes(self) -> None:
        registry = OperatorRegistry()
        registry.register(
            spec(
                "fs.commit",
                family=OperatorFamily.FILESYSTEM_MUTATION,
                side_effects=("write:output_root",),
            )
        )
        assert "fs.commit" in registry


class TestRegistration:
    def test_rejects_duplicate_name(self) -> None:
        registry = OperatorRegistry()
        registry.register(spec("dupe"))
        with pytest.raises(RegistryError, match="already registered"):
            registry.register(spec("dupe"))

    def test_resolves_schemas_from_models(self) -> None:
        registry = OperatorRegistry()
        registry.register(spec("typed"), input_model=Input, output_model=Output)
        resolved = registry.get("typed").spec
        assert resolved.input_schema == Input.model_json_schema()
        assert resolved.output_schema == Output.model_json_schema()

    def test_rejects_declared_schema_that_does_not_match_its_model(self) -> None:
        registry = OperatorRegistry()
        mismatched = spec("mismatch").model_copy(
            update={"input_schema": {"type": "object", "properties": {}}}
        )
        with pytest.raises(RegistryError, match="does not match its model"):
            registry.register(mismatched, input_model=Input)

    def test_unknown_operator_raises(self) -> None:
        with pytest.raises(RegistryError, match="unknown operator"):
            OperatorRegistry().get("nope")

    def test_validates_input_and_output(self) -> None:
        registry = OperatorRegistry()
        registry.register(spec("typed"), input_model=Input, output_model=Output)
        registered = registry.get("typed")
        assert registered.validate_input({"value": 3}) == {"value": 3}
        with pytest.raises(ValidationError):
            registered.validate_input({"value": "not an int", "extra": 1})
