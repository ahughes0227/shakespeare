"""Bounds that were implemented but never tested.

Budget denial beyond operator calls, write containment at runtime rather than by
inspection, and the closed failure taxonomy the SLIs depend on.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from shakespeare.contracts import (
    BudgetEnvelope,
    Composition,
    DomainSpec,
    ErrorCode,
    Invocation,
    OperatorFamily,
)
from shakespeare.executor import Budget, Executor
from shakespeare.operators.builtin import BUILTIN, RUNTIME_ONLY, build_registry
from shakespeare.verifier import Denial, Verifier

DOMAIN = DomainSpec(
    id="probe",
    scope="exercise the bounds",
    catalog=frozenset({"fs.scan", "text.normalize"}),
    config_groups=frozenset({"extract"}),
)


def _composition(count: int = 1) -> Composition:
    return Composition(
        domain_id="probe",
        invocations=tuple(
            Invocation(
                invocation_id=f"i{index}",
                operator="text.normalize",
                parameters={"values": {"v": "x"}},
            )
            for index in range(count)
        ),
    )


class TestBudgetDenial:
    def test_operator_calls_are_capped(self) -> None:
        budget = Budget(envelope=BudgetEnvelope(operator_calls="2"), items=0)
        budget.consume_operator_call()
        budget.consume_operator_call()
        with pytest.raises(Denial) as caught:
            budget.consume_operator_call()
        assert caught.value.code is ErrorCode.BUDGET_EXHAUSTED

    def test_model_invocations_are_capped(self) -> None:
        budget = Budget(envelope=BudgetEnvelope(model_invocations="1"), items=0)
        budget.consume_model_invocation()
        with pytest.raises(Denial, match="model invocation budget"):
            budget.consume_model_invocation()

    def test_cost_is_capped(self) -> None:
        """A runaway prompt must stop costing money, not merely be noticed afterwards."""
        budget = Budget(
            envelope=BudgetEnvelope(model_invocations="10", max_cost_usd=0.05), items=0
        )
        budget.consume_model_invocation(cost_usd=0.04)
        with pytest.raises(Denial, match="cost budget"):
            budget.consume_model_invocation(cost_usd=0.04)

    def test_wall_time_is_capped(self) -> None:
        budget = Budget(envelope=BudgetEnvelope(wall_time_seconds=1), items=0)
        budget.check_wall_time()
        budget._started = time.monotonic() - 5
        with pytest.raises(Denial, match="wall-time budget"):
            budget.check_wall_time()

    def test_per_item_allowances_scale_with_the_input(self) -> None:
        """File counts are unbounded, so a fixed constant would strangle a large run."""
        envelope = BudgetEnvelope(operator_calls="10 + 3*n")
        assert Budget(envelope=envelope, items=0).operator_calls == 10
        assert Budget(envelope=envelope, items=100).operator_calls == 310

    def test_a_composition_larger_than_the_remaining_budget_is_refused(self) -> None:
        verifier = Verifier(build_registry())
        with pytest.raises(Denial) as caught:
            verifier.verify_composition(_composition(5), DOMAIN, operator_call_budget=2)
        assert caught.value.code is ErrorCode.BUDGET_EXHAUSTED

    def test_a_composition_that_cannot_fit_does_no_work_at_all(self, tmp_path: Path) -> None:
        """The budget is checked before execution, not exhausted partway through it.

        That matters: a composition refused up front leaves no half-finished work for the
        next attempt to reason about.
        """
        registry = build_registry()
        executor = Executor(registry, Verifier(registry))
        budget = Budget(envelope=BudgetEnvelope(operator_calls="10"), items=0)
        budget.usage = budget.usage.model_copy(update={"operator_calls": 9})

        with pytest.raises(Denial) as caught:
            executor.execute(
                _composition(3),
                DOMAIN,
                stage_inputs={},
                config={},
                workspace=tmp_path,
                budget=budget,
            )
        assert caught.value.code is ErrorCode.BUDGET_EXHAUSTED
        assert budget.usage.operator_calls == 9, "no invocation may have been charged"


class TestWriteContainment:
    def test_non_mutation_operators_perform_no_writes(self, tmp_path: Path) -> None:
        """Asserted at runtime, not by reading the source.

        Every write path in the codebase goes through pathlib or shutil, so denying both
        catches an operator that quietly grew one.
        """
        import shutil as shutil_module

        registry = build_registry()
        executor = Executor(registry, Verifier(registry))
        attempted: list[str] = []

        def forbid(name):
            def guard(*args, **kwargs):
                attempted.append(name)
                raise AssertionError(f"a non-mutation operator called {name}")

            return guard

        original = {
            "Path.write_text": Path.write_text,
            "Path.write_bytes": Path.write_bytes,
            "shutil.copy2": shutil_module.copy2,
            "shutil.move": shutil_module.move,
        }
        Path.write_text = forbid("Path.write_text")  # type: ignore[method-assign]
        Path.write_bytes = forbid("Path.write_bytes")  # type: ignore[method-assign]
        shutil_module.copy2 = forbid("shutil.copy2")  # type: ignore[assignment]
        shutil_module.move = forbid("shutil.move")  # type: ignore[assignment]
        try:
            source = tmp_path / "in"
            source.mkdir()
            executor.execute(
                Composition(
                    domain_id="probe",
                    invocations=(
                        Invocation(invocation_id="s", operator="fs.scan", inputs=("root",)),
                        Invocation(
                            invocation_id="n",
                            operator="text.normalize",
                            parameters={"values": {"v": " a  b "}},
                        ),
                    ),
                ),
                DOMAIN,
                stage_inputs={"root": str(source)},
                config={},
                workspace=tmp_path / "work",
                budget=Budget(envelope=BudgetEnvelope(), items=0),
            )
        finally:
            Path.write_text = original["Path.write_text"]  # type: ignore[method-assign]
            Path.write_bytes = original["Path.write_bytes"]  # type: ignore[method-assign]
            shutil_module.copy2 = original["shutil.copy2"]  # type: ignore[assignment]
            shutil_module.move = original["shutil.move"]  # type: ignore[assignment]

        assert attempted == []

    def test_no_stage_package_grants_a_mutation_operator(self) -> None:
        """Asserted across every registered stage, not only the ones a test happens to use."""
        from shakespeare.stages import StageRegistry

        registry = StageRegistry()
        for ref in registry.refs():
            for domain in registry.get(ref).domains:
                assert not (domain.catalog & RUNTIME_ONLY), f"{ref}.{domain.id}"

    def test_every_mutation_operator_declares_a_write(self) -> None:
        for spec, _ in BUILTIN.values():
            if spec.family is OperatorFamily.FILESYSTEM_MUTATION:
                assert any(item.startswith("write") for item in spec.side_effects), spec.name


class TestFailureTaxonomy:
    def test_every_denial_carries_a_closed_code(self) -> None:
        verifier = Verifier(build_registry())
        cases = [
            lambda: verifier.verify_composition(
                Composition(
                    domain_id="probe",
                    invocations=(Invocation(invocation_id="a", operator="fs.commit"),),
                ),
                DOMAIN,
            ),
            lambda: verifier.verify_composition(_composition(9), DOMAIN, operator_call_budget=1),
        ]
        for case in cases:
            with pytest.raises(Denial) as caught:
                case()
            assert isinstance(caught.value.code, ErrorCode)

    def test_operator_failures_are_recorded_with_a_code_not_free_text(
        self, tmp_path: Path
    ) -> None:
        registry = build_registry()
        executor = Executor(registry, Verifier(registry))
        results = executor.execute(
            Composition(
                domain_id="probe",
                invocations=(
                    Invocation(invocation_id="s", operator="fs.scan", inputs=("missing",)),
                ),
            ),
            DOMAIN,
            stage_inputs={},
            config={},
            workspace=tmp_path,
            budget=Budget(envelope=BudgetEnvelope(), items=0),
        )
        assert not results[0].succeeded
        assert isinstance(results[0].error_code, ErrorCode)
        assert results[0].journal_row()["error_code"] in {code.value for code in ErrorCode}

    def test_extraction_unavailability_uses_its_own_code(self) -> None:
        assert ErrorCode.EXTRACTION_UNAVAILABLE.value == "extraction_unavailable"
        assert len({code.value for code in ErrorCode}) == len(list(ErrorCode))
