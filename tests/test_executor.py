"""Execution-time failures, and whether their detail is any use to the next round."""

from __future__ import annotations

from pathlib import Path

from shakespeare.contracts import BudgetEnvelope, Composition, DomainSpec, Invocation
from shakespeare.executor import Budget, Executor
from shakespeare.operators.builtin import build_registry
from shakespeare.verifier import Verifier


class TestBindingFailuresTeach:
    """The detail reaches the next round, so it should end the mistake rather than name it.

    A live run lost a whole goal to one misreading: the catalog lists what an operator
    produces, and the model mirrored that list into `bindings` as though outputs had to be
    declared. "no resolved source" was true and useless.
    """

    def _run(self, tmp_path: Path, invocation: Invocation):
        operators = build_registry()
        return Executor(operators, Verifier(operators)).execute(
            Composition(domain_id="survey", rationale="walk", invocations=(invocation,)),
            DomainSpec(id="survey", scope="inventory", catalog=frozenset({"fs.dirs"})),
            stage_inputs={"root": str(tmp_path)},
            config={},
            workspace=tmp_path / "work",
            budget=Budget(envelope=BudgetEnvelope(), items=0),
        )

    def test_binding_an_operators_own_output_says_so(self, tmp_path: Path) -> None:
        result = self._run(
            tmp_path,
            Invocation(
                invocation_id="dirs",
                operator="fs.dirs",
                inputs=("root",),
                bindings={"directories": "directories"},
            ),
        )
        assert not result[0].succeeded
        assert "output of fs.dirs" in (result[0].error_detail or "")

    def test_any_other_unresolved_binding_names_what_is_available(
        self, tmp_path: Path
    ) -> None:
        result = self._run(
            tmp_path,
            Invocation(
                invocation_id="dirs",
                operator="fs.dirs",
                inputs=("root",),
                bindings={"root": "nowhere"},
            ),
        )
        assert not result[0].succeeded
        assert "bindable here: root" in (result[0].error_detail or "")


class TestAMissingArgumentSaysWhatIsAvailable:
    """The value it needed was in front of it under another name, three attempts running.

    A live run lost the last goal of the workflow to this: `fs.dirs` wants `root`, the
    staged tree is in the context as `staging_root`, and the model wrote the binding
    backwards. "Field required" and "no resolved source" were both true and neither
    pointed anywhere.
    """

    def _run(self, tmp_path: Path, invocation: Invocation):
        operators = build_registry()
        return Executor(operators, Verifier(operators)).execute(
            Composition(domain_id="review", rationale="verify", invocations=(invocation,)),
            DomainSpec(id="review", scope="verify", catalog=frozenset({"fs.dirs"})),
            stage_inputs={"staging_root": str(tmp_path)},
            config={},
            workspace=tmp_path / "work",
            budget=Budget(envelope=BudgetEnvelope(), items=0),
        )

    def test_a_reversed_binding_is_named_as_reversed(self, tmp_path: Path) -> None:
        result = self._run(
            tmp_path,
            Invocation(
                invocation_id="d",
                operator="fs.dirs",
                inputs=("staging_root",),
                bindings={"staging_root": "root"},
            ),
        )
        detail = result[0].error_detail or ""
        assert "reversed" in detail
        assert "Write root=staging_root" in detail, "and the message contains the fix"

    def test_a_missing_argument_lists_what_could_have_filled_it(
        self, tmp_path: Path
    ) -> None:
        result = self._run(
            tmp_path,
            Invocation(invocation_id="d", operator="fs.dirs", inputs=("staging_root",)),
        )
        detail = result[0].error_detail or ""
        assert "root: Field required" in detail
        assert "bindable here: staging_root" in detail

    def test_the_binding_it_suggests_actually_works(self, tmp_path: Path) -> None:
        """A message that names a fix has to be right, or it is worse than silence."""
        result = self._run(
            tmp_path,
            Invocation(
                invocation_id="d",
                operator="fs.dirs",
                inputs=("staging_root",),
                bindings={"root": "staging_root"},
            ),
        )
        assert result[0].succeeded

    def test_runtime_plumbing_is_not_offered(self, tmp_path: Path) -> None:
        result = self._run(
            tmp_path,
            Invocation(invocation_id="d", operator="fs.dirs", inputs=("staging_root",)),
        )
        detail = result[0].error_detail or ""
        assert "config" not in detail and "operation" not in detail
