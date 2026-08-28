"""Execution-time failures, and whether their detail is any use to the next round."""

from __future__ import annotations

from pathlib import Path

from system.components.builtin import build_registry
from system.contracts import BudgetEnvelope, Composition, DomainSpec, Invocation
from system.runtime.executor import Budget, Executor
from system.runtime.verifier import Verifier


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


class TestADottedBindingNamesWhatWasProduced:
    """The mistake is usually the invocation id, so listing working values does not help.

    A live run guessed `collide.resolutions` at an invocation actually called
    `resolve_collisions`, and was told what was bindable without being told what it had
    just produced. It guessed again, twice, and the run lost its last goal.
    """

    def _run(self, tmp_path: Path, binding: dict[str, str]):
        operators = build_registry()
        return Executor(operators, Verifier(operators)).execute(
            Composition(
                domain_id="compose",
                rationale="plan",
                invocations=(
                    Invocation(
                        invocation_id="resolve_collisions",
                        operator="name.collide",
                        inputs=("candidates", "unrendered"),
                    ),
                    Invocation(
                        invocation_id="assemble",
                        operator="plan.assemble",
                        # Everything else resolves, so only the binding under test can
                        # be the one that fails.
                        inputs=(
                            "run_id", "workflow_id", "workflow_digest", "skipped",
                            "items", "digest",
                        ),
                        bindings={"scanned": "items", "decision_digest": "digest", **binding},
                    ),
                ),
            ),
            DomainSpec(
                id="compose",
                scope="plan",
                catalog=frozenset({"name.collide", "plan.assemble"}),
            ),
            stage_inputs={
                "candidates": [],
                "unrendered": [{"item_id": "a", "reason": "no_text"}],
                "items": [
                    {
                        "item_id": "a",
                        "relpath": "q/a.pdf",
                        "sha256": "0" * 64,
                        "size_bytes": 3,
                        "media_type": "application/pdf",
                    }
                ],
                "skipped": [],
                "run_id": "r",
                "workflow_id": "w",
                "workflow_digest": "d",
                "digest": "s",
            },
            config={},
            workspace=tmp_path / "work",
            budget=Budget(envelope=BudgetEnvelope(operator_calls="9"), items=1),
        )

    def test_a_wrong_invocation_id_is_answered_with_the_right_one(
        self, tmp_path: Path
    ) -> None:
        result = self._run(tmp_path, {"planned": "collide.resolutions"})
        detail = result[1].error_detail or ""
        assert "names no earlier invocation" in detail
        assert "resolve_collisions.resolutions" in detail

    def test_the_binding_it_offers_actually_works(self, tmp_path: Path) -> None:
        """Including on the all-quarantine set that produced the live failure."""
        result = self._run(tmp_path, {"planned": "resolve_collisions.resolutions"})
        assert all(item.succeeded for item in result), [
            item.error_detail for item in result if not item.succeeded
        ]

    def test_it_does_not_offer_working_values_for_a_dotted_source(
        self, tmp_path: Path
    ) -> None:
        """They were never the answer, and offering them is what sent it round again."""
        detail = self._run(tmp_path, {"planned": "collide.resolutions"})[1].error_detail or ""
        assert "workflow_digest" not in detail


class TestAMissingArgumentNamesEverySourceThereIs:
    """An invocation binds only from what its own `inputs` reference, so listing those is
    not enough when the mistake is not having referenced the right thing.

    A live run needed `decision_digest` three attempts running. The value that fills it was
    in the working set the whole time under `digest`, and every message it received listed
    everything except that.
    """

    def _run(self, tmp_path: Path):
        operators = build_registry()
        items = [
            {
                "item_id": "a",
                "relpath": "q/a.pdf",
                "sha256": "0" * 64,
                "size_bytes": 3,
                "media_type": "application/pdf",
            }
        ]
        return Executor(operators, Verifier(operators)).execute(
            Composition(
                domain_id="compose",
                rationale="plan",
                invocations=(
                    Invocation(
                        invocation_id="resolve_collisions",
                        operator="name.collide",
                        inputs=("candidates", "unrendered"),
                    ),
                    Invocation(
                        invocation_id="assemble",
                        operator="plan.assemble",
                        inputs=("run_id", "workflow_id", "workflow_digest", "items", "skipped"),
                        bindings={
                            "planned": "resolve_collisions.resolutions",
                            "scanned": "items",
                        },
                    ),
                ),
            ),
            DomainSpec(
                id="compose",
                scope="plan",
                catalog=frozenset({"name.collide", "plan.assemble"}),
            ),
            stage_inputs={
                "candidates": [],
                "unrendered": [{"item_id": "a", "reason": "no_text"}],
                "items": items,
                "skipped": [],
                "run_id": "r",
                "workflow_id": "w",
                "workflow_digest": "d",
                "digest": "s",
            },
            config={},
            workspace=tmp_path / "work",
            budget=Budget(envelope=BudgetEnvelope(operator_calls="9"), items=1),
        )

    def test_it_names_a_value_the_invocation_never_referenced(self, tmp_path: Path) -> None:
        detail = self._run(tmp_path)[1].error_detail or ""
        assert "decision_digest: Field required" in detail
        assert "referenceable via inputs: candidates, digest" in detail

    def test_it_names_what_earlier_invocations_produced(self, tmp_path: Path) -> None:
        detail = self._run(tmp_path)[1].error_detail or ""
        assert "resolve_collisions.resolutions" in detail

    def test_it_does_not_repeat_what_is_already_bound(self, tmp_path: Path) -> None:
        """Two lists that overlap read as one long list, and get skimmed."""
        detail = self._run(tmp_path)[1].error_detail or ""
        referenceable = detail.split("referenceable via inputs: ")[1].split(";")[0]
        assert "run_id" not in referenceable and "items" not in referenceable
