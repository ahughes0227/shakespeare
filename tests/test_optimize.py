"""Offline prompt optimization: the metric and, more importantly, the promotion gate.

The gate is what keeps a self-improving prompt safe. Without it an optimizer that overfits
a stale eval set would quietly become the prompt every future run pins.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from system.contracts import (
    AdmissionChoice,
    DecidedBy,
    OptimizationRun,
    PromptArtifact,
)
from system.prompt_store import PromptStore
from system.tuning import PromotionGate, PromotionOutcome, obligation_score
from system.tuning.metric import METRIC_WEIGHTS, RunSignals, signals_from_attempt
from system.tuning.promotion import DEFAULT_MARGIN


def signals(**overrides: object) -> RunSignals:
    base: dict[str, object] = {
        "composition_valid": True,
        "obligations_total": 4,
        "obligations_passed": 4,
        "attempts": 1,
        "cost_usd": 0.0,
    }
    base.update(overrides)
    return RunSignals(**base)  # type: ignore[arg-type]


class TestMetric:
    def test_weights_sum_to_one(self) -> None:
        assert sum(METRIC_WEIGHTS.values()) == pytest.approx(1.0)

    def test_a_perfect_run_scores_one(self) -> None:
        assert obligation_score(signals()) == pytest.approx(1.0)

    def test_an_invalid_composition_scores_zero(self) -> None:
        """A refused composition did no work, so nothing about it is worth crediting."""
        assert obligation_score(signals(composition_valid=False)) == 0.0

    def test_failed_obligations_reduce_the_score(self) -> None:
        assert obligation_score(signals(obligations_passed=2)) < obligation_score(signals())

    def test_reruns_reduce_the_score(self) -> None:
        """Converging first time is better than converging eventually."""
        assert obligation_score(signals(attempts=3)) < obligation_score(signals(attempts=1))

    def test_cost_reduces_the_score(self) -> None:
        assert obligation_score(signals(cost_usd=0.5)) < obligation_score(signals())

    def test_signals_read_from_an_audited_attempt(self) -> None:
        derived = signals_from_attempt(
            {
                "attempt_no": 2,
                "nodes": [{"succeeded": True}, {"succeeded": True}],
                "obligations": [{"passed": True}, {"passed": False}],
            }
        )
        assert derived.composition_valid
        assert derived.obligation_rate == pytest.approx(0.5)
        assert derived.attempts == 2

    def test_a_failed_invocation_makes_the_composition_invalid(self) -> None:
        derived = signals_from_attempt(
            {
                "attempt_no": 1,
                "nodes": [{"succeeded": True}, {"succeeded": False}],
                "obligations": [],
            }
        )
        assert not derived.composition_valid


def run(**overrides: object) -> OptimizationRun:
    base: dict[str, object] = {
        "optimization_id": "opt-1",
        "signature_id": "field_resolution",
        "optimizer": "BootstrapFewShot",
        "eval_set_digest": "d" * 64,
        "incumbent_version": "1.0.0",
        "incumbent_score": 0.80,
        "candidate_version": "1.1.0",
        "candidate_score": 0.90,
        "fixture_regressions": (),
    }
    base.update(overrides)
    return OptimizationRun(**base)  # type: ignore[arg-type]


class TestPromotionGate:
    def test_a_clear_improvement_auto_promotes(self) -> None:
        outcome, _ = PromotionGate().assess(run(), incumbent=_artifact("1.0.0"))
        assert outcome is PromotionOutcome.AUTO_PROMOTE

    def test_a_regression_is_rejected_outright(self) -> None:
        outcome, reason = PromotionGate().assess(run(candidate_score=0.5), incumbent=_artifact())
        assert outcome is PromotionOutcome.REJECT
        assert "does not beat" in reason

    def test_a_golden_fixture_regression_is_rejected_even_when_scoring_higher(self) -> None:
        """A higher average must never buy a regression on a case we know the answer to."""
        outcome, reason = PromotionGate().assess(
            run(candidate_score=0.99, fixture_regressions=("invoice_with_no_po",)),
            incumbent=_artifact(),
        )
        assert outcome is PromotionOutcome.REJECT
        assert "invoice_with_no_po" in reason

    def test_an_improvement_within_noise_goes_to_a_human(self) -> None:
        outcome, reason = PromotionGate().assess(
            run(candidate_score=0.80 + DEFAULT_MARGIN / 2), incumbent=_artifact()
        )
        assert outcome is PromotionOutcome.HUMAN_REVIEW
        assert "noise" in reason

    def test_a_first_prompt_is_a_human_decision(self) -> None:
        outcome, reason = PromotionGate().assess(run(incumbent_version=None), incumbent=None)
        assert outcome is PromotionOutcome.HUMAN_REVIEW
        assert "first prompt" in reason

    def test_a_changed_signature_goes_to_a_human(self) -> None:
        """A score cannot tell you whether downstream contracts still hold."""
        outcome, reason = PromotionGate().assess(
            run(),
            incumbent=_artifact(signature_id="field_resolution"),
            candidate=_artifact(signature_id="field_resolution_v2"),
        )
        assert outcome is PromotionOutcome.HUMAN_REVIEW
        assert "signature" in reason

    def test_an_unscored_incumbent_goes_to_a_human(self) -> None:
        outcome, _ = PromotionGate().assess(run(incumbent_score=None), incumbent=_artifact())
        assert outcome is PromotionOutcome.HUMAN_REVIEW

    def test_decision_records_who_decided(self) -> None:
        decision, outcome = PromotionGate().decide(run(), incumbent=_artifact())
        assert outcome is PromotionOutcome.AUTO_PROMOTE
        assert decision.decided_by is DecidedBy.AUTO
        assert decision.choice is AdmissionChoice.APPROVE

    def test_a_rejected_candidate_is_never_approved(self) -> None:
        decision, _ = PromotionGate().decide(run(candidate_score=0.1), incumbent=_artifact())
        assert decision.choice is AdmissionChoice.DENY


class TestVersioning:
    def test_a_compiled_artifact_takes_the_next_version_never_the_current_one(
        self, tmp_path: Path
    ) -> None:
        """Overwriting a pinned version would change a past run's behaviour."""
        from system.tuning.compile import Compiler

        store = PromptStore(tmp_path)
        store.save(_artifact("1.0.0"))
        compiler = Compiler(store=store)
        assert compiler.next_version("field_resolution") == "1.1.0"
        store.save(_artifact("1.1.0"))
        assert compiler.next_version("field_resolution") == "1.2.0"

    def test_first_version_when_nothing_exists(self, tmp_path: Path) -> None:
        from system.tuning.compile import Compiler

        assert Compiler(store=PromptStore(tmp_path)).next_version("new.signature") == "1.0.0"


class TestOptionalDependency:
    def test_the_runtime_imports_without_dspy(self) -> None:
        """DSPy is an optional extra; nothing on the run path may need it."""
        import system.runtime  # noqa: F401
        import system.services  # noqa: F401

    def test_compiling_without_dspy_explains_how_to_install_it(self) -> None:
        from system.tuning.compile import OptimizeError, require_dspy

        try:
            require_dspy()
        except OptimizeError as exc:
            assert "uv sync --extra optimize" in str(exc)


def _artifact(version: str = "1.0.0", signature_id: str = "field_resolution") -> PromptArtifact:
    return PromptArtifact(
        signature_id=signature_id, version=version, instructions="resolve the declared fields"
    )
