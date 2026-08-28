"""The promotion gate.

Mirrors operator admission: a candidate is auto-promoted only when it clearly beats the
incumbent and regresses nothing.  Everything else goes to a person.

Without this gate, an optimizer that overfits a stale eval set would quietly become the
prompt every future run pins.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from ..contracts import (
    AdmissionChoice,
    DecidedBy,
    OptimizationRun,
    PromotionDecision,
    PromptArtifact,
)

#: How much better a candidate must be before it promotes itself.  Small enough to reward
#: a real improvement, large enough that eval-set noise does not.
DEFAULT_MARGIN = 0.02


class PromotionOutcome(StrEnum):
    AUTO_PROMOTE = "auto_promote"
    HUMAN_REVIEW = "human_review"
    REJECT = "reject"


@dataclass(frozen=True)
class PromotionGate:
    margin: float = DEFAULT_MARGIN

    def assess(
        self,
        run: OptimizationRun,
        *,
        incumbent: PromptArtifact | None = None,
        candidate: PromptArtifact | None = None,
    ) -> tuple[PromotionOutcome, str]:
        if run.fixture_regressions:
            return (
                PromotionOutcome.REJECT,
                f"regressed golden fixtures: {list(run.fixture_regressions)}",
            )

        if incumbent is None:
            return (
                PromotionOutcome.HUMAN_REVIEW,
                "no incumbent to compare against; a first prompt is a human decision",
            )

        if candidate is not None and _signature_changed(incumbent, candidate):
            # A changed signature means downstream contracts may no longer be satisfied,
            # which a score cannot tell you.
            return (
                PromotionOutcome.HUMAN_REVIEW,
                "the candidate changes the signature, not just the instructions",
            )

        if run.incumbent_score is None:
            return (PromotionOutcome.HUMAN_REVIEW, "the incumbent was never scored")

        delta = run.candidate_score - run.incumbent_score
        if delta <= 0:
            return (
                PromotionOutcome.REJECT,
                f"candidate does not beat the incumbent ({delta:+.4f})",
            )
        if delta < self.margin:
            return (
                PromotionOutcome.HUMAN_REVIEW,
                f"improvement {delta:+.4f} is within noise of the {self.margin} margin",
            )
        return (PromotionOutcome.AUTO_PROMOTE, f"beats the incumbent by {delta:+.4f}")

    def decide(
        self,
        run: OptimizationRun,
        *,
        incumbent: PromptArtifact | None = None,
        candidate: PromptArtifact | None = None,
    ) -> tuple[PromotionDecision, PromotionOutcome]:
        outcome, rationale = self.assess(run, incumbent=incumbent, candidate=candidate)
        decision = PromotionDecision(
            decision_id=uuid4().hex,
            optimization_id=run.optimization_id,
            decided_by=DecidedBy.AUTO
            if outcome is PromotionOutcome.AUTO_PROMOTE
            else DecidedBy.HUMAN,
            choice=AdmissionChoice.APPROVE
            if outcome is PromotionOutcome.AUTO_PROMOTE
            else AdmissionChoice.DENY,
            rationale=rationale,
        )
        return decision, outcome


def _signature_changed(incumbent: PromptArtifact, candidate: PromptArtifact) -> bool:
    return incumbent.signature_id != candidate.signature_id
