"""The optimizer metric.

Deliberately the same definitions the agent-ops SLIs use, so operational health and
prompt quality can never be measured differently: a prompt that scores well here is one
that produces runs that look healthy in `shakespeare metrics`.

No human labelling is involved. The obligations are deterministic checks, so the training
signal is a byproduct of the architecture rather than something that must be collected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: What the score is made of.  Composition validity dominates because an invalid
#: composition is refused outright and does no work at all.
METRIC_WEIGHTS: dict[str, float] = {
    "composition_valid": 0.4,
    "obligations_passed": 0.4,
    "first_attempt": 0.15,
    "cost": 0.05,
}


@dataclass(frozen=True)
class RunSignals:
    """What one training example records about a run or a stage attempt."""

    composition_valid: bool
    obligations_total: int
    obligations_passed: int
    attempts: int
    cost_usd: float = 0.0
    quarantined: int = 0
    items: int = 0

    @property
    def obligation_rate(self) -> float:
        return self.obligations_passed / self.obligations_total if self.obligations_total else 1.0

    @property
    def quarantine_rate(self) -> float:
        return self.quarantined / self.items if self.items else 0.0


def obligation_score(signals: RunSignals, *, cost_ceiling: float = 1.0) -> float:
    """Score in [0, 1].  Higher is better.

    An invalid composition scores zero outright: it was refused, so nothing else about it
    is worth crediting.
    """
    if not signals.composition_valid:
        return 0.0

    first_attempt = 1.0 if signals.attempts <= 1 else 1.0 / signals.attempts
    cost = max(0.0, 1.0 - (signals.cost_usd / cost_ceiling)) if cost_ceiling else 1.0

    score = (
        METRIC_WEIGHTS["composition_valid"] * 1.0
        + METRIC_WEIGHTS["obligations_passed"] * signals.obligation_rate
        + METRIC_WEIGHTS["first_attempt"] * first_attempt
        + METRIC_WEIGHTS["cost"] * cost
    )
    return round(min(1.0, max(0.0, score)), 6)


def signals_from_attempt(attempt: dict[str, Any]) -> RunSignals:
    """Build signals from one audited stage attempt."""
    obligations = attempt.get("obligations", [])
    nodes = attempt.get("nodes", [])
    return RunSignals(
        composition_valid=bool(nodes) and all(node.get("succeeded") for node in nodes),
        obligations_total=len(obligations),
        obligations_passed=sum(1 for item in obligations if item.get("passed")),
        attempts=int(attempt.get("attempt_no", 1)),
        cost_usd=float(attempt.get("cost_usd", 0.0)),
    )
