"""Gate evaluation.

A gate asks whether the artifacts now available sufficiently satisfy a goal (§5).

The order matters and is deliberate: required evidence first, then deterministic checks,
then judgment — and judgment only if the first two hold. Never ask a model something a
check can settle (§11), and never ask it to judge evidence that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .artifacts import Artifact, ArtifactStore
from .domain.planning import CHECK_REQUIREMENTS, run_check
from .goals import Gate, GateKind, GateOutcome, GateResult, Goal, evidence_for


class GateJudge(Protocol):
    """The semantic half. Asked only when a deterministic answer is impossible."""

    def judge(
        self,
        *,
        goal: Goal,
        rubric: str,
        artifacts: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> tuple[bool, str]: ...


@dataclass
class GateEvaluator:
    artifacts: ArtifactStore
    judge: GateJudge | None = None

    def evaluate(
        self, goal: Goal, context: dict[str, Any], *, exhausted: bool = False
    ) -> GateResult:
        """`exhausted` says the capability stopped because it ran out of rounds.

        A gate judges evidence, not process — but "it stopped because it was finished"
        and "it stopped because it ran out" are different situations, and a judge that
        cannot tell them apart will call a partial answer sufficient.
        """
        gate = goal.gate
        available: tuple[Artifact, ...] = self.artifacts.all()

        missing = evidence_for(gate, available)
        if missing:
            # Blocked, not insufficient: the question cannot even be asked yet.
            return GateResult(
                gate_id=gate.id,
                goal_id=goal.id,
                outcome=GateOutcome.BLOCKED,
                missing_kinds=missing,
                rationale=f"required evidence is absent: {', '.join(missing)}",
            )

        failed = self._failed_checks(gate, context)
        if failed:
            return GateResult(
                gate_id=gate.id,
                goal_id=goal.id,
                outcome=GateOutcome.INSUFFICIENT,
                failed_checks=failed,
                rationale=f"deterministic checks failed: {', '.join(failed)}",
            )

        if gate.kind is GateKind.DETERMINISTIC:
            return GateResult(
                gate_id=gate.id,
                goal_id=goal.id,
                outcome=GateOutcome.SATISFIED,
                rationale="required evidence present and every check passed",
            )

        if self.judge is None:
            # No judge configured. Deterministic evidence held, so the honest answer is
            # to pass rather than to block on a judgment nobody can make.
            return GateResult(
                gate_id=gate.id,
                goal_id=goal.id,
                outcome=GateOutcome.SATISFIED,
                rationale="deterministic evidence held; no judge configured",
            )

        satisfied, rationale = self.judge.judge(
            goal=goal,
            rubric=gate.rubric,
            artifacts=self.artifacts.describe(),
            evidence={
                **_evidence_for_judgment(gate, context),
                "capability_exhausted": exhausted,
            },
        )
        return GateResult(
            gate_id=gate.id,
            goal_id=goal.id,
            outcome=GateOutcome.SATISFIED if satisfied else GateOutcome.INSUFFICIENT,
            rationale=rationale,
        )

    def _failed_checks(self, gate: Gate, context: dict[str, Any]) -> tuple[str, ...]:
        failed: list[str] = []
        for check in gate.checks:
            payload = _payload_for(check, context)
            if payload is None:
                # Evidence for the check was never produced. An unevaluated check is not
                # a passed one.
                failed.append(check)
                continue
            if not run_check(check, check, payload).passed:
                failed.append(check)
        return tuple(failed)


def _payload_for(check: str, context: dict[str, Any]) -> dict[str, Any] | None:
    required = CHECK_REQUIREMENTS.get(check, ())
    payload = {key: context[key] for key in required if key in context}
    return payload if len(payload) == len(required) else None


def _evidence_for_judgment(gate: Gate, context: dict[str, Any]) -> dict[str, Any]:
    """Counts and coverage only. A judge weighs sufficiency, it does not read documents."""
    summary: dict[str, Any] = {}
    for key, value in context.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list):
            summary[key] = len(value)
    return summary
