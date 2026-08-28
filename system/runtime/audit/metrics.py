"""Agent-ops SLIs.

These definitions are shared with the DSPy optimizer metric, so operational health and
prompt quality can never be measured differently.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from . import schema
from .store import AuditStore

SLI_NAMES: tuple[str, ...] = (
    "obligation_pass_rate",
    "attempts_per_stage",
    "composition_rejection_rate",
    "quarantine_rate",
    "admission_escalation_rate",
    "undo_rate",
    "cost_per_run",
    "stage_latency_ms",
)


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def obligation_pass_rate(store: AuditStore) -> dict[str, float]:
    with store.engine.begin() as connection:
        rows = connection.execute(
            select(
                schema.stage_attempts.c.stage_name,
                schema.obligation_results.c.passed,
                func.count().label("n"),
            )
            .select_from(
                schema.obligation_results.join(
                    schema.stage_attempts,
                    schema.obligation_results.c.attempt_id
                    == schema.stage_attempts.c.attempt_id,
                )
            )
            .group_by(schema.stage_attempts.c.stage_name, schema.obligation_results.c.passed)
        ).all()
    totals: dict[str, list[int]] = {}
    for stage, passed, count in rows:
        bucket = totals.setdefault(stage, [0, 0])
        bucket[0] += count if passed else 0
        bucket[1] += count
    return {stage: _ratio(good, total) for stage, (good, total) in totals.items()}


def attempts_per_stage(store: AuditStore) -> dict[str, float]:
    """Mean attempts per stage.  Above 1.0 means the planner is rerunning."""
    with store.engine.begin() as connection:
        rows = connection.execute(
            select(
                schema.stage_attempts.c.stage_name,
                func.count().label("attempts"),
                func.count(func.distinct(schema.stage_attempts.c.run_id)).label("runs"),
            ).group_by(schema.stage_attempts.c.stage_name)
        ).all()
    return {stage: _ratio(attempts, runs) for stage, attempts, runs in rows}


def undo_rate(store: AuditStore) -> float:
    with store.engine.begin() as connection:
        committed = connection.execute(select(func.count()).select_from(schema.commits)).scalar()
        reversed_ = connection.execute(
            select(func.count()).select_from(schema.reversals)
        ).scalar()
    return _ratio(float(reversed_ or 0), float(committed or 0))


def admission_escalation_rate(store: AuditStore) -> float:
    with store.engine.begin() as connection:
        total = connection.execute(
            select(func.count()).select_from(schema.admission_reports)
        ).scalar()
        escalated = connection.execute(
            select(func.count())
            .select_from(schema.admission_reports)
            .where(schema.admission_reports.c.disposition == "human_review")
        ).scalar()
    return _ratio(float(escalated or 0), float(total or 0))


def cost_per_run(store: AuditStore) -> float:
    with store.engine.begin() as connection:
        runs = connection.execute(select(func.count()).select_from(schema.runs)).scalar()
        cost = connection.execute(
            select(func.sum(schema.model_invocations.c.cost_usd))
        ).scalar()
    return _ratio(float(cost or 0.0), float(runs or 0))


def snapshot(store: AuditStore) -> dict[str, Any]:
    return {
        "obligation_pass_rate": obligation_pass_rate(store),
        "attempts_per_stage": attempts_per_stage(store),
        "undo_rate": undo_rate(store),
        "admission_escalation_rate": admission_escalation_rate(store),
        "cost_per_run": cost_per_run(store),
    }
