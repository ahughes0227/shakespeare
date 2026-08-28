"""Write and query the append-only audit log."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Engine, create_engine, event, insert, select
from sqlalchemy.engine import Connection

from ...contracts import (
    AdmissionDecision,
    AdmissionReport,
    ChangeAction,
    ChangePlan,
    Composition,
    Measurement,
    MeasurementKind,
    ObligationResult,
    OperatorRequest,
    OperatorSpec,
    OptimizationRun,
    PromotionDecision,
    PromptArtifact,
    ReversalRecord,
    StagePlan,
    StageVerdict,
    canonical_json,
    utc_now,
)
from . import schema


def _now() -> str:
    return utc_now().isoformat()


def _id() -> str:
    return uuid4().hex


class AuditStore:
    """One SQLite file per workspace.  Facts only; never working state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.engine: Engine = create_engine(f"sqlite:///{path}", future=True)

        @event.listens_for(self.engine, "connect")
        def _pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        with self.engine.begin() as connection:
            schema.metadata.create_all(connection)
            schema.install_append_only_triggers(connection)

    def close(self) -> None:
        self.engine.dispose()

    def _insert(self, connection: Connection, table: Any, **values: Any) -> None:
        connection.execute(insert(table).values(**values))

    # -- runs ---------------------------------------------------------------------------

    def record_run(
        self,
        *,
        run_id: str,
        workflow_id: str,
        workflow_version: str,
        workflow_digest: str,
        request_digest: str,
        input_root_digest: str,
    ) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.runs,
                run_id=run_id,
                created_at=_now(),
                workflow_id=workflow_id,
                workflow_version=workflow_version,
                workflow_digest=workflow_digest,
                request_digest=request_digest,
                input_root_digest=input_root_digest,
            )

    def record_run_outcome(
        self, *, run_id: str, outcome: str, error_code: str | None = None
    ) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.run_outcomes,
                run_id=run_id,
                ended_at=_now(),
                outcome=outcome,
                error_code=error_code,
            )

    # -- stages -------------------------------------------------------------------------

    def record_attempt(
        self,
        *,
        run_id: str,
        stage_name: str,
        stage_version: str,
        attempt_no: int,
        started_at: str,
        plan: StagePlan,
        compositions: Sequence[tuple[Composition, Sequence[dict[str, Any]]]],
        obligations: Sequence[ObligationResult],
        verdict: StageVerdict,
        error_code: str | None = None,
    ) -> str:
        """Write one complete attempt.

        Attempts are recorded individually, including failed ones, so `replay` can
        reproduce the exact sequence rather than an idealised one.
        """
        attempt_id = _id()
        plan_id = _id()
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.stage_attempts,
                attempt_id=attempt_id,
                run_id=run_id,
                stage_name=stage_name,
                stage_version=stage_version,
                attempt_no=attempt_no,
                started_at=started_at,
                ended_at=_now(),
                error_code=error_code,
            )
            self._insert(
                connection,
                schema.stage_plans,
                plan_id=plan_id,
                attempt_id=attempt_id,
                digest=plan.digest(),
                payload=canonical_json(plan),
            )
            for goal in plan.activated:
                self._insert(
                    connection,
                    schema.domain_goals,
                    goal_id=_id(),
                    plan_id=plan_id,
                    domain_id=goal.domain_id,
                    activated=True,
                    goal=goal.goal,
                    success_criterion=goal.success_criterion,
                    skip_reason=None,
                )
            for skip in plan.skipped:
                self._insert(
                    connection,
                    schema.domain_goals,
                    goal_id=_id(),
                    plan_id=plan_id,
                    domain_id=skip.domain_id,
                    activated=False,
                    goal=None,
                    success_criterion=None,
                    skip_reason=skip.reason,
                )

            for composition, results in compositions:
                composition_id = _id()
                self._insert(
                    connection,
                    schema.compositions,
                    composition_id=composition_id,
                    attempt_id=attempt_id,
                    domain_id=composition.domain_id,
                    digest=composition.digest(),
                    payload=canonical_json(composition),
                )
                by_id = {item["invocation_id"]: item for item in results}
                for seq, invocation in enumerate(composition.invocations):
                    outcome = by_id.get(invocation.invocation_id, {})
                    self._insert(
                        connection,
                        schema.invocations,
                        invocation_id=f"{composition_id}:{invocation.invocation_id}",
                        composition_id=composition_id,
                        seq=seq,
                        operator=invocation.operator,
                        operator_version=outcome.get("operator_version", "unknown"),
                        selections=canonical_json(invocation.selections),
                        parameters=canonical_json(invocation.parameters),
                        started_at=outcome.get("started_at", started_at),
                        ended_at=outcome.get("ended_at", _now()),
                        succeeded=bool(outcome.get("succeeded", False)),
                        output_digest=outcome.get("output_digest"),
                        error_code=outcome.get("error_code")
                        or (None if invocation.invocation_id in by_id else "not_reached"),
                    )
                for source, target in composition.edges():
                    self._insert(
                        connection,
                        schema.invocation_edges,
                        composition_id=composition_id,
                        from_invocation=f"{composition_id}:{source}",
                        to_invocation=f"{composition_id}:{target}",
                    )

            for result in obligations:
                self._insert(
                    connection,
                    schema.obligation_results,
                    result_id=_id(),
                    attempt_id=attempt_id,
                    obligation_id=result.obligation_id,
                    passed=result.passed,
                    detail=canonical_json(result.detail),
                )

            self._insert(
                connection,
                schema.stage_verdicts,
                verdict_id=_id(),
                attempt_id=attempt_id,
                met=verdict.met,
                decision=str(verdict.decision),
                unmet=canonical_json(list(verdict.unmet)),
                rationale=verdict.rationale,
            )
        return attempt_id

    # -- model --------------------------------------------------------------------------

    def record_model_invocation(
        self,
        *,
        run_id: str,
        role: str,
        profile_id: str,
        requested_model: str,
        resolved_model: str | None = None,
        provider: str | None = None,
        prompt_version: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
        error_code: str | None = None,
    ) -> str:
        invocation_id = _id()
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.model_invocations,
                invocation_id=invocation_id,
                run_id=run_id,
                role=role,
                profile_id=profile_id,
                requested_model=requested_model,
                resolved_model=resolved_model,
                provider=provider,
                prompt_version=prompt_version,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                error_code=error_code,
                recorded_at=_now(),
            )
        return invocation_id

    # -- operator admission -------------------------------------------------------------

    def record_operator_request(self, request: OperatorRequest) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.operator_requests,
                request_id=request.request_id,
                run_id=request.run_id,
                domain_id=request.domain_id,
                kind=str(request.kind),
                family=str(request.family),
                name=request.name,
                payload=canonical_json(request),
                recorded_at=_now(),
            )

    def record_capability_gap(
        self, *, run_id: str, prompt_digest: str, rationale: str, requires: tuple[str, ...]
    ) -> None:
        """A request nothing registered could serve, and what the router judged it needs."""
        from uuid import uuid4

        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.capability_gaps,
                gap_id=uuid4().hex,
                run_id=run_id,
                prompt_digest=prompt_digest,
                rationale=rationale,
                requires=canonical_json(list(requires)),
                recorded_at=_now(),
            )

    def capability_gaps(self) -> list[dict[str, Any]]:
        """Every unmet request, newest first, with how often each has been asked."""
        import json

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(schema.capability_gaps).order_by(
                    schema.capability_gaps.c.recorded_at.desc()
                )
            ).mappings()
            gaps = [dict(row) for row in rows]
        # One row per unmet request, not per occurrence. A backlog is read to decide what
        # to build next, and the same request listed three times is three things to skim
        # past rather than one thing wanted three times. The newest analysis wins, since
        # the catalog it was judged against is the one that exists now.
        grouped: dict[str, dict[str, Any]] = {}
        for gap in gaps:
            digest = gap["prompt_digest"]
            if digest not in grouped:
                grouped[digest] = {**gap, "requires": json.loads(gap["requires"]), "asked": 0}
            grouped[digest]["asked"] += 1
        return list(grouped.values())

    def record_admission_report(self, report: AdmissionReport, *, request_id: str,
                                package_digest: str) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.admission_reports,
                report_id=report.report_id,
                request_id=request_id,
                candidate_id=report.candidate_id,
                computed_risk=str(report.computed_risk),
                disposition=str(report.disposition),
                package_digest=package_digest,
                reproducible=report.reproducible,
                findings=canonical_json([f.model_dump(mode="json") for f in report.findings]),
                test_results=canonical_json(report.test_results),
            )

    def record_admission_decision(self, decision: AdmissionDecision) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.admission_decisions,
                decision_id=decision.decision_id,
                report_id=decision.report_id,
                decided_by=str(decision.decided_by),
                choice=str(decision.choice),
                rationale=decision.rationale,
                decided_at=_now(),
            )

    def record_operator_registration(self, spec: OperatorSpec) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.operator_registrations,
                registration_id=_id(),
                name=spec.name,
                version=spec.version,
                family=str(spec.family),
                package_digest=spec.package_digest,
                admission_id=spec.admission_id,
                registered_at=_now(),
            )

    # -- prompts ------------------------------------------------------------------------

    def record_prompt_artifact(self, artifact: PromptArtifact) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.prompt_artifacts,
                signature_id=artifact.signature_id,
                version=artifact.version,
                digest=artifact.digest(),
                payload=canonical_json(artifact),
                recorded_at=_now(),
            )

    def record_optimization(self, run: OptimizationRun) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.optimization_runs,
                optimization_id=run.optimization_id,
                signature_id=run.signature_id,
                optimizer=run.optimizer,
                eval_set_digest=run.eval_set_digest,
                incumbent_version=run.incumbent_version,
                incumbent_score=run.incumbent_score,
                candidate_version=run.candidate_version,
                candidate_score=run.candidate_score,
                fixture_regressions=canonical_json(list(run.fixture_regressions)),
                recorded_at=_now(),
            )

    def record_promotion(self, decision: PromotionDecision) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.promotion_decisions,
                decision_id=decision.decision_id,
                optimization_id=decision.optimization_id,
                decided_by=str(decision.decided_by),
                choice=str(decision.choice),
                rationale=decision.rationale,
                decided_at=_now(),
            )

    # -- measurements -------------------------------------------------------------------

    def record_measurements(self, *, run_id: str, measurements: Sequence[Measurement]) -> int:
        """Record what a run observed. Returns how many rows were written.

        Called once, after the run has finished doing the thing it measured. Nothing
        reads these rows during a run: they exist so a declared constant can later be
        replaced by a measured one through an explicit promotion, which is what keeps a
        run's behaviour fixed at its start and keeps `replay` meaningful.
        """
        if not measurements:
            return 0
        with self.engine.begin() as connection:
            for item in measurements:
                self._insert(
                    connection,
                    schema.measurements,
                    measurement_id=_id(),
                    run_id=run_id,
                    kind=str(item.kind),
                    subject=item.subject,
                    resolved_model=item.resolved_model,
                    prompt_version=item.prompt_version,
                    value=item.value,
                    weight=item.weight,
                    count=item.count,
                    outcome=item.outcome,
                    bound=str(item.bound) if item.bound else None,
                    recorded_at=_now(),
                )
        return len(measurements)

    def measurements(
        self,
        *,
        kind: MeasurementKind | None = None,
        subject: str | None = None,
        resolved_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recorded observations, newest last, with the run each came from."""
        query = select(schema.measurements).order_by(schema.measurements.c.recorded_at)
        if kind is not None:
            query = query.where(schema.measurements.c.kind == str(kind))
        if subject is not None:
            query = query.where(schema.measurements.c.subject == subject)
        if resolved_model is not None:
            query = query.where(schema.measurements.c.resolved_model == resolved_model)
        with self.engine.begin() as connection:
            return [dict(row) for row in connection.execute(query).mappings().all()]

    def measured_subjects(self, kind: MeasurementKind) -> list[dict[str, Any]]:
        """What has been measured of a given kind, and how much of it there is."""
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.measurements(kind=kind):
            key = (row["subject"], row["resolved_model"])
            entry = grouped.setdefault(
                key,
                {
                    "subject": row["subject"],
                    "resolved_model": row["resolved_model"],
                    "observations": 0,
                    "runs": set(),
                    "latest": "",
                },
            )
            entry["observations"] += 1
            entry["runs"].add(row["run_id"])
            entry["latest"] = max(entry["latest"], row["recorded_at"])
        return [
            {**entry, "runs": len(entry["runs"])}
            for entry in sorted(grouped.values(), key=lambda item: item["subject"])
        ]

    def resolved_models(self, run_id: str) -> list[str]:
        """Which models actually answered in a run, busiest first.

        A resolved model rather than a requested one: the point of recording it is that a
        provider can change what an alias means, which is the drift `canary` exists for.
        """
        query = select(schema.model_invocations.c.resolved_model).where(
            schema.model_invocations.c.run_id == run_id
        )
        with self.engine.begin() as connection:
            rows = [row[0] for row in connection.execute(query) if row[0]]
        counted: dict[str, int] = {}
        for name in rows:
            counted[name] = counted.get(name, 0) + 1
        return [name for name, _ in sorted(counted.items(), key=lambda item: -item[1])]

    def run_costs(self, run_ids: Sequence[str]) -> dict[str, float]:
        """Total spend per run, for the runs asked about.

        Whole-run cost, not per-goal: a run's spend cannot honestly be attributed to one
        goal's choice, and pretending otherwise would make a shape look expensive because
        something else in the run was.
        """
        if not run_ids:
            return {}
        query = select(
            schema.model_invocations.c.run_id, schema.model_invocations.c.cost_usd
        ).where(schema.model_invocations.c.run_id.in_(list(run_ids)))
        with self.engine.begin() as connection:
            rows = connection.execute(query).mappings().all()
        totals: dict[str, float] = {}
        for row in rows:
            totals[row["run_id"]] = totals.get(row["run_id"], 0.0) + (row["cost_usd"] or 0.0)
        return totals

    def run_endings(self, run_ids: Sequence[str]) -> dict[str, str]:
        """How each run ended. A run with no outcome row never finished."""
        if not run_ids:
            return {}
        query = select(schema.run_outcomes.c.run_id, schema.run_outcomes.c.outcome).where(
            schema.run_outcomes.c.run_id.in_(list(run_ids))
        )
        with self.engine.begin() as connection:
            return {row["run_id"]: row["outcome"] for row in connection.execute(query).mappings()}

    def attempts_by_goal(self) -> dict[str, list[tuple[int, bool]]]:
        """Every recorded attempt as (attempt number, whether its gate was met).

        Needs no measurement of its own: whether a goal that failed once ever recovers is
        already a fact of the log, which is why `max_goal_attempts` can be checked against
        evidence without recording anything new.
        """
        query = (
            select(
                schema.stage_attempts.c.stage_name,
                schema.stage_attempts.c.attempt_no,
                schema.stage_verdicts.c.met,
            )
            .select_from(
                schema.stage_attempts.join(
                    schema.stage_verdicts,
                    schema.stage_attempts.c.attempt_id == schema.stage_verdicts.c.attempt_id,
                )
            )
            .order_by(schema.stage_attempts.c.stage_name, schema.stage_attempts.c.attempt_no)
        )
        with self.engine.begin() as connection:
            rows = connection.execute(query).mappings().all()
        grouped: dict[str, list[tuple[int, bool]]] = {}
        for row in rows:
            grouped.setdefault(row["stage_name"], []).append(
                (int(row["attempt_no"]), bool(row["met"]))
            )
        return grouped

    # -- mutation -----------------------------------------------------------------------

    def record_mutation(
        self,
        *,
        run_id: str,
        target_ref: str,
        operation: str,
        reversal: ReversalRecord,
        invocation_id: str | None = None,
        before_digest: str | None = None,
        after_digest: str | None = None,
    ) -> str:
        mutation_id = reversal.mutation_id
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.mutations,
                mutation_id=mutation_id,
                run_id=run_id,
                invocation_id=invocation_id,
                target_ref=target_ref,
                operation=operation,
                before_digest=before_digest,
                after_digest=after_digest,
                reversal=canonical_json(reversal),
                recorded_at=_now(),
            )
        return mutation_id

    def record_plan(self, *, run_id: str, plan: ChangePlan) -> str:
        """Record the plan a run produced, whether or not it was committed.

        Replay needs something to compare against; without this the log could say a run
        committed three files but not what it decided they should be called.
        """
        plan_id = _id()
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.plans,
                plan_id=plan_id,
                run_id=run_id,
                digest=plan.digest(),
                fingerprint=plan.fingerprint(),
                entry_count=len(plan.entries),
                changed=plan.count(ChangeAction.CHANGED),
                unchanged=plan.count(ChangeAction.UNCHANGED),
                unresolved=plan.count(ChangeAction.UNRESOLVED),
                payload=canonical_json(plan),
                recorded_at=_now(),
            )
        return plan_id

    def recorded_plan(self, run_id: str) -> ChangePlan | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(schema.plans).where(schema.plans.c.run_id == run_id)
            ).mappings().first()
        return ChangePlan.model_validate(json.loads(row["payload"])) if row else None

    def find_commit(self, *, plan_digest: str, output_root: str) -> dict[str, Any] | None:
        """`plan_digest` is a `ChangePlan.fingerprint()`, not a `digest()`."""
        """A commit already made for this exact plan and destination.

        This is the idempotency receipt: re-applying a satisfied plan should be a no-op,
        not an error about the output root existing.
        """
        with self.engine.begin() as connection:
            row = connection.execute(
                select(schema.commits, schema.plans)
                .select_from(
                    schema.commits.join(
                        schema.plans, schema.commits.c.run_id == schema.plans.c.run_id
                    )
                )
                .where(schema.plans.c.fingerprint == plan_digest)
                .where(schema.commits.c.output_root == output_root)
            ).mappings().first()
        return dict(row) if row else None

    def record_commit(self, *, run_id: str, plan: ChangePlan, staging_digest: str,
                      output_root: str) -> str:
        commit_id = _id()
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.commits,
                commit_id=commit_id,
                run_id=run_id,
                staging_digest=staging_digest,
                output_root=output_root,
                entry_count=len(plan.entries),
                committed_at=_now(),
            )
        return commit_id

    def record_reversal(self, *, commit_id: str, detail: dict[str, Any]) -> None:
        with self.engine.begin() as connection:
            self._insert(
                connection,
                schema.reversals,
                reversal_id=_id(),
                commit_id=commit_id,
                detail=canonical_json(detail),
                reversed_at=_now(),
            )

    # -- queries ------------------------------------------------------------------------

    def dag(self, run_id: str, stage_name: str) -> dict[str, Any]:
        """The stage DAG for every attempt, including the ones that failed."""
        with self.engine.begin() as connection:
            attempts = connection.execute(
                select(schema.stage_attempts)
                .where(schema.stage_attempts.c.run_id == run_id)
                .where(schema.stage_attempts.c.stage_name == stage_name)
                .order_by(schema.stage_attempts.c.attempt_no)
            ).mappings().all()

            output: list[dict[str, Any]] = []
            for attempt in attempts:
                compositions = connection.execute(
                    select(schema.compositions).where(
                        schema.compositions.c.attempt_id == attempt["attempt_id"]
                    )
                ).mappings().all()
                nodes: list[dict[str, Any]] = []
                edges: list[dict[str, str]] = []
                for composition in compositions:
                    nodes.extend(
                        dict(row)
                        for row in connection.execute(
                            select(schema.invocations)
                            .where(
                                schema.invocations.c.composition_id
                                == composition["composition_id"]
                            )
                            .order_by(schema.invocations.c.seq)
                        ).mappings()
                    )
                    edges.extend(
                        dict(row)
                        for row in connection.execute(
                            select(schema.invocation_edges).where(
                                schema.invocation_edges.c.composition_id
                                == composition["composition_id"]
                            )
                        ).mappings()
                    )
                verdict = connection.execute(
                    select(schema.stage_verdicts).where(
                        schema.stage_verdicts.c.attempt_id == attempt["attempt_id"]
                    )
                ).mappings().first()
                output.append(
                    {
                        "attempt": dict(attempt),
                        "nodes": nodes,
                        "edges": edges,
                        "verdict": dict(verdict) if verdict else None,
                    }
                )
            return {"run_id": run_id, "stage": stage_name, "attempts": output}

    def replay_source(self, run_id: str) -> dict[str, Any]:
        """Everything needed to re-execute a run with no model calls.

        Failed attempts are included: replay reproduces the sequence that actually
        happened, not an idealised one, or it would not be a replay.
        """
        with self.engine.begin() as connection:
            run = connection.execute(
                select(schema.runs).where(schema.runs.c.run_id == run_id)
            ).mappings().first()
            if run is None:
                raise KeyError(f"unknown run: {run_id}")

            attempts = connection.execute(
                select(schema.stage_attempts)
                .where(schema.stage_attempts.c.run_id == run_id)
                .order_by(schema.stage_attempts.c.started_at, schema.stage_attempts.c.attempt_no)
            ).mappings().all()

            recorded: list[dict[str, Any]] = []
            for attempt in attempts:
                plan = connection.execute(
                    select(schema.stage_plans).where(
                        schema.stage_plans.c.attempt_id == attempt["attempt_id"]
                    )
                ).mappings().first()
                verdict = connection.execute(
                    select(schema.stage_verdicts).where(
                        schema.stage_verdicts.c.attempt_id == attempt["attempt_id"]
                    )
                ).mappings().first()
                compositions = connection.execute(
                    select(schema.compositions).where(
                        schema.compositions.c.attempt_id == attempt["attempt_id"]
                    )
                ).mappings().all()
                recorded.append(
                    {
                        "stage_name": attempt["stage_name"],
                        "stage_version": attempt["stage_version"],
                        "attempt_no": attempt["attempt_no"],
                        "plan": json.loads(plan["payload"]) if plan else None,
                        "verdict": dict(verdict) if verdict else None,
                        "compositions": {
                            row["domain_id"]: json.loads(row["payload"]) for row in compositions
                        },
                    }
                )

        return {
            "run_id": run_id,
            "workflow_id": run["workflow_id"],
            "workflow_digest": run["workflow_digest"],
            "attempts": recorded,
        }

    def pending_admissions(self) -> list[dict[str, Any]]:
        """Requests that have a report but no decision yet.

        A high-risk request suspends its run, so this is the queue a person works from.
        """
        with self.engine.begin() as connection:
            decided = {
                row[0]
                for row in connection.execute(select(schema.admission_decisions.c.report_id))
            }
            rows = connection.execute(
                select(schema.admission_reports, schema.operator_requests)
                .select_from(
                    schema.admission_reports.join(
                        schema.operator_requests,
                        schema.admission_reports.c.request_id
                        == schema.operator_requests.c.request_id,
                    )
                )
                .order_by(schema.operator_requests.c.recorded_at)
            ).mappings().all()

        return [
            {
                "report_id": row["report_id"],
                "request_id": row["request_id"],
                "candidate_id": row["candidate_id"],
                "name": row["name"],
                "family": row["family"],
                "kind": row["kind"],
                "computed_risk": row["computed_risk"],
                "disposition": row["disposition"],
                "package_digest": row["package_digest"],
                "reproducible": bool(row["reproducible"]),
                "findings": json.loads(row["findings"]),
                "test_results": json.loads(row["test_results"]),
                "request": json.loads(row["payload"]),
            }
            for row in rows
            if row["report_id"] not in decided
        ]

    def promoted_prompts(self) -> list[dict[str, Any]]:
        """Optimization runs and how each was decided."""
        with self.engine.begin() as connection:
            runs = connection.execute(
                select(schema.optimization_runs).order_by(
                    schema.optimization_runs.c.recorded_at
                )
            ).mappings().all()
            decisions = {
                row["optimization_id"]: dict(row)
                for row in connection.execute(select(schema.promotion_decisions)).mappings()
            }
        return [{**dict(run), "decision": decisions.get(run["optimization_id"])} for run in runs]

    def costs(self, run_id: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                select(schema.model_invocations).where(
                    schema.model_invocations.c.run_id == run_id
                )
            ).mappings().all()
        return {
            "run_id": run_id,
            "model_invocations": len(rows),
            "prompt_tokens": sum(row["prompt_tokens"] or 0 for row in rows),
            "completion_tokens": sum(row["completion_tokens"] or 0 for row in rows),
            "cost_usd": round(sum(row["cost_usd"] or 0.0 for row in rows), 6),
            "by_role": _group_costs(rows),
        }


def _group_costs(rows: Sequence[Any]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for row in rows:
        grouped[row["role"]] = round(grouped.get(row["role"], 0.0) + (row["cost_usd"] or 0.0), 6)
    return grouped


def loads(payload: str) -> Any:
    return json.loads(payload)
