"""Append-only audit schema.

Every row is a fact, inserted once at the moment it becomes true.  There are no status
columns to mutate: progress and resumption belong to the LangGraph checkpointer, not
here.  Immutability is enforced by SQLite triggers rather than by convention, so a coding
mistake cannot quietly rewrite history.

`invocations` are the nodes of each stage DAG and `invocation_edges` are the data
dependencies a domain subagent expressed in its composition.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    text,
)
from sqlalchemy.engine import Connection

metadata = MetaData()


def _table(name: str, *columns: Column[Any]) -> Table:
    return Table(name, metadata, *columns)


runs = _table(
    "runs",
    Column("run_id", String, primary_key=True),
    Column("created_at", String, nullable=False),
    Column("workflow_id", String, nullable=False),
    Column("workflow_version", String, nullable=False),
    Column("workflow_digest", String, nullable=False),
    Column("request_digest", String, nullable=False),
    Column("input_root_digest", String, nullable=False),
)

#: How a run *ended*. A planned-but-uncommitted run has no row here, which is the
#: difference between "not finished" and "finished without committing".
run_outcomes = _table(
    "run_outcomes",
    Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
    Column("ended_at", String, nullable=False),
    Column("outcome", String, nullable=False),  # committed | aborted | reversed
    Column("error_code", String),
)

stage_attempts = _table(
    "stage_attempts",
    Column("attempt_id", String, primary_key=True),
    Column("run_id", String, ForeignKey("runs.run_id"), nullable=False),
    Column("stage_name", String, nullable=False),
    Column("stage_version", String, nullable=False),
    Column("attempt_no", Integer, nullable=False),
    Column("started_at", String, nullable=False),
    Column("ended_at", String, nullable=False),
    Column("error_code", String),
)

stage_plans = _table(
    "stage_plans",
    Column("plan_id", String, primary_key=True),
    Column("attempt_id", String, ForeignKey("stage_attempts.attempt_id"), nullable=False),
    Column("digest", String, nullable=False),
    Column("payload", Text, nullable=False),
)

domain_goals = _table(
    "domain_goals",
    Column("goal_id", String, primary_key=True),
    Column("plan_id", String, ForeignKey("stage_plans.plan_id"), nullable=False),
    Column("domain_id", String, nullable=False),
    Column("activated", Boolean, nullable=False),
    Column("goal", Text),
    Column("success_criterion", Text),
    Column("skip_reason", Text),
)

compositions = _table(
    "compositions",
    Column("composition_id", String, primary_key=True),
    Column("attempt_id", String, ForeignKey("stage_attempts.attempt_id"), nullable=False),
    Column("domain_id", String, nullable=False),
    Column("digest", String, nullable=False),
    Column("payload", Text, nullable=False),
)

invocations = _table(
    "invocations",
    Column("invocation_id", String, primary_key=True),
    Column("composition_id", String, ForeignKey("compositions.composition_id"), nullable=False),
    Column("seq", Integer, nullable=False),
    Column("operator", String, nullable=False),
    Column("operator_version", String, nullable=False),
    Column("selections", Text, nullable=False),
    Column("parameters", Text, nullable=False),
    Column("started_at", String, nullable=False),
    Column("ended_at", String, nullable=False),
    Column("succeeded", Boolean, nullable=False),
    Column("output_digest", String),
    Column("error_code", String),
)

invocation_edges = _table(
    "invocation_edges",
    Column("composition_id", String, ForeignKey("compositions.composition_id"), primary_key=True),
    Column("from_invocation", String, primary_key=True),
    Column("to_invocation", String, primary_key=True),
)

obligation_results = _table(
    "obligation_results",
    Column("result_id", String, primary_key=True),
    Column("attempt_id", String, ForeignKey("stage_attempts.attempt_id"), nullable=False),
    Column("obligation_id", String, nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("detail", Text, nullable=False),
)

stage_verdicts = _table(
    "stage_verdicts",
    Column("verdict_id", String, primary_key=True),
    Column("attempt_id", String, ForeignKey("stage_attempts.attempt_id"), nullable=False),
    Column("met", Boolean, nullable=False),
    Column("decision", String, nullable=False),
    Column("unmet", Text, nullable=False),
    Column("rationale", Text, nullable=False),
)

model_invocations = _table(
    "model_invocations",
    Column("invocation_id", String, primary_key=True),
    Column("run_id", String, ForeignKey("runs.run_id"), nullable=False),
    Column("role", String, nullable=False),
    Column("profile_id", String, nullable=False),
    Column("requested_model", String, nullable=False),
    # Requested vs resolved is how a provider silently changing behind an alias becomes
    # attributable instead of invisible.
    Column("resolved_model", String),
    Column("provider", String),
    Column("prompt_version", String),
    Column("prompt_tokens", Integer),
    Column("completion_tokens", Integer),
    Column("cost_usd", Float),
    Column("error_code", String),
    Column("recorded_at", String, nullable=False),
)

operator_requests = _table(
    "operator_requests",
    Column("request_id", String, primary_key=True),
    Column("run_id", String, nullable=False),
    Column("domain_id", String, nullable=False),
    Column("kind", String, nullable=False),
    Column("family", String, nullable=False),
    Column("name", String, nullable=False),
    Column("payload", Text, nullable=False),
    Column("recorded_at", String, nullable=False),
)

admission_reports = _table(
    "admission_reports",
    Column("report_id", String, primary_key=True),
    Column("request_id", String, ForeignKey("operator_requests.request_id"), nullable=False),
    Column("candidate_id", String, nullable=False),
    Column("computed_risk", String, nullable=False),
    Column("disposition", String, nullable=False),
    Column("package_digest", String, nullable=False),
    Column("reproducible", Boolean, nullable=False),
    Column("findings", Text, nullable=False),
    Column("test_results", Text, nullable=False),
)

admission_decisions = _table(
    "admission_decisions",
    Column("decision_id", String, primary_key=True),
    Column("report_id", String, ForeignKey("admission_reports.report_id"), nullable=False),
    Column("decided_by", String, nullable=False),
    Column("choice", String, nullable=False),
    Column("rationale", Text, nullable=False),
    Column("decided_at", String, nullable=False),
)

operator_registrations = _table(
    "operator_registrations",
    Column("registration_id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("version", String, nullable=False),
    Column("family", String, nullable=False),
    Column("package_digest", String, nullable=False),
    Column("admission_id", String, nullable=False),
    Column("registered_at", String, nullable=False),
)

prompt_artifacts = _table(
    "prompt_artifacts",
    Column("signature_id", String, primary_key=True),
    Column("version", String, primary_key=True),
    Column("digest", String, nullable=False),
    Column("payload", Text, nullable=False),
    Column("recorded_at", String, nullable=False),
)

optimization_runs = _table(
    "optimization_runs",
    Column("optimization_id", String, primary_key=True),
    Column("signature_id", String, nullable=False),
    Column("optimizer", String, nullable=False),
    Column("eval_set_digest", String, nullable=False),
    Column("incumbent_version", String),
    Column("incumbent_score", Float),
    Column("candidate_version", String, nullable=False),
    Column("candidate_score", Float, nullable=False),
    Column("fixture_regressions", Text, nullable=False),
    Column("recorded_at", String, nullable=False),
)

promotion_decisions = _table(
    "promotion_decisions",
    Column("decision_id", String, primary_key=True),
    Column("optimization_id", String, ForeignKey("optimization_runs.optimization_id")),
    Column("decided_by", String, nullable=False),
    Column("choice", String, nullable=False),
    Column("rationale", Text, nullable=False),
    Column("decided_at", String, nullable=False),
)

mutations = _table(
    "mutations",
    Column("mutation_id", String, primary_key=True),
    Column("run_id", String, ForeignKey("runs.run_id"), nullable=False),
    Column("invocation_id", String),
    Column("target_ref", String, nullable=False),
    Column("operation", String, nullable=False),
    Column("before_digest", String),
    Column("after_digest", String),
    Column("reversal", Text, nullable=False),
    Column("recorded_at", String, nullable=False),
)

plans = _table(
    "plans",
    Column("plan_id", String, primary_key=True),
    Column("run_id", String, ForeignKey("runs.run_id"), nullable=False),
    Column("digest", String, nullable=False),
    # Identity of the decisions, independent of run_id. The idempotency receipt matches
    # on this, because two runs of the same request never share a plan digest.
    Column("fingerprint", String, nullable=False, server_default=""),
    Column("entry_count", Integer, nullable=False),
    Column("changed", Integer, nullable=False),
    Column("unchanged", Integer, nullable=False),
    Column("unresolved", Integer, nullable=False),
    Column("payload", Text, nullable=False),
    Column("recorded_at", String, nullable=False),
)

commits = _table(
    "commits",
    Column("commit_id", String, primary_key=True),
    Column("run_id", String, ForeignKey("runs.run_id"), nullable=False),
    Column("staging_digest", String, nullable=False),
    Column("output_root", String, nullable=False),
    Column("entry_count", Integer, nullable=False),
    Column("committed_at", String, nullable=False),
)

reversals = _table(
    "reversals",
    Column("reversal_id", String, primary_key=True),
    Column("commit_id", String, ForeignKey("commits.commit_id"), nullable=False),
    Column("detail", Text, nullable=False),
    Column("reversed_at", String, nullable=False),
)


#: What runs actually measured, one row per observation.
#:
#: Deliberately observations rather than aggregates: an average cannot be re-derived
#: under a new weighting and cannot be invalidated when the model behind it changes.
#: Nothing reads this table during a run — a measured constant reaches a run only by
#: being promoted into a manifest, so what a run does stays fixed and digested at its
#: start and `replay` keeps meaning what it means.
measurements = _table(
    "measurements",
    Column("measurement_id", String, primary_key=True),
    Column("run_id", String, ForeignKey("runs.run_id"), nullable=False),
    Column("kind", String, nullable=False),  # schedule_cost | confidence
    Column("subject", String, nullable=False),  # capability ref | field name
    # Identity, not metadata: a measurement taken under one model says nothing about
    # another, and mixing them silently is how a promoted constant becomes wrong.
    Column("resolved_model", String, nullable=False),
    Column("prompt_version", String, nullable=False),
    Column("value", Float, nullable=False),
    Column("weight", Float, nullable=False),
    Column("count", Integer, nullable=False),
    Column("outcome", Boolean, nullable=False),
    Column("bound", String),  # "lower" when value limits the quantity rather than reporting it
    Column("recorded_at", String, nullable=False),
)


def install_append_only_triggers(connection: Connection) -> None:
    """Reject UPDATE and DELETE on every audit table.

    This is what makes the log an audit log rather than a working table.

    Only tables that actually exist are covered. The trigger set is derived from the
    model list so that it cannot drift from it, but an older migration runs against a
    database that has not yet grown the newer tables — deriving from the models and
    ignoring what is present would make every historical migration fail the moment a
    table is added.
    """
    present = {
        name
        for (name,) in connection.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
    }
    for table in metadata.sorted_tables:
        if table.name not in present:
            continue
        for action in ("UPDATE", "DELETE"):
            trigger = f"{table.name}_no_{action.lower()}"
            connection.execute(
                text(
                    f"CREATE TRIGGER IF NOT EXISTS {trigger} "
                    f"BEFORE {action} ON {table.name} "
                    f"BEGIN SELECT RAISE(ABORT, "
                    f"'{table.name} is append-only: {action} rejected'); END"
                )
            )
