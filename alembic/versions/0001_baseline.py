"""Baseline audit schema.

A frozen snapshot, deliberately not a call to `metadata.create_all`. A metadata-driven
baseline silently rewrites itself whenever a model changes, so a later migration adding a
column collides with a baseline that has already grown it — and an existing database and
a fresh one diverge with nothing to notice.

Append-only triggers are installed from `schema.install_append_only_triggers`, which is
derived from the table list rather than the columns, so it does not drift.

There is no meaningful downgrade: dropping an append-only ledger would destroy the
provenance record it exists to keep.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from shakespeare.runtime.audit import schema

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'operator_registrations',
        sa.Column('registration_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('family', sa.String(), nullable=False),
        sa.Column('package_digest', sa.String(), nullable=False),
        sa.Column('admission_id', sa.String(), nullable=False),
        sa.Column('registered_at', sa.String(), nullable=False),
    )
    op.create_table(
        'operator_requests',
        sa.Column('request_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('domain_id', sa.String(), nullable=False),
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('family', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('payload', sa.Text(), nullable=False),
        sa.Column('recorded_at', sa.String(), nullable=False),
    )
    op.create_table(
        'optimization_runs',
        sa.Column('optimization_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('signature_id', sa.String(), nullable=False),
        sa.Column('optimizer', sa.String(), nullable=False),
        sa.Column('eval_set_digest', sa.String(), nullable=False),
        sa.Column('incumbent_version', sa.String(), nullable=True),
        sa.Column('incumbent_score', sa.Float(), nullable=True),
        sa.Column('candidate_version', sa.String(), nullable=False),
        sa.Column('candidate_score', sa.Float(), nullable=False),
        sa.Column('fixture_regressions', sa.Text(), nullable=False),
        sa.Column('recorded_at', sa.String(), nullable=False),
    )
    op.create_table(
        'prompt_artifacts',
        sa.Column('signature_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('version', sa.String(), nullable=False, primary_key=True),
        sa.Column('digest', sa.String(), nullable=False),
        sa.Column('payload', sa.Text(), nullable=False),
        sa.Column('recorded_at', sa.String(), nullable=False),
    )
    op.create_table(
        'runs',
        sa.Column('run_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('workflow_id', sa.String(), nullable=False),
        sa.Column('workflow_version', sa.String(), nullable=False),
        sa.Column('workflow_digest', sa.String(), nullable=False),
        sa.Column('request_digest', sa.String(), nullable=False),
        sa.Column('input_root_digest', sa.String(), nullable=False),
    )
    op.create_table(
        'admission_reports',
        sa.Column('report_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('request_id', sa.String(), sa.ForeignKey('operator_requests.request_id'), nullable=False),
        sa.Column('candidate_id', sa.String(), nullable=False),
        sa.Column('computed_risk', sa.String(), nullable=False),
        sa.Column('disposition', sa.String(), nullable=False),
        sa.Column('package_digest', sa.String(), nullable=False),
        sa.Column('reproducible', sa.Boolean(), nullable=False),
        sa.Column('findings', sa.Text(), nullable=False),
        sa.Column('test_results', sa.Text(), nullable=False),
    )
    op.create_table(
        'commits',
        sa.Column('commit_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('run_id', sa.String(), sa.ForeignKey('runs.run_id'), nullable=False),
        sa.Column('staging_digest', sa.String(), nullable=False),
        sa.Column('output_root', sa.String(), nullable=False),
        sa.Column('entry_count', sa.Integer(), nullable=False),
        sa.Column('committed_at', sa.String(), nullable=False),
    )
    op.create_table(
        'model_invocations',
        sa.Column('invocation_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('run_id', sa.String(), sa.ForeignKey('runs.run_id'), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('profile_id', sa.String(), nullable=False),
        sa.Column('requested_model', sa.String(), nullable=False),
        sa.Column('resolved_model', sa.String(), nullable=True),
        sa.Column('provider', sa.String(), nullable=True),
        sa.Column('prompt_version', sa.String(), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
        sa.Column('cost_usd', sa.Float(), nullable=True),
        sa.Column('error_code', sa.String(), nullable=True),
        sa.Column('recorded_at', sa.String(), nullable=False),
    )
    op.create_table(
        'mutations',
        sa.Column('mutation_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('run_id', sa.String(), sa.ForeignKey('runs.run_id'), nullable=False),
        sa.Column('invocation_id', sa.String(), nullable=True),
        sa.Column('target_ref', sa.String(), nullable=False),
        sa.Column('operation', sa.String(), nullable=False),
        sa.Column('before_digest', sa.String(), nullable=True),
        sa.Column('after_digest', sa.String(), nullable=True),
        sa.Column('reversal', sa.Text(), nullable=False),
        sa.Column('recorded_at', sa.String(), nullable=False),
    )
    op.create_table(
        'plans',
        sa.Column('plan_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('run_id', sa.String(), sa.ForeignKey('runs.run_id'), nullable=False),
        sa.Column('digest', sa.String(), nullable=False),
        sa.Column('entry_count', sa.Integer(), nullable=False),
        sa.Column('changed', sa.Integer(), nullable=False),
        sa.Column('unchanged', sa.Integer(), nullable=False),
        sa.Column('unresolved', sa.Integer(), nullable=False),
        sa.Column('payload', sa.Text(), nullable=False),
        sa.Column('recorded_at', sa.String(), nullable=False),
    )
    op.create_table(
        'promotion_decisions',
        sa.Column('decision_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('optimization_id', sa.String(), sa.ForeignKey('optimization_runs.optimization_id'), nullable=True),
        sa.Column('decided_by', sa.String(), nullable=False),
        sa.Column('choice', sa.String(), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
        sa.Column('decided_at', sa.String(), nullable=False),
    )
    op.create_table(
        'run_outcomes',
        sa.Column('run_id', sa.String(), sa.ForeignKey('runs.run_id'), nullable=False, primary_key=True),
        sa.Column('ended_at', sa.String(), nullable=False),
        sa.Column('outcome', sa.String(), nullable=False),
        sa.Column('error_code', sa.String(), nullable=True),
    )
    op.create_table(
        'stage_attempts',
        sa.Column('attempt_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('run_id', sa.String(), sa.ForeignKey('runs.run_id'), nullable=False),
        sa.Column('stage_name', sa.String(), nullable=False),
        sa.Column('stage_version', sa.String(), nullable=False),
        sa.Column('attempt_no', sa.Integer(), nullable=False),
        sa.Column('started_at', sa.String(), nullable=False),
        sa.Column('ended_at', sa.String(), nullable=False),
        sa.Column('error_code', sa.String(), nullable=True),
    )
    op.create_table(
        'admission_decisions',
        sa.Column('decision_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('report_id', sa.String(), sa.ForeignKey('admission_reports.report_id'), nullable=False),
        sa.Column('decided_by', sa.String(), nullable=False),
        sa.Column('choice', sa.String(), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
        sa.Column('decided_at', sa.String(), nullable=False),
    )
    op.create_table(
        'compositions',
        sa.Column('composition_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('attempt_id', sa.String(), sa.ForeignKey('stage_attempts.attempt_id'), nullable=False),
        sa.Column('domain_id', sa.String(), nullable=False),
        sa.Column('digest', sa.String(), nullable=False),
        sa.Column('payload', sa.Text(), nullable=False),
    )
    op.create_table(
        'obligation_results',
        sa.Column('result_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('attempt_id', sa.String(), sa.ForeignKey('stage_attempts.attempt_id'), nullable=False),
        sa.Column('obligation_id', sa.String(), nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=False),
        sa.Column('detail', sa.Text(), nullable=False),
    )
    op.create_table(
        'reversals',
        sa.Column('reversal_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('commit_id', sa.String(), sa.ForeignKey('commits.commit_id'), nullable=False),
        sa.Column('detail', sa.Text(), nullable=False),
        sa.Column('reversed_at', sa.String(), nullable=False),
    )
    op.create_table(
        'stage_plans',
        sa.Column('plan_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('attempt_id', sa.String(), sa.ForeignKey('stage_attempts.attempt_id'), nullable=False),
        sa.Column('digest', sa.String(), nullable=False),
        sa.Column('payload', sa.Text(), nullable=False),
    )
    op.create_table(
        'stage_verdicts',
        sa.Column('verdict_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('attempt_id', sa.String(), sa.ForeignKey('stage_attempts.attempt_id'), nullable=False),
        sa.Column('met', sa.Boolean(), nullable=False),
        sa.Column('decision', sa.String(), nullable=False),
        sa.Column('unmet', sa.Text(), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
    )
    op.create_table(
        'domain_goals',
        sa.Column('goal_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('plan_id', sa.String(), sa.ForeignKey('stage_plans.plan_id'), nullable=False),
        sa.Column('domain_id', sa.String(), nullable=False),
        sa.Column('activated', sa.Boolean(), nullable=False),
        sa.Column('goal', sa.Text(), nullable=True),
        sa.Column('success_criterion', sa.Text(), nullable=True),
        sa.Column('skip_reason', sa.Text(), nullable=True),
    )
    op.create_table(
        'invocation_edges',
        sa.Column('composition_id', sa.String(), sa.ForeignKey('compositions.composition_id'), nullable=False, primary_key=True),
        sa.Column('from_invocation', sa.String(), nullable=False, primary_key=True),
        sa.Column('to_invocation', sa.String(), nullable=False, primary_key=True),
    )
    op.create_table(
        'invocations',
        sa.Column('invocation_id', sa.String(), nullable=False, primary_key=True),
        sa.Column('composition_id', sa.String(), sa.ForeignKey('compositions.composition_id'), nullable=False),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('operator', sa.String(), nullable=False),
        sa.Column('operator_version', sa.String(), nullable=False),
        sa.Column('selections', sa.Text(), nullable=False),
        sa.Column('parameters', sa.Text(), nullable=False),
        sa.Column('started_at', sa.String(), nullable=False),
        sa.Column('ended_at', sa.String(), nullable=False),
        sa.Column('succeeded', sa.Boolean(), nullable=False),
        sa.Column('output_digest', sa.String(), nullable=True),
        sa.Column('error_code', sa.String(), nullable=True),
    )

    schema.install_append_only_triggers(op.get_bind())


def downgrade() -> None:
    raise RuntimeError(
        "the audit log is append-only and permanent; there is no supported downgrade"
    )
