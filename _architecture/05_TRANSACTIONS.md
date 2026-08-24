# 05 — Transactions, audit, telemetry

## Two-phase commit

Execute writes only into a staging tree. Review verifies. Commit is one atomic move.
Rollback is discarding staging. `fs.commit` is runtime-only and in no domain catalog.

## Balanced accounting

Every input item ends in exactly one terminal state: `changed`, `unchanged`, `unresolved`.
Unbalanced accounting aborts before commit. An item that did not resolve confidently is
quarantined with a reason, never guessed at.

## Audit log

SQLite via SQLAlchemy, migrated by Alembic, append-only enforced by triggers. Rows are
facts inserted once; there are no status columns to mutate. `invocations` are DAG nodes
and `invocation_edges` are the data dependencies a subagent expressed.

Mutable in-flight state belongs to the LangGraph checkpointer, never to the audit log.

## Telemetry

Three channels correlated by `run_id`: the audit log (facts, permanent, local), LangSmith
traces (live, digests only), and the workspace (content, purgeable).

Redaction is architectural: LangGraph state carries references not payloads; model calls
go through our own gateway; `TelemetryEnvelope` is the only exportable shape. Client-side
masking is defence in depth. A fake-exporter test asserts no fixture content ever ships.

## Replay and undo

Same inputs plus the same pinned workflow digest give a byte-identical result. `replay`
re-executes journaled compositions with zero model calls. Every mutation declares and
journals a reverse operation.
