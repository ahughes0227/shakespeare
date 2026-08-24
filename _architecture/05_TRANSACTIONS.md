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

Redaction is architectural: model calls go through our own gateway rather than LangChain
LM wrappers, and `TelemetryEnvelope` is the only exportable shape — there is no parameter
on `Tracer.span` through which content could be passed. Client-side masking is defence in
depth. A fake-exporter test asserts no fixture content ever ships.

LangGraph's checkpointer is *local working state* and does contain content-derived values;
it lives in the run's own workspace beside the extracted text and is protected the same
way. The one path by which it could escape is LangChain's automatic node tracing, so
`graph.disable_autotracing()` turns that off unless deliberately overridden.

## Replay and undo

Same inputs plus the same pinned workflow digest give a byte-identical result. `replay`
re-executes journaled compositions with zero model calls. Every mutation declares and
journals a reverse operation.
