# AGENTS.md

Shakespeare is a staged, transactional file-operations runtime:

    prompt -> planner selects a prebuilt workflow -> ordered versioned stages
           -> domain subagents compile Hydra compositions -> runtime executes
           -> obligations verified -> planner reviews -> atomic commit

Read `docs/00_OVERVIEW.md` through `05_TRANSACTIONS.md` in order before making
architectural changes, then `docs/reference/glossary.md`.

## Non-negotiable boundaries

- The model never authors code, structure, or authority. The planner selects a workflow
  and issues goals; a domain subagent emits a `Composition` and may *request* an operator.
- A domain subagent compiles a configuration; the **runtime executes it**. A subagent
  never observes its own operator output.
- A new stage is required wherever outputs must be transformed or verified by a model.
- Stages constrain the *type* of work, never the method. Do not script an agent's route.
- Adaptation happens only in the planner, only at stage boundaries, only within
  `max_attempts`, and only by revising goals.
- Registered operators are the only executable primitives. Generated operator packages
  contain no callable; behaviour lives in a family's pinned trusted runner.
- Hydra composition is allowlisted, never interpolated.
- `filesystem_mutation` is the only family that may write. `fs.commit` belongs to no
  domain catalog.
- Execute writes to staging only. Commit happens after Review passes, atomically.
- Every input item ends in exactly one terminal state. Unbalanced accounting aborts.
- The SQL audit log is append-only committed facts. Mutable in-flight state belongs to
  the LangGraph checkpointer.
- Telemetry carries digests and metadata only. Never document content.
- Prompts improve offline under version pinning. Nothing self-improves inside a run.

Do not add a second orchestration, execution, registry, or configuration abstraction.
Do not let a model register operators, grant approvals, widen a catalog, or raise a budget.

Prefer strict Pydantic contracts, append-only records, deterministic tests, and maintained
external libraries. Reimplementing a maintained library requires an ADR in
`docs/adr/`. Core tests must run offline with fake planner and agents.
