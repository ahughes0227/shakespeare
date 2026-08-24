# Shakespeare

Shakespeare edits files according to a prompt. It is a **transactional** program: it
values consistency over creativity, so the model's authority is deliberately narrow and
everything it decides is checked before anything is written.

```
prompt -> planner selects a prebuilt workflow -> ordered versioned stages
       -> domain subagents compile Hydra compositions -> runtime executes
       -> obligations verified -> planner reviews -> atomic commit
```

The first workflow renames files — arbitrary count, arbitrary nesting, output mirroring
the input tree. Sequential numbering works, but the intended case is reasoning over
document content, e.g. invoices to `YYYYMM, vendor, invoice number, PO number.pdf`.

## The four levels

| Level | Unit | Who decides | Bounded by |
| --- | --- | --- | --- |
| — | **Planner** | model | Selects among registered workflows; issues goals; reviews each stage; reruns within `max_attempts` |
| 0 | **Workflow** | programmer | An ordered spine of version-pinned stages |
| 1 | **Stage** | programmer | Typed inputs and outputs; a new stage wherever output must be interpreted by a model |
| 2 | **Domain** | model composes | An immutable scope, an operator catalog, and Hydra config groups |
| 3 | **Operator** | programmer | A Copier family template and a pinned trusted runner |

Two rules decide the shape of everything else:

- **Stage boundary rule.** A new stage is required wherever outputs must be transformed
  or verified by a large model.
- **Composition rule.** A domain subagent *compiles* a configuration; the **runtime
  executes it**. A subagent never sees its own operator output — if that output needs
  interpreting, that is the next stage.

A stage constrains the *type* of work, never the method. Nothing tells an agent how to do
its job; the surface it may act on is what is bounded.

## What guarantees consistency

Not the agents. Consistency is a property of code and of the commit:

- **Names are rendered, never written.** A convention is compiled once, validated, and
  digested; every file is then rendered from it mechanically. `plan.assemble` refuses a
  filename an agent supplied as a parameter, so names must flow from the renderer.
- **Nothing is guessed.** An item whose fields do not resolve confidently is quarantined
  under its original name with a reason, never given a plausible one.
- **Accounting balances.** Every input ends in exactly one state — `changed`,
  `unchanged`, or `unresolved` — or the run aborts before touching anything.
- **Two-phase commit.** Execute writes only into a staging tree. Review verifies. Commit
  is one atomic move; a failed review discards staging and the output root is never
  created.
- **Only the runtime writes.** Every `filesystem_mutation` operator is runtime-only and
  appears in no domain catalog. Agents plan; the runtime commits.
- **Everything is reversible and replayable.** Each mutation journals a reversal, and the
  audit log records the operator sequence that actually ran.

## Install and run

```bash
uv sync --group dev
export SHAKESPEARE_MODEL=openrouter/openai/gpt-5-mini   # a fixed model; aliases are refused
uv run shakespeare run \
  --prompt "rename these invoices to YYYYMM, vendor, invoice number, PO number" \
  --input ./invoices --output ./renamed
```

`run` plans, previews, and asks before committing. `plan` stops before any write.

```bash
uv run shakespeare plan -p "..." -i ./in -o ./out --plan-out plan.json
uv run shakespeare workflows list | validate     # validate type-checks every spine
uv run shakespeare stages | operators
uv run shakespeare journal dag <run-id> extract  # what actually ran, failed attempts included
uv run shakespeare metrics                       # agent-ops SLIs
uv run shakespeare undo <run-id>
```

## Extending it

Adding a workflow is composition, not construction:

```bash
uv run shakespeare new-workflow --id edit_sop     # a spine of stage refs + a ten-field card
uv run shakespeare new-stage --name edit_design   # only for genuinely new work
uv run shakespeare new-operator --family pure_transform --name doc.transform \
    --operation render_template
```

The planner picks up a new workflow automatically: it reads only `workflow-context.yml`,
so routing extends without touching `planner.py`. Registration type-checks the spine —
contracts must line up, stage versions must resolve, catalog operators must be
registered, config groups must exist, and no domain catalog may contain an operator that
writes.

A requested operator can select vetted behaviour and configure it. **New behaviour is a
human change** to the allowlists in `shakespeare/runners.py`; a generated package
contains no callable, which is what makes it safe for a subagent to ask for one.

## Telemetry

Three channels, correlated by `run_id`:

| Channel | Holds | Leaves the machine |
| --- | --- | --- |
| Audit log (SQLite) | Committed facts, the per-stage DAG, costs, decisions | Never |
| Traces (LangSmith) | Live spans over run → stage → attempt → invocation | **Digests and metadata only** |
| Workspace | Extracted text, staged files | Never |

The inputs are customer documents, so redaction is architectural rather than a setting:
`Tracer.span` accepts only envelope primitives, so there is no parameter through which
content could be passed. Nothing is exported at all unless `LANGSMITH_PROJECT` and
`LANGSMITH_API_KEY` are both set.

LangGraph's checkpointer holds local working state, including content-derived values. It
sits in the run's own workspace alongside the extracted text and is protected the same
way. LangChain's automatic node tracing is disabled, since it would ship whole node
payloads and bypass the envelope entirely.

## Development

```bash
uv run ruff check shakespeare tests
uv run mypy shakespeare
uv run pytest -q
```

The suite runs offline. Every model touchpoint has a fake, and the fakes validate exactly
as production does, so a response that violates its contract fails in a test the same way
it would in a run. `tests/test_runtime.py` drives a throwaway `noop_passthrough` workflow
through the same driver to prove the spine is generic.

**Note:** `tesseract` is not required to install, but without it the OCR backends return
`ocr_unavailable` and scanned documents are quarantined rather than named. Install it
(`brew install tesseract`) to exercise that path.

Start with `_architecture/00_OVERVIEW.md` and `_core/glossary.md`.
