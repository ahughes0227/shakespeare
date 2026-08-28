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
uv run shakespeare replay <run-id> -i ./in    # re-execute from the journal, no model calls
uv run shakespeare apply --plan plan.json -i ./in -o ./out
uv run shakespeare requests list | review <id> | approve <id> | deny <id>
uv run shakespeare prompts list | promote <sig> --candidate 1.1.0 --score 0.9
uv run shakespeare canary list | record <name> | run
uv run shakespeare measurements list | propose | recovery
```

`replay` swaps only the planner and the domain agents for journal-backed ones — the same
verifier, executor and obligations run — so a replay that reproduces the original plan is
evidence that the recorded compositions determine the result. It refuses to run against a
changed workflow digest.

`apply` re-verifies the plan's recorded source digests before staging, so a file changed
or deleted since planning stops the commit rather than being renamed on stale information.

`canary` re-runs golden cases through the real model on purpose: the point is to notice
when the same prompt over the same files stops producing the same answer — a promoted
prompt, a new operator version, or a provider changing silently behind an alias.

`measurements` is where a declared constant goes to be replaced by a measured one. Every
run already measures what a batch costs and every run used to throw it away, so the
estimate each one starts from is whatever number a person typed into a manifest. Runs now
record those observations; `propose` says which constant the accumulated evidence supports
and prints the file to write it into. Nothing reads the ledger during a run — a measured
constant reaches a run only as a manifest edit, which is what keeps a run determined by
its journal and keeps `replay` a statement about what was recorded rather than about what
a database happened to hold that day.

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
human change** to the allowlists in `system/runners.py`; a generated package
contains no callable, which is what makes it safe for a subagent to ask for one.

## Telemetry

Three channels, correlated by `run_id`:

| Channel | Holds | Leaves the machine |
| --- | --- | --- |
| Audit log (SQLite) | Committed facts, the per-stage DAG, costs, decisions | Never |
| Traces (OpenTelemetry, LangSmith) | Live spans over run → goal → round → component | **Digests and metadata only** |
| Workspace | Extracted text, staged files | Never |

The inputs are customer documents, so redaction is architectural rather than a setting:
`Tracer.span` accepts only envelope primitives, so there is no parameter through which
content could be passed. Nothing is exported at all unless `LANGSMITH_PROJECT` and
`LANGSMITH_API_KEY` are both set.

Tracing activates on the standard `OTEL_EXPORTER_OTLP_ENDPOINT`, so a collector already
running for something else picks Shakespeare up with no further configuration. LangSmith
needs `LANGSMITH_PROJECT` and `LANGSMITH_API_KEY`. Both can run at once, and with neither
set nothing is exported at all.

Span attributes are set from `TelemetryEnvelope` fields and nothing else, so there is no
path by which document content could become an attribute — the same guarantee as the
envelope itself, enforced at the same single point.

LangGraph's checkpointer holds local working state, including content-derived values. It
sits in the run's own workspace alongside the extracted text and is protected the same
way. LangChain's automatic node tracing is disabled, since it would ship whole node
payloads and bypass the envelope entirely.

## Development

```bash
uv run ruff check shakespeare tests
uv run mypy shakespeare
uv run pytest -q
uv run alembic upgrade head    # the audit log is migrated, never recreated
```

The suite runs offline. Every model touchpoint has a fake, and the fakes validate exactly
as production does, so a response that violates its contract fails in a test the same way
it would in a run. `tests/test_runtime.py` drives a throwaway `noop_passthrough` workflow
through the same driver to prove the spine is generic.

Extraction is tested against genuinely-parseable files — real PDFs, DOCX, XLSX, email and
images, generated by `tests/fixtures/build.py` rather than committed, so a backend that
drifts from its library's API fails in the suite rather than on a user's invoices.

There is an opt-in live lane that spends money and is skipped by default, so it can never
be mistaken for coverage:

```bash
SHAKESPEARE_LIVE=1 SHAKESPEARE_MODEL=openrouter/openai/gpt-5-mini \
  uv run pytest tests/test_durability.py -k Live -s
```

**Note:** `tesseract` is not required to install. Without it the OCR backends return
`ocr_unavailable`, scanned documents are quarantined rather than named, and the OCR tests
*skip* rather than silently pass — so the gap stays visible in the test output. Install it
with `brew install tesseract` to cover that path.

Start with `docs/00_OVERVIEW.md` and `docs/reference/glossary.md`.
