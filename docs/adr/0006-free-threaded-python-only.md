# ADR 0006 — Free-threaded CPython only, and the three stand-ins it cost

**Status:** accepted · **Date:** 2026-08-28

## Context

Shakespeare now targets free-threaded CPython 3.14 (`cp314t`) and nothing else.
`.python-version` pins `3.14t`, `requires-python` is `>=3.14`, and `system/__init__.py`
refuses to import under an interpreter holding the GIL.

The motivation is ahead of the code: nothing in the runtime is threaded today. Committing
to the interpreter first is the cheap ordering. Extraction over a corpus, the per-item work
inside a batch, and verification across a staging tree are all embarrassingly parallel and
all currently serial; the alternative ordering — write the concurrency, then discover which
dependency cannot run without a GIL — is the expensive one.

The version floor is the easy half. `requires-python` can pin an interpreter's version but
not its build: a GIL-enabled 3.14 satisfies `>=3.14` and installs perfectly happily. The
hard half is that a free-threaded interpreter is not one property but two, and the
dependency tree fights both.

## Decisions

### 1. The interpreter is checked at import, not documented

`sys._is_gil_enabled()` is the only thing that can tell a free-threaded build from an
ordinary one, so the check lives at the import every entry point passes through, and it
raises rather than warns. A transactional runtime that silently ran under a different
execution model than the one it was tested on would be exactly the class of thing the rest
of this system exists to prevent.

### 2. The GIL must be pinned off, not merely off

Loading an extension module that has not declared free-threaded support switches the GIL
back on *mid-process*, with nothing but a `RuntimeWarning` to say so. Two of ours do it:
`lxml` (through openpyxl, python-docx and extract-msg) and SQLAlchemy's `cyextension`.
Both ship `cp314t` wheels; neither declares `Py_MOD_GIL_NOT_USED`.

So the guard asserts `sys.flags.gil == 0` as well — `None` means unpinned, and unpinned
means the first DOCX in a corpus quietly re-acquires the GIL for the rest of the run.
Pinning is `PYTHON_GIL=0` (or `-X gil=0`) at process start, which is why every documented
command carries it.

This runs those two libraries in a mode their authors have not vouched for. That is a real
risk being accepted, and it is bounded by the runtime being single-threaded today: the
declaration is about thread-safety, and there are no threads. The order of work matters —
whatever gets parallelised first must not be the XML or ORM path until lxml 7 and
SQLAlchemy declare support, at which point this decision reduces to nothing.

### 3. Three dependencies have no free-threaded build, and `compat/` stands in for them

`orjson`, `tokenizers` and `fastuuid` cannot be installed on `cp314t`. orjson's build
script *refuses* a free-threaded interpreter outright and has never published a `t` wheel;
tokenizers and fastuuid publish abi3 wheels, and abi3 does not exist for a free-threaded
build, so there is nothing to install and their sdists fail to link.

None of the three can be dropped, because each is imported at module scope by something we
depend on: `langgraph_sdk` (hence `langgraph`, hence `system/runtime/durability.py`),
`litellm.utils`, and `litellm._uuid`. So `compat/` holds a pure-Python stand-in for each,
resolved by path through `[tool.uv.sources]`, and `uv.lock` records them as local
directories rather than as registry packages.

This is reimplementing maintained libraries, which is what this ADR exists to authorise.
The three are not equally comfortable:

- **fastuuid** is free. It is a faster `uuid`; the stand-in re-exports the standard
  library's, and the only loss is speed.
- **orjson** is the one that costs less than this ADR first claimed. The claim was that
  LangGraph serializes every checkpoint through it, so a difference in what it emits is a
  difference in what a resumed run believes. That is not what happens:
  `langgraph.checkpoint.serde.jsonplus` packs through **ormsgpack**, and an offline run
  over the graph, durability, replay and control-loop suites calls `orjson.dumps` zero
  times. orjson is on the import path — `langgraph_sdk` and `langsmith` both import it at
  module scope, which is why the stand-in has to exist — but on this configuration nothing
  on the run path calls it. The paths that would are the LangGraph *server* client, which
  we do not use, and LangSmith export, which is off unless both its variables are set.

  So the stand-in still reproduces the observable contract — `bytes` out, compact
  separators, RFC 3339 datetimes, `null` for non-finite floats, non-string keys refused
  unless `OPT_NON_STR_KEYS`, `Fragment` spliced in verbatim — and
  `tests/test_free_threading.py` pins each of those. What it does not reproduce is the
  speed, and that turns out not to matter here. Measured on a 33,000-entry plan, the
  stand-in encodes in 46 ms against a 23 ms floor for the stdlib's C encoder, so even a
  perfect pure-Python fast path is worth 23 ms on a call nothing on the run path makes.
  It was left alone deliberately: see *Rejected* below.
- **tokenizers** is the one that loses something. LiteLLM reaches for a HuggingFace
  tokenizer only to count tokens for models whose tokenizer is not tiktoken's — Claude,
  Cohere, Llama. Those counts cannot be approximated honestly, so the stand-in raises
  `TokenizerUnavailable` there instead of inventing a number that would become a wrong bill
  and a wrong batch size. OpenAI-shaped models, which is what `SHAKESPEARE_MODEL` pins, go
  through tiktoken and never touch it.

Each stand-in names the package it replaces and the day it should be deleted: the one its
upstream publishes a free-threaded wheel.

### 4. `pyarrow` moved to `>=22`

`pyarrow` 21 has no `cp314t` wheel and its sdist wants cmake. 22 was the first release with
one. The upper bound moved to `<26` to admit it.

### 5. Extraction is threaded; lxml is not

The prospective benefit is now taken. `doc.extract` runs a corpus through
`extraction.extract_many`, which is the first threaded work in the runtime. Extraction was
the right first choice for the reason this ADR predicted: it is per-item, share-nothing,
and it is where nearly all of a run's serial time goes.

Three things bound it.

**lxml runs alone.** Decision 2 accepted lxml under a pinned-off GIL *because there were no
threads*. Now there are, so the deal is kept explicitly: `extract_many` splits the corpus by
whether an item's backend chain could reach lxml — DOCX, XLSX and email — and runs those
first, serially, on the calling thread, before any worker exists. The split is conservative:
a chain containing one lxml backend counts as lxml-bound whether or not the fallback is
taken. Deleting `_LXML_BACKENDS` is the whole migration once lxml declares support.

**Order is not left to the scheduler.** Results are placed by input index, never by
completion. A plan is portable data, and two scans of one tree must produce the same
inventory in the same order.

**The worker count is measured, and it is not one per CPU.** Over 120 padded invoice PDFs on
a 15-CPU host: 2 workers 1.50x, 4 workers 2.29x, 8 workers 1.98x, 15 workers 1.76x. It
regresses, and the interpreter is not why — the same host scales pure arithmetic 6.8x and
allocation-heavy work 4.3x at 15 threads. pdfminer is why: `PSLiteralTable` and
`PSKeywordTable` are module-level singletons whose `intern` does a check-then-insert into
one shared dict for every token in every document. So `worker_count()` caps at four.

That shared intern table is also a correctness question, not only a throughput one: two
threads meeting a name neither has interned can each build a distinct object for it. The
mitigation is that the first threadable item is extracted on the calling thread before the
pool starts, which interns the standard PDF names while nothing is racing. What remains is
document-specific and compared by value.

### 6. ormsgpack is a fourth native dependency, and it got it right

Auditing what actually serializes turned up a native extension this ADR had not inventoried.
`langgraph.checkpoint.serde.jsonplus` packs checkpoints through **ormsgpack**, which
publishes a `cp314t` wheel and — unlike lxml and SQLAlchemy — declares free-threaded
support: imported unpinned, it leaves the GIL off. So the durability path is native, fast
and declared, and needs nothing from `compat/`. `tests/test_free_threading.py` pins that,
because it is the one property of ormsgpack this system depends on.

## Rejected: a fast path in the orjson stand-in

The stand-in's `dumps` walks the whole object in Python before handing it to `json.dumps`,
which looked like the obvious thing to speed up at corpus scale. Measured, it is not worth
doing, for three reasons in increasing order of finality.

It is worth little: 46 ms against a 23 ms floor on a 33,000-entry plan — 2.0x, not the order
of magnitude orjson itself would give.

It is not reachable in pure Python: replacing the rebuilding walk with a cheaper
validate-only walk measured *slower* (26.5 ms to check, against ~23 ms to rebuild). Reaching
the floor means skipping the walk entirely and letting `json.dumps` raise — and that
silently diverges, because the C encoder coerces `{1: "a"}` to `{"1": "a"}` where orjson
raises without `OPT_NON_STR_KEYS`. Trading a silent encoding difference for 23 ms is exactly
the trade decision 3 exists to refuse.

And it is worth nothing here anyway, per the correction in decision 3: nothing on the run
path calls `orjson.dumps`.

## Consequences

The offline suite runs on `cp314t` with the GIL pinned off, and grew by 25 tests that
assert the interpreter and the stand-ins, plus 25 more that assert extraction's ordering,
its lxml quarantine and its accounting under threads. Every documented command carries
`PYTHON_GIL=0`; without it the runtime refuses to start.

`compat/` is now a small surface this project maintains that it did not write and does not
want. It is covered by tests written against the upstream behaviour rather than against the
implementation, so the day each dependency ships a free-threaded wheel, deleting a
directory and a `[tool.uv.sources]` entry is the whole migration.

## Still open

- **The extraction speedup is 2.1x, not 4x.** Four workers on a 15-CPU host is a poor
  return, and the ceiling is pdfminer's shared intern tables rather than anything here. The
  fix is a parser that does not have them; `pypdfium2` is the candidate worth measuring,
  because its bindings are ctypes over a bundled library rather than an extension module,
  so the wheel-availability problem this ADR is about does not apply to it.
- **The worker ceiling is one measurement on one machine.** ADR 0005 calls this shape a
  measured constant and puts it in the measurement store; `_WORKER_CEILING` is a literal in
  `extraction.py` because there is one measurement. The knee moves with the host and with
  the corpus.
- **Nothing else is threaded.** Verification across a staging tree and the per-item work
  inside a batch are still serial, and both are embarrassingly parallel.
- **lxml and SQLAlchemy run undeclared.** No longer bound by there being no threads —
  bound now by extraction routing lxml items away from the pool, and by nothing else being
  threaded. Revisit when lxml 7 leaves beta, and drop `PYTHON_GIL=0` the moment both
  declare support.
- **Claude, Cohere and Llama cannot be token-counted.** A run pinned to one of them fails
  loudly at the counting step rather than silently mis-billing. If one of those becomes the
  pinned model before tokenizers ships a wheel, the answer is a real tokenizer, not a
  guess.
- **The stand-ins are pinned to the upstream versions they satisfy.** `compat/orjson`
  declares `3.12.0` because that is the floor LangGraph's SDK resolves against, not because
  it implements orjson 3.12.0. A future dependency raising its floor needs the number here
  raised by hand.
