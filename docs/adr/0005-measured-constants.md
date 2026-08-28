# ADR 0005 — Memory is measured constants, not a scratchpad

**Status:** accepted · **Date:** 2026-08-26

## Context

Three numbers decide how this system behaves, and none of them is measured.

- `cost_per_item`, declared in each capability manifest, seeds every scheduling decision.
- The confidence floor, declared in `configs/confidence/*.yaml`, decides which files get a
  name and which get quarantined.
- `max_goal_attempts`, a field default on the controller, decides when a run gives up.

ADR 0001 and ADR 0003 both closed with the same two entries still open: *"field confidences
remain uncalibrated"* and *"`max_goal_attempts` is three, chosen rather than measured."* They
stayed open for a structural reason rather than a scheduling one. The evidence that would
settle them is produced by every run and destroyed at the run boundary.

`plan_batch` measures the true cost per unit of material within a run and adapts to it —
ADR 0003 records `resolve` growing from batches of fourteen to twenty-three as real cost
came in — and then the process exits and the next run starts again from the number in the
manifest. `calibration.py` can derive the floor the evidence supports, but `calibrate`
printed a report and stored nothing, so a floor could only ever be measured over one corpus
in one sitting. Whether a retry ever recovered has been a fact of the audit log since before
anyone asked the question, and nothing ever asked it.

So the question was not whether to add memory. It was what memory is allowed to be here.

## Decisions

### 1. What is remembered is measurements, and what is used is constants

A knowledge graph, a vector store, or a scratchpad the planner writes to and reads back
would all fail the same test. Retrieval from them is approximate, and the central claim of
this system is that everything entering a decision path is typed, verified, digested and
replayable. `replay` refuses against a changed workflow digest precisely so that a run is
determined by its journal. A similarity query at run time is unverified, unpinned input
feeding a decision, and it would cost replay to buy convenience.

What actually needed remembering was never graph-shaped. It is a scalar under a composite
key: `(capability, model, prompt version) -> tokens per unit of material`. That is a
relation, and there is already SQLAlchemy under Alembic holding every other fact this
system knows.

### 2. Nothing reads the ledger during a run

This is the decision the rest depend on, and it is pinned by a test that seeds forty
observations screaming a different number and asserts the run uses the manifest's value
anyway.

A measured constant reaches a run by being written into the manifest or config that
declares it. That makes adopting one a versioned edit, visible in git, that a person made
and can revert — the same shape as promoting a prompt, and for the same reason. The
alternative, reading a seed live at run start, saves a manual step and makes the run's
behaviour depend on state that is not pinned by anything the journal records.

The consequence worth stating plainly: this memory cannot make a run worse. The worst it
can do is propose a bad number that a person declines to write down.

### 3. Observations, never aggregates

The ledger records what a batch spent and how much material it covered. It does not record
the rate, which is derived on read.

An aggregate cannot be re-derived under a new weighting, and cannot be invalidated when the
model behind it changes. Rows can. `resolved_model` is part of the identity rather than
metadata attached to it, for the reason `canary` exists: a provider can change what an
alias resolves to, and a cost measured under one model is not evidence about another.
Mixing models is refused rather than averaged.

### 4. Evidence is scoped to the configuration declared now

The model is part of a measurement's identity, and so are the capability's version and its
pinned prompt: all three change what a capability spends. Left unscoped, bumping a
capability to 1.1.0 would quietly propose 1.0.0's measured cost as 1.1.0's declared one —
the same mistake as averaging two models together, in a place that looks like bookkeeping
rather than like evidence.

What is set aside is reported rather than dropped quietly. A proposal built from a third of
the ledger looks identical to one built from all of it unless somebody says so.

### 5. A truncated batch is a bound, not a measurement

A batch cut off at the output ceiling never reported what it would have cost. It proved
the cost is at least what fitted. `plan_batch` has always treated it that way within a run;
the ledger now records the distinction so a derivation can too.

A batch that failed for any other reason is excluded entirely. It is evidence about the
capability, not about the arithmetic, and averaging it in would drag the estimate toward
whatever a failure happened to spend before dying.

### 6. The estimate leans high, because the two errors do not cost the same

A batch sized too large is cut off, and that call is billed, wasted and retried. A batch
sized too small is smaller than it needed to be, and every call in it still does work. So
the proposal sits at a high quantile of observed rates rather than at the middle — and not
at the maximum, which would let one pathological batch set every later one.

### 7. A proposal refuses to average away the thing it cannot represent

Cost is measured per unit of material; `cost_per_item` is that rate declared for an item of
average weight. Where item weight is uniform the distinction is invisible. Where it is not
— ADR 0003's remaining open item, a corpus of one-line receipts and forty-line statements —
no single per-item number describes both ends.

Rather than emit a confident average, a proposal whose observed item weights vary by more
than 3x reports that the rate is sound and the conversion to one item is the part to
distrust. Naming the limit is worth more than hiding it behind a number.

### 8. A floor is never promoted on evidence alone

Every other constant here has a cheap direction to be wrong in. A confidence floor does
not: too high quarantines files a person then renames by hand, too low produces a
confidently wrong name. `floor_proposal` therefore never returns `SUPPORTED`. It reports
the lowest floor reaching the requested precision, says how many claims that keeps, and
leaves the judgment where it belongs.

When no floor reaches the precision at all, it says so rather than proposing a higher one —
because at that point the claims are not worth anything and raising the floor will not fix
that.

### 9. A choice between shapes is measured like a constant is

The planner now decides between `resolve` and `transcribe` for the `named` goal, using the
corpus size and each candidate's *declared* per-item cost. That is the same arithmetic the
scheduler used to do privately, and it had the same gap: computed, shown to a model, and
never checked against what happened.

Only goals that several capabilities could answer are recorded. A goal with one candidate
was not chosen for, and a foregone conclusion in with the real decisions makes a shape look
reliable because most of its record was never in doubt.

The corpus size is the fact that was missing; everything else joins from tables the log
already had. What is reported is the goal's own gate result rather than the run's outcome —
a later goal failing says nothing about this pick — with the run's total cost shown beside
it and labelled as the run's, because a run's spend cannot honestly be attributed to one
goal's choice.

### 10. `max_goal_attempts` needed no new measurement at all

Whether an attempt ever recovered is already recorded in `stage_attempts` and
`stage_verdicts`. `measurements recovery` reads it: attempts by number, the deepest one
that ever satisfied a gate, and how much budget every run is free to spend past it.

The number remains chosen. What has changed is that choosing it is now an argument with
evidence rather than an argument with intuition.

## Consequences

Runs record what their batches cost and which shape was chosen for the one goal that has a
choice, `calibrate` keeps its claims instead of printing them, and `measurements propose`
says which declared constant the accumulated evidence supports. The offline suite grew by
43 tests.

The append-only trigger installer now covers only tables that exist. It is still derived
from the model list so it cannot drift from it, but a historical migration runs against a
database that has not yet grown the newer tables, and deriving from the models while
ignoring what is present would make every past migration fail the moment a table is added.

## Still open

- **No constant has actually been promoted from live evidence yet.** The machinery is
  tested offline and the thresholds — eight observations, two runs, the 0.8 quantile, the
  3x spread — are chosen rather than measured. This ADR replaces three guessed constants
  with a mechanism containing four more. The difference is that these govern a proposal a
  person reads, not a run that spends money.
- **Confidence is measured only under `calibrate`**, which needs a truth file. An ordinary
  run makes claims nobody checks, so the floor accumulates evidence only as fast as someone
  builds labelled corpora.
- **A shape choice is recorded but nothing scores it.** `measurements shapes` reports what
  followed each pick; deciding that one shape is simply better is left to a person, because
  the runs are not controlled — a corpus, a model and a prompt all vary underneath.
- **Per-item cost is still not measured per item.** Decision 6 detects the case and refuses
  to describe it; describing it would mean weighing each item's own cost, which is a
  per-item observation the runtime does not currently make.
