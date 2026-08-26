# ADR 0003 — Scheduling by measurement, and the end of the blind retry

**Status:** accepted · **Date:** 2026-08-26

## Context

ADR 0001 left one thing open: *"Per-item transcription does not scale. Chunking that stage
is the next real design question."* This is the answer, and the seven runs it took to get
there.

The starting position was a capability that decided its own slicing every round — it called
`batch.window`, threaded its own progress, and judged how much it could answer at once. A
sixty-invoice run spent thirteen of twenty-one rounds failing to do that arithmetic.
Framework §10 puts scheduling in the runtime, not in the work.

Seven live runs over sixty generated invoices, about $0.35 in total. The corpus was built
deliberately non-uniform — twenty one-line receipts, twenty ordinary invoices, twenty
forty-line itemised statements — because a uniform corpus cannot exercise an adaptive
scheduler: every batch would measure the same thing.

## Decisions

### 1. Scheduling is an operator the runtime calls, and only when the work does not fit

It stays an operator, so the decision is verified, journalled and traced like any other.
No capability lists it in its catalog, so none can schedule itself. A capability that
declares no `cost_per_item` is never divided: collision resolution and plan assembly need
the whole set, and say so by not measuring.

### 2. The batch size is measured, never fixed

A fixed size is only right when every item costs the same. `cost_per_item` is a starting
estimate that measurement replaces: each batch reports what it actually spent, and the next
is sized from that. The adjustment is deliberately asymmetric — an estimate never falls
below what was just measured, a truncation proves the true cost is at least the ceiling
spread across that batch and caps every later one, and growth is capped so a cheap batch
cannot undo the caution an expensive one earned. A batch that was cut off is re-sized and
retried rather than lost, bounded because each attempt is billed.

### 3. Batching must be invisible to everything downstream

Three separate defects were the same mistake. The working set was replaced by each batch
and never restored, so a gate judged the last thirty files and reported *"All 30
inventoried files were processed"* on a sixty-file run. An operator's output replaces the
key it writes — right within a batch, wrong across them — so the second batch's extractions
erased the first's. And a capability shown thirty items beside an artifact summary saying
sixty honestly reported itself incomplete, so no batch ever finished.

A capability may be asked in pieces; nothing after it should be able to tell. The whole set
is restored on every exit, anything keyed by `item_id` is merged by item, and the context
carries `batch_number`, `batch_remaining` and `batch_total` so `sufficient` unambiguously
means *this batch is answered*.

### 4. A capability that must read the content is shown the content

Describing rather than handing over is right for a capability whose components do the
reading: it binds `items` by name and the operator gets the real list, so counts are all
the model needs. It is exactly wrong for one that must do the reading itself. Three
attempts died saying so: *"the available context provides only aggregate item and
extraction counts, not the individual item IDs, paths, extensions, or extracted invoice
text."*

A capability handed a batch receives that batch, and the per-item evidence belonging to it,
in full. The batch was sized to fit one response; this is what it was sized for.

### 5. Evidence outranks a round's account of itself

A round whose components all failed still published its artifact. An empty `FileInventory`
satisfied a deterministic gate, and the next capability spent three attempts extracting
text from nothing. A round now publishes, ends a capability, and makes an outcome
sufficient only if its components actually succeeded.

This one had been latent since before the batching work. It stayed invisible because the
gate was already reading a batch as the whole inventory — an empty artifact and a partial
one looked alike.

### 6. A retry is told what was rejected

**This is the finding that matters most.** Five runs failed a goal three times each, and
every attempt began identically: the same request, and no idea what the gate had objected
to. The verdict reached telemetry and the audit log — everywhere except the one place that
could act on it.

The gate's failed checks, missing evidence and rationale now reach the next attempt as
`previous_attempt`, shown in full rather than summarised, because describing the shape of a
diagnosis informs nobody. Two goals in the successful run failed once and passed on the
retry. Neither would have recovered blind.

### 7. A retry keeps the work it did

Scheduling divided the whole set every attempt, so a run that resolved fifty-nine of sixty
items restarted at sixty:

    attempt 1:  60 -> 46 -> 23
    attempt 2:  60 -> 46 -> 24 -> 1
    attempt 3:  60 -> 46 -> 22 -> 6

It now divides what is outstanding, judged by what *this* capability's own components have
produced. Never a shared record: one capability's per-item output says nothing about
whether another has done its work, and a global "done" would let each inherit the other's
progress and skip its own.

### 8. Staging is the first phase of the commit, not part of it

The `reviewed` goal asks whether the staged tree matches the plan, and staging happened
after the control loop returned — so the only goal that could observe a staged tree ran
before one existed. Two-phase commit is stage, verify, move; the verify was fenced off from
the stage.

The loop now announces each satisfied goal to the runtime, which stages the moment a plan
is in the context. The announcement carries no goal name: the loop does not know which goal
produces a plan or which one reviews it, and acting on the news stays with the runtime,
because acting on it means writing.

### 9. A refusal that reaches a model should end the mistake, not name it

The catalog lists what an operator produces, and a model mirrored that list into `bindings`
as though outputs had to be declared. `binding directories=directories has no resolved
source` was true and useless, and it cost a goal. A binding failure now says the binding
names an output rather than an input, or lists what is actually bindable there — with
runtime plumbing excluded, since offering `config` and `operation` would invite the next
mistake.

## Consequences

The seventh run committed sixty of sixty files, every name verified against what the
generator wrote into the PDF rather than against the plan, source tree untouched, for
$0.08 and thirty model calls. Scheduling adapted within the run — `acquire` took batches of
thirty, `resolve` grew from fourteen to twenty-three as measured cost came in — and two
goals recovered on a second attempt they were given a reason for.

The offline suite grew by 37 tests. Each of the nine decisions above is pinned by tests
that fail without its fix; that was checked by reverting each fix in turn, not assumed.

## What this says about the offline suite

ADR 0001 concluded that fakes are written by the person who wrote the contract. This round
sharpens it: **fakes also succeed on the first try.** Every defect here lived in a path the
offline suite never walked — the second batch, the second attempt, the round after a
failure. A fake agent returns the right answer immediately, so partial progress, recovery
and retry were all structurally untested while looking well covered.

The offline suite tests what the system does. The live lane tests what it does *next*.

## Still open

- **Field confidences remain uncalibrated.** Unchanged from ADR 0001.
- **`max_goal_attempts` is three, chosen rather than measured.** Now that a retry is
  informed, the right number may be different, and it may differ per goal.
- **Batch cost is measured per capability, not per item.** A corpus where one item is a
  hundred times another still gets a batch sized for the average. The truncation backoff
  covers it, at the price of one wasted call.
