# ADR 0004 — The model is not the data path

**Status:** accepted · **Date:** 2026-08-26

## Context

Seven live runs and three calibration runs went into making a sixty-invoice rename work.
Most of that effort — adaptive batch sizing, weight-based cost, truncation backoff,
outstanding-only scheduling, batch accounting — exists to solve one problem: sixty items'
worth of field values have to fit inside one model response.

They do not have to. That constraint is not in the task; it is in the shape I gave the
task.

The model's only irreducible job here is reading a document: turning a page of invoice text
into four field values. Everything after that — formatting a date, sanitising a vendor
name, resolving a collision, assembling a plan — is arithmetic on those values, and every
piece of it is already deterministic code. But the values have no way to reach that code
except through the model's own response, because nothing in the system persists a per-item
record. So the model is not just the reader; it is the transport. Sixty documents means
sixty records in one response, and every mechanism above exists to make that survivable.

Making it survivable is what stopped anyone asking why it was necessary.

## Why nothing noticed

The system had the evidence and no way to act on it.

**The measurement was computed and discarded.** `674 tokens/item × 60 items` against a
16,384-token ceiling is a proof that the model is in the data path. The scheduler computes
it, uses it privately to size a batch, and shows it to nobody.

**The planner learns the shape too late to use it.** Routing happens before anything scans
the input, so the one moment the planner may say `supported: false` is the one moment it
does not know whether there are three documents or sixty thousand. By the time `survey`
reports — and it does report; the judge quoted "All 60 inventoried files" — the workflow is
already chosen and cannot be revisited.

**No return type can carry an objection.** `select_goal` returns a goal id and a sentence.
`select_capability` returns a capability id. `judge` returns a boolean and a rationale.
There is no shape for "the plan is fine and the architecture is wrong", so a planner that
reached that conclusion could only write it into a gate rationale, where it would be filed
as a verdict and change nothing.

**The capability charter pre-empts the question.** `resolve`'s standing goal is "Determine
each item's field values *and render its name*." A capability whose remit includes
rendering will not conclude that rendering belongs in a loop over a table. The capabilities
are not too narrow; they are cut along the wrong axis — by position in a pipeline rather
than by nature of work.

Nobody in the system had both the whole picture and a channel to object. The planner had
the picture without the mechanism; each capability had the mechanism without the picture;
the human had both and is only ever consulted about operator admission.

## Decisions

### 1. A record store, and the model writes records rather than carrying them

A new component family persists one row per item to a Parquet table in the run workspace.
It is a mutation family — it writes — but its containment is narrower than the filesystem
one: it may only write the run's own record store, never the input or output trees.

Reading a document now ends in an append rather than in a value that must survive until the
next operator call. Consequences follow immediately, and none of them needed inventing:
progress is durable across attempts and across runs, a batch that fails loses only itself,
and the renderer reads the table instead of the response.

### 2. Rendering is deterministic work over a table

`transcribe` reads documents into records. Rendering runs from `record.read`, which is the
same `name.render` as before with its input coming from storage rather than from a model.
The model no longer sees a filename, a template, or a collision. Semantic reasoning happens
exactly once per document, which is the only place the task requires it.

### 3. The planner chooses the shape, from evidence

`named` now names two capabilities. `resolve` answers in-response and is right for a set
small enough to fit. `transcribe` reads to storage and is right for one that is not. The
planner already picks a capability per goal; it has simply never had a choice to make or
the facts to make it with.

So the facts go to it: the corpus size, the measured per-item cost, and the response
ceiling, alongside each candidate's declared cost model. This is the evidence that was
being computed and thrown away.

### 4. A capability and a planner can both raise an impediment

Two things a capability could say before: done, and not done. It can now say a third:
*this cannot be done this way, and here is what is in the way.* A planner can do the same
when no candidate fits the corpus it has been shown.

An impediment is a contract, not a sentence in a rationale — the runtime ends the run as
`escalated`, distinct from `aborted`, and the reason reaches the audit log and the operator
of the system rather than the next retry.

This is the missing half of the retry work in ADR 0003. That gave a rejected attempt the
reason it was rejected, which fixes a retry that could have worked. This covers the retry
that could not.

### 5. Capabilities stay narrow, and are recut by nature of work

Broadening them — a `coder`, an `explorer` — buys creativity this system does not want. A
capability's catalog is its authority surface, and the verifier, the write containment and
the path guard all bound it; a wide capability is a wide surface. The problem was never
breadth.

The axis is the fix. Cut by nature of work — read, transform, materialise — nothing in the
reader's charter mentions filenames, so "should this be deterministic?" becomes a question
the system can answer rather than one its own charter has already answered.

`transcribe` is that recut applied where it was costing the most. The rest follows when
there is a second workflow to justify it, not before.

### 6. Capabilities do not collaborate

They share a substrate and they escalate. Two capabilities cooperate when one writes a
record another reads, sequenced by the goal graph — a blackboard, not a conversation.

Direct messaging between capabilities was considered and rejected. Replay, workflow
digests, the audit DAG and `commit_planned` all depend on a run being reconstructible from
recorded decisions; a conversation between agents is state that lives nowhere and replays
differently every time. Everything wanted from collaboration is available upward, through
the planner, where it is recorded.

## Consequences

The scheduler stops being load-bearing. It still divides work — a hundred documents is
still more than one response should carry — but nothing depends on it being right, because
a batch that goes wrong costs one batch and not a run's progress. The adaptive sizing from
ADR 0003 keeps its value and loses its criticality.

`resolve` is not deleted. It is correct and cheaper for a set that fits, and keeping both
is what makes the planner's choice a real one rather than a rename.

## What this says about the earlier work

ADR 0003 concluded that fakes succeed on the first try, so the offline suite never walks
the second batch or the second attempt. This is the next layer of the same lesson: **a
mechanism that successfully works around a bad shape removes the pressure that would have
exposed it.** Every fix in ADR 0003 was correct, and together they made a design mistake
survivable enough to stop being visible.

The scheduler measured the exact number that proved the shape was wrong, and spent seven
runs using it to cope.

## Still open

- **Confidence calibration has a working instrument and no hard cases.** All 240 claims on
  the current corpus landed in one band at 100% accuracy, so the floor is unfalsified
  rather than validated. It needs a corpus with genuinely ambiguous documents.
- **The read/transform/materialise recut is applied to one goal**, not the workflow.
- **An impediment is raised and recorded, and nothing yet acts on it automatically.** A
  human reads it. Whether a planner should be allowed to re-route on one is a question this
  ADR deliberately does not answer.
