# ADR 0001 — What live runs found that the offline suite could not

**Status:** accepted · **Date:** 2026-08-24

## Context

The offline suite was comprehensive — 370 tests, every plan obligation covered — and every
model touchpoint was faked. Ten live runs against `openrouter/openai/gpt-5-mini` over the
fixture tree found nine defects it could not have found, at a total cost of about $0.45.

The pattern is worth naming: **fakes are written by the person who wrote the contract.**
A fake agent binds `scanned` from `items` because the author knew that was the wire. A
real model does not know, so it guesses — and every guess lands on a place where the
contract was underspecified rather than wrong.

## What the failures had in common

Seven of the nine were the same shape: **an interface the system knew and the model could
not.**

- Operators declared a family, a risk and side effects, but not their arguments. The model
  invented `path`, `dir`, `out_dir`, `output_format`.
- Operators declared no outputs either, so the model could not wire one into the next and
  bound `planned` from `collide.planned` instead of `resolutions`.
- `inputs` and `bindings` sit adjacent and mean adjacent things; the model wrote a mapping
  into `inputs`.
- `NamingSpec` rejected the commentary keys a model naturally adds, without saying which.
- `aliases` accepted variant→canonical; the model wrote canonical→[variants], which is at
  least as natural a reading of the word.
- A date field asked for `%Y%m` and then rejected `202402` — a contract arguing with
  itself.
- An extension supplied as `pdf` rather than `.pdf` produced `...po-88120pdf`.

The remaining two were recovery defects: a refused composition and a malformed model
response each killed the whole run instead of becoming a failed attempt the planner could
review, and in the first case the reason was not journaled at all — so the run died of
something nobody could see.

## Decisions

1. **An operator declares its arguments and its outputs**, and both are given to the
   subagent. A catalog that says what may be called but not how is not a contract.
2. **Be forgiving at the model boundary, strict at the trust boundary.** Coerce a mapping
   in `inputs`, a bare extension, an already-formatted date, an inverted alias map. Do not
   coerce anything that touches authority: the operator catalog, the config catalog, the
   write boundary and the path guard stay exact.
3. **Every refusal is a journaled attempt.** A model that returns something unusable is
   the case the attempt loop exists for, and the reason has to reach both the journal and
   the planner — a planner told only `operator_failed` reruns into the same wall.
4. **A rejection says what was wrong.** Naming the offending keys turned a wasted rerun
   into a corrected one.
5. **Guard rules distinguish configuration from data.** The same conflation cost two
   separate fixes: a path separator inside an item payload, then a vendor name as an alias
   key. Argument *names* must look like argument names; nested keys and values are data.

## Consequences

Ten runs took the workflow from failing in the first stage to every stage accepting on the
first attempt, committing a mirrored tree with balanced accounting. The offline suite grew
by 26 tests, each pinning a specific live failure.

The live lane is not optional coverage. It is the only thing that tests the interfaces
between the system and a model that has not read its source.

## Still open

- **Per-item transcription does not scale.** `field_resolution` reports values for every
  item in one response. That is fine for tens of files and will not hold for thousands: it
  is bounded by the output token limit, not by the budget. Chunking that stage is the next
  real design question.
- **Field confidences are reported but uncalibrated.** The floor is enforced; whether a
  model's 0.8 means anything is unmeasured. The canary lane is where that gets watched.
