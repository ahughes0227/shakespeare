# ADR 0002 — Converge on the LASI framework

**Status:** accepted · **Date:** 2026-08-24 · **Supersedes parts of** 00–05

## Context

Shakespeare was built as an ordered spine of stages. Reading the LASI framework document
showed that is the wrong control model in four specific ways, and the fourth had just
caused a live failure.

| Framework | As built |
| --- | --- |
| §3 A workflow is a graph of goals and dependencies, defining what must become true | An ordered spine of stages — a procedure |
| §12 Human stage names need not be first-class execution primitives | Stage names *were* the execution primitive |
| §6 The planner asks what it needs next and selects a capability | The planner walked a fixed spine |
| §8 Adaptive organization belongs inside the bounded capability | A domain composed once, never saw results, could not adapt |

The last one bit: a sixty-invoice run aborted with "40 items were quarantined but 20 of 60
remain unaccounted". The capability reached its own limit and had no way to organize
around it. The first response was to add windowing at the runtime level through a
CONTINUE verdict — which §8 says is the wrong layer. That decomposition belongs to the
capability.

## Decision

Converge fully, including first-class artifacts.

- **Workflow** becomes a graph of **goals** and dependencies. A goal states what must
  become true, never how.
- **Gates** evaluate whether the available artifacts sufficiently satisfy a goal.
  Deterministic where possible, semantic only where judgment is required, hybrid where
  both apply.
- **Capabilities** replace domains: a bounded, goal-directed set of components that owns
  its own decomposition. A capability may run components, observe the artifacts they
  produce, and decide whether more work is needed — within a declared round bound.
- **Components** are the former operators, unchanged in contract.
- **Artifacts** become the evidence layer. Progress gates on artifact presence and
  quality rather than on stage completion.
- **The planner** selects which capability answers the next open goal, rather than
  following an order.

## What this keeps

The parts that already matched the framework, and the transactional guarantees that are
Shakespeare's own: components with explicit contracts, deterministic gates preferred over
semantic, containment through bounded catalogs, computed risk on admission, two-phase
commit, balanced accounting, journalled provenance, and replay.

## What this costs

The stage packages, the spine type-check, the attempt loop and the runtime-level
windowing are all replaced. Human-facing stage names survive as labels on goals for
dashboards and audit, per §12, but stop being execution primitives.
