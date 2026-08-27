# 00 — Overview

Shakespeare edits files according to a user's prompt. It is a **transactional** program:
consistency matters more than creativity.

## Work breakdown

| Level | Unit | Authored by |
| --- | --- | --- |
| — | Planner — selects a workflow, issues goals, reviews stages, reruns | model, bounded |
| — | Human — approves high-risk admissions and the commit | person |
| 0 | Workflow — an ordered spine of versioned stages | programmer |
| 1 | Stage — atomic work, typed in/out, versioned, reusable | programmer |
| 2 | Domain — a *type* of work in a stage; one subagent each | programmer scopes, model composes |
| 3 | Operator — atomic config-driven code | programmer |

## Two rules that decide the design

**Stage boundary rule.** A new stage is required wherever outputs must be transformed or
verified by a large model.

**Composition rule.** A domain subagent compiles an operator configuration via Hydra; the
runtime executes it. A subagent never observes its own operator output — if that output
needs model interpretation, that is the next stage.

## Consequence

Determinism lives in operators, obligations, and the commit — never in a scripted agent
procedure. The programmer fixes the surface; the planner chooses which work to do and
judges whether it landed; the subagent chooses the composition within its bounds.
