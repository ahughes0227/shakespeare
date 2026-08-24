# 03 — Compositions

A `Composition` is the entire output surface of a domain subagent: an ordered list of
`Invocation`s, each naming a registered operator, a Hydra `group=choice` selection with
bounded parameters, and inputs drawn from stage inputs or a prior invocation's output.

The subagent decides order and choices. The runtime resolves, validates and executes.
Failures are recorded as data; the planner reacts at the stage boundary, the subagent
never does.

Validation rejects: an operator outside the domain catalog, a group outside the domain's
config groups, an unknown `group=choice`, and any Hydra escape (`_target_`, `${`, `~`,
`+`, path separators, leading-underscore keys).

A subagent that lacks an operator may submit an `OperatorRequest`. See 04.
