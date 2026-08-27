# Glossary

**Admission** — the deterministic decision to register a requested operator. Risk is
computed; low-risk declarative variants may be auto-approved, anything else escalates.

**Attempt** — one pass through a stage. Bounded by `max_attempts` in the stage package.

**Composition** — the entire output surface of a domain subagent: ordered `Invocation`s.

**Domain** — a type of work available within a stage; one bounded subagent each.
Declares an immutable `scope`, a `catalog`, `config_groups`, and `skippable`.

**DomainGoal** — the planner's per-run, verifiable goal for an activated domain. Sits
inside the domain's scope and never widens its surface.

**Invocation** — one operator call: operator name, Hydra selections, parameters, inputs.
A node in the stage DAG.

**Obligation** — a deterministic check that must hold at a stage's output. Checked by
`check.assert`, never by an agent. A hard gate.

**Operator** — atomic, config-driven, registered, versioned code. One per verb; backends
chosen by configuration.

**Operator family** — a closed template kind defining lifecycle, allowed configuration
slots, risk model, and a pinned trusted runner.

**Planner** — selects a prebuilt workflow, issues `DomainGoal`s, reviews each stage,
reruns within bounds, and auto-approves low-risk operator requests. Never builds a
workflow, stage, domain, or operator.

**PromptArtifact** — a versioned, digested prompt compiled offline by DSPy. Pinned per
domain; feeds the workflow digest.

**Stage** — an atomic unit of work with typed inputs and outputs, versioned and reusable.
A new stage is required wherever outputs must be transformed or verified by a model.

**StageVerdict** — the planner's judgment on a completed attempt: accept, rerun, or abort.

**Staging tree** — where Execute writes. Nothing user-visible changes until commit.

**TelemetryEnvelope** — the only shape permitted to reach an exporter. Digests and
metadata only.

**Trusted runner** — the vetted entrypoint for an operator family. Generated packages
contain no callable; the runner dispatches a named operation from a closed allowlist.

**Workflow** — an ordered spine of versioned stage refs. Programmer-authored only.
