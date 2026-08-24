# 02 — Stages and domains

A stage is a versioned package under `_stages/<name>/`:

- `stage.yml` — name, version, purpose, goal, input/output contract, `max_attempts`,
  domains, obligations, budget, side effects
- `stage-context.yml` — the ten-field semantic card
- `prompts/` — versioned prompt artifacts, pinned per domain
- `tests/`

A domain declares an immutable `scope`, whether it is `skippable`, its operator `catalog`,
its Hydra `config_groups`, and its pinned `prompt_version`. The planner issues a per-run
`DomainGoal` inside that scope. **A goal never widens the surface**: catalog and config
groups come from the package regardless of what the goal says.

Non-skippable domains cannot be planned away. Safety and commit-gating domains are
non-skippable.

A workflow (`_workflows/<id>/workflow.yml`) is an ordered spine of `stage@version` refs.
Registration type-checks the spine: contracts must line up, versions must resolve,
catalog operators must be registered, config groups must exist.
