# 04 — Operators, families, admission

An operator family is a three-part contract:

1. `family.yml` — `family`, `revision`, `allowed_features` (closed config slots)
2. `family-context.yml` — the ten-field semantic card
3. a pinned trusted runner in `FAMILY_RUNNERS`, all in `shakespeare/runners.py`

Registration rejects any spec whose entrypoint is not its family's trusted runner.

Rendered packages are declarative: no generated callable. Behaviour lives in the runner,
which dispatches a named `operation` against a closed allowlist of vetted functions.

## Families

| Family | Side effects | Risk |
| --- | --- | --- |
| `readonly_scan` | none | low |
| `content_extract` | workspace artifacts | low |
| `pure_transform` | none | low |
| `filesystem_mutation` | writes under staging or output root | high |

## Admission

A subagent may request an operator. **Variant** requests are declarative over an existing
runner operation and can be admitted automatically. **Behaviour** requests need a runner
operation that does not exist; no model can satisfy them and they become human backlog.

Risk is computed, never declared: side effects -> high, dependencies -> medium, else low.
Auto-admission requires disposition AUTO_ADMIT, no findings, all four test tiers green, a
reproducible second render, a clean dependency policy, and a `pure_transform` or
`readonly_scan` family. Anything else escalates to a human via `interrupt()`.
