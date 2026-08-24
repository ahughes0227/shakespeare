"""Trusted family runners.

Generated operator packages are declarative and contain no callable.  All behaviour
reaches the runtime through exactly one runner per family, and each runner dispatches a
named `operation` against a closed allowlist of vetted functions.

This is what makes it safe for a subagent to request an operator: a request can select
vetted behaviour and configure it, but adding a *new* operation requires a human to edit
the allowlists below and pass the family test tiers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import ChangePlan, OperatorFamily, ReversalRecord
from .operators import extraction, filesystem, mutation, naming, planning, text

Operation = Callable[[dict[str, Any], Path], dict[str, Any]]


class RunnerError(RuntimeError):
    pass


def _cfg(arguments: dict[str, Any], group: str, key: str, default: Any = None) -> Any:
    """Read a value from the composed Hydra config, allowing a direct override.

    The executor passes the whole composed mapping under `config`; a caller may still
    pass a flat key, which is what keeps operators unit-testable without a composition.
    """
    if key in arguments:
        return arguments[key]
    group_config = (arguments.get("config") or {}).get(group) or {}
    return group_config.get(key, default)


def _dispatch(
    arguments: dict[str, Any], workspace: Path, allowed: dict[str, Operation]
) -> dict[str, Any]:
    payload = dict(arguments)
    operation = payload.pop("operation", None)
    if operation not in allowed:
        raise RunnerError(
            f"unsupported operation: {operation!r}. "
            f"Vetted operations for this family: {sorted(allowed)}"
        )
    return allowed[operation](payload, workspace)


# --------------------------------------------------------------------------------------
# readonly_scan
# --------------------------------------------------------------------------------------


def _walk(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    items, skipped = filesystem.scan(
        Path(arguments["root"]),
        depth_limit=int(arguments.get("depth_limit", 32)),
        include_hidden=bool(arguments.get("include_hidden", False)),
    )
    return {
        "items": [item.model_dump(mode="json") for item in items],
        "skipped": list(skipped),
        "count": len(items),
    }


def _directories(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return {"directories": list(filesystem.directories(Path(arguments["root"])))}


# --------------------------------------------------------------------------------------
# content_extract
# --------------------------------------------------------------------------------------


def _extract(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    options = extraction.ExtractOptions(
        page_limit=int(_cfg(arguments, "extract", "page_limit", 20)),
        char_limit=int(_cfg(arguments, "extract", "char_limit", 200_000)),
    )
    results = [
        extraction.extract(
            item_id=item["item_id"],
            path=Path(arguments["root"]) / item["relpath"],
            media_type=item.get("media_type", "application/octet-stream"),
            backend=extraction.Backend(_cfg(arguments, "extract", "backend", "auto_chain")),
            options=options,
        )
        for item in arguments["items"]
    ]
    return {
        "items": [item.model_dump(mode="json") for item in results],
        "usable": sum(1 for item in results if item.usable),
        "unavailable": sum(1 for item in results if not item.usable),
    }


# --------------------------------------------------------------------------------------
# pure_transform
# --------------------------------------------------------------------------------------


def _render_template(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    fields = tuple(naming.FieldDecl.model_validate(item) for item in arguments["fields"])
    naming_config = (arguments.get("config") or {}).get("naming") or {}
    policy = naming.NamePolicy.model_validate(
        arguments.get(
            "policy",
            {
                key: value
                for key, value in naming_config.items()
                if key in naming.NamePolicy.model_fields
            },
        )
    )
    results = [
        naming.render(
            item_id=item["item_id"],
            template=arguments["template"],
            fields=fields,
            values=item.get("values", {}),
            policy=policy,
            extension=item.get("extension", ""),
            sequence=item.get("sequence"),
        )
        for item in arguments["items"]
    ]
    return {"results": [item.model_dump(mode="json") for item in results]}


def _normalize(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return {
        "values": text.normalize(
            arguments["values"],
            collapse_whitespace=bool(arguments.get("collapse_whitespace", True)),
            strip=bool(arguments.get("strip", True)),
            aliases=arguments.get("aliases"),
            case=str(arguments.get("case", "preserve")),
        )
    }


def _collision_resolve(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    candidates = tuple(naming.Candidate.model_validate(item) for item in arguments["candidates"])
    resolutions = naming.resolve_collisions(
        candidates, naming.CollisionPolicy(_cfg(arguments, "collision", "policy", "suffix_n"))
    )
    return {"resolutions": [item.model_dump(mode="json") for item in resolutions]}


def _plan_assemble(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    plan = planning.assemble_plan(
        run_id=arguments["run_id"],
        workflow_id=arguments["workflow_id"],
        workflow_digest=arguments["workflow_digest"],
        decision_digest=arguments["decision_digest"],
        scanned=tuple(planning.ScannedItem.model_validate(i) for i in arguments["scanned"]),
        planned=tuple(planning.PlannedName.model_validate(i) for i in arguments["planned"]),
        operator_versions=arguments.get("operator_versions"),
    )
    return {"plan": plan.model_dump(mode="json")}


def _obligation_check(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    result = planning.run_check(
        arguments["obligation_id"], arguments["check"], arguments.get("payload", {})
    )
    return result.model_dump(mode="json")


# --------------------------------------------------------------------------------------
# filesystem_mutation
# --------------------------------------------------------------------------------------


def _stage_write(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    plan = ChangePlan.model_validate(arguments["plan"])
    reversals = mutation.stage_plan(
        plan=plan,
        input_root=Path(arguments["input_root"]),
        staging_root=Path(arguments["staging_root"]),
        quarantine_dirname=_cfg(arguments, "write", "quarantine_dirname", "_unresolved"),
    )
    return {"reversals": [item.model_dump(mode="json") for item in reversals]}


def _verify_staging(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return dict(
        mutation.verify_tree(
            plan=ChangePlan.model_validate(arguments["plan"]),
            staging_root=Path(arguments["staging_root"]),
        )
    )


def _atomic_move(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    record = mutation.commit(
        staging_root=Path(arguments["staging_root"]),
        output_root=Path(arguments["output_root"]),
    )
    return {"reversal": record.model_dump(mode="json")}


def _journal_reverse(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    record = ReversalRecord.model_validate(arguments["reversal"])
    mutation.reverse(record)
    return {"reversed": record.mutation_id}


def _discard(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    mutation.discard(Path(arguments["staging_root"]))
    return {"discarded": arguments["staging_root"]}


# --------------------------------------------------------------------------------------
# The closed allowlists
# --------------------------------------------------------------------------------------

_ALLOWLISTS: dict[OperatorFamily, dict[str, Operation]] = {
    OperatorFamily.READONLY_SCAN: {
        "walk": _walk,
        "directories": _directories,
        "verify_staging": _verify_staging,
    },
    OperatorFamily.CONTENT_EXTRACT: {
        "extract": _extract,
    },
    OperatorFamily.PURE_TRANSFORM: {
        "render_template": _render_template,
        "normalize": _normalize,
        "collision_resolve": _collision_resolve,
        "plan_assemble": _plan_assemble,
        "obligation_check": _obligation_check,
    },
    OperatorFamily.FILESYSTEM_MUTATION: {
        "stage_write": _stage_write,
        "atomic_move": _atomic_move,
        "journal_reverse": _journal_reverse,
        "discard": _discard,
    },
}


def allowlist(family: OperatorFamily | str) -> frozenset[str]:
    """The vetted operations for a family.  Used by the template's functional test tier."""
    return frozenset(_ALLOWLISTS[OperatorFamily(family)])


def readonly_scan(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return _dispatch(arguments, workspace, _ALLOWLISTS[OperatorFamily.READONLY_SCAN])


def content_extract(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return _dispatch(arguments, workspace, _ALLOWLISTS[OperatorFamily.CONTENT_EXTRACT])


def pure_transform(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return _dispatch(arguments, workspace, _ALLOWLISTS[OperatorFamily.PURE_TRANSFORM])


def filesystem_mutation(arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
    return _dispatch(arguments, workspace, _ALLOWLISTS[OperatorFamily.FILESYSTEM_MUTATION])
