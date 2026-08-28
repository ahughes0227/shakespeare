"""Plan assembly and deterministic obligation checks.

`plan.assemble` is where balanced accounting is enforced: every scanned item leaves with
exactly one terminal state, or the plan is refused.  An item that did not resolve is
quarantined with a reason — it is never given a guessed name.
"""

from __future__ import annotations

from typing import Any

from ...contracts import (
    ChangeAction,
    ChangeEntry,
    ChangePlan,
    Contract,
)


class RenameEntry(ChangeEntry):
    """A `ChangeEntry` for the rename workflow.

    The runtime only ever reads the base fields, which is what keeps accounting, preview,
    commit and undo generic across workflows.
    """

    target_relpath: str | None = None
    source_sha256: str | None = None


class ScannedItem(Contract):
    item_id: str
    relpath: str
    sha256: str
    media_type: str = "application/octet-stream"
    size_bytes: int = 0


class PlannedName(Contract):
    item_id: str
    directory: str
    name: str | None
    reason: str | None = None


class AssemblyError(ValueError):
    pass


def assemble_plan(
    *,
    run_id: str,
    workflow_id: str,
    workflow_digest: str,
    decision_digest: str,
    scanned: tuple[ScannedItem, ...],
    planned: tuple[PlannedName, ...] = (),
    skipped: tuple[dict[str, str], ...] = (),
    operator_versions: dict[str, str] | None = None,
    default_action: ChangeAction = ChangeAction.UNRESOLVED,
) -> ChangePlan:
    """Build a plan and refuse to return an unbalanced one."""
    by_item = {item.item_id: item for item in planned}
    unknown = set(by_item) - {item.item_id for item in scanned}
    if unknown:
        raise AssemblyError(f"planned names for unscanned items: {sorted(unknown)}")

    entries: list[ChangeEntry] = []
    for item in sorted(scanned, key=lambda value: value.relpath):
        decision = by_item.get(item.item_id)
        if decision is None:
            entries.append(
                RenameEntry(
                    item_id=item.item_id,
                    source_ref=item.relpath,
                    action=default_action,
                    reason="" if default_action is ChangeAction.UNCHANGED else "no_decision",
                    target_relpath=(
                        item.relpath if default_action is ChangeAction.UNCHANGED else None
                    ),
                    source_sha256=item.sha256,
                    digests={"source": item.sha256},
                )
            )
            continue
        if decision.name is None:
            entries.append(
                RenameEntry(
                    item_id=item.item_id,
                    source_ref=item.relpath,
                    action=ChangeAction.UNRESOLVED,
                    reason=decision.reason or "unresolved",
                    source_sha256=item.sha256,
                )
            )
            continue

        directory = "" if decision.directory in (".", "/") else decision.directory.strip("/")
        target = f"{directory}/{decision.name}" if directory else decision.name
        current = item.relpath
        action = ChangeAction.UNCHANGED if target == current else ChangeAction.CHANGED
        entries.append(
            RenameEntry(
                item_id=item.item_id,
                source_ref=current,
                action=action,
                target_relpath=target,
                source_sha256=item.sha256,
                digests={"source": item.sha256},
            )
        )

    for absent in sorted(skipped, key=lambda value: value.get("relpath", "")):
        # Unreadable inputs must appear in the plan or a user renaming a thousand files
        # gets nine hundred and ninety-seven outputs and no sign the rest existed.
        entries.append(
            RenameEntry(
                item_id=absent.get("relpath", ""),
                source_ref=absent.get("relpath", ""),
                action=ChangeAction.UNRESOLVED,
                reason=absent.get("reason", "skipped"),
                digests={"unreadable": "true"},
            )
        )

    plan = ChangePlan(
        run_id=run_id,
        workflow_id=workflow_id,
        workflow_digest=workflow_digest,
        decision_digest=decision_digest,
        operator_versions=operator_versions or {},
        entries=tuple(entries),
    )
    total = len(scanned) + len(skipped)
    if not plan.balanced(total):
        # Two different failures wear the same counts, so the message says which. Entries
        # short of the total means work went missing; the right number of entries sharing
        # ids means two things were treated as one, which is what a live run hit on five
        # byte-identical documents.
        distinct = len({entry.item_id for entry in plan.entries})
        cause = (
            f"{len(plan.entries) - distinct} share an item_id"
            if len(plan.entries) == total
            else "some items produced no entry"
        )
        raise AssemblyError(
            f"unbalanced plan: {len(plan.entries)} entries ({distinct} distinct) for "
            f"{len(scanned)} scanned and {len(skipped)} skipped items - {cause}"
        )
    return plan


# --------------------------------------------------------------------------------------
# Obligation checkers
# --------------------------------------------------------------------------------------


class WindowItem(Contract):
    item_id: str


def next_window(
    *,
    items: tuple[dict[str, Any], ...],
    completed: tuple[Any, ...] = (),
    window_size: int = 20,
) -> dict[str, Any]:
    """Take the next slice of work that has not been done yet.

    This is what lets a stage too large for one model response finish anyway. It is
    stateless: the caller passes what is already done, so windows can be replayed and the
    operator holds no hidden position.
    """
    # `completed` may arrive as ids or as the accumulated records themselves, since the
    # accumulator holds records and asking an agent to project them out is friction.
    done = {
        item.get("item_id", "") if isinstance(item, dict) else str(item)
        for item in completed
    }
    remaining = [item for item in items if item.get("item_id") not in done]
    window = remaining[: max(1, window_size)]
    return {
        "window": window,
        "window_size": len(window),
        "remaining": max(0, len(remaining) - len(window)),
        "completed_count": len(done),
        "exhausted": len(remaining) <= len(window),
    }


def plan_batch(
    *,
    remaining: tuple[dict[str, Any], ...],
    capacity: int,
    cost_per_item: int,
    observations: tuple[dict[str, Any], ...] = (),
    weights: tuple[int, ...] = (),
    reserve: float = 0.6,
    growth: float = 2.0,
    backoff: float = 0.5,
) -> dict[str, Any]:
    """Size the next batch of work from what the last ones actually cost.

    Returns `needed: False` when everything left already fits, and the caller then hands
    the whole set over undivided.

    Cost is measured against `weights` — how much material each item carries — rather
    than against a count. A one-line receipt and a forty-line statement are both one
    item, and sizing by count gives the batch of statements the average's budget and
    truncates it. `cost_per_item` remains the declared starting estimate; it is converted
    to a rate per unit of material using the mean weight of the set, so a capability still
    declares one measured number. Omit `weights` and every item weighs the same, which is
    the count behaviour exactly.

    The adjustment is deliberately asymmetric. Cost rises fast and is corrected slowly:
    an estimate never falls below what was just measured, a truncation proves the true
    cost is at least the whole ceiling divided across that batch's material, and any
    batch that hit a ceiling caps every later one. Growth is capped at `growth`x the last
    batch, so one cheap batch cannot undo the caution an expensive one earned.
    """
    usable = max(1, int(capacity * reserve))
    measured_weights = _weigh(remaining, weights)
    mean = sum(measured_weights) / max(1, len(measured_weights))
    # Per unit of material, so a heavy item is charged for what it carries.
    rate = max(1e-9, float(max(1, cost_per_item)) / max(1.0, mean))
    last_weight = 0.0
    #: A batch weight proven too large. Everything after it stays below.
    ceiling: float | None = None

    for observation in observations:
        weight = float(observation.get("weight") or observation.get("items") or 0)
        if weight <= 0:
            continue
        last_weight = weight
        spent = int(observation.get("completion_tokens") or 0)
        if observation.get("truncated"):
            # It ran out of room, so the real rate is at least the ceiling spread over
            # the material that batch carried — more than whatever we had estimated.
            rate = max(rate, capacity / weight)
            ceiling = weight if ceiling is None else min(ceiling, weight)
        elif spent > 0:
            observed = spent / weight
            # Half the weight on history, but never estimate below what was just seen.
            rate = max((rate + observed) / 2, observed)
        if observation.get("failed"):
            ceiling = weight if ceiling is None else min(ceiling, weight)

    allowance = usable / rate
    if ceiling is not None:
        allowance = min(allowance, max(1.0, ceiling * backoff))
    if last_weight:
        allowance = min(allowance, max(1.0, last_weight * growth))

    size, carried = 0, 0.0
    for weight in measured_weights:
        # Always take one: a single item over the allowance still has to be attempted,
        # and the truncation backoff is what handles it.
        if size and carried + weight > allowance:
            break
        size, carried = size + 1, carried + weight

    return {
        "needed": size < len(remaining),
        "batch": list(remaining[:size]),
        "batch_size": size,
        "batch_weight": int(carried),
        "remaining_count": max(0, len(remaining) - size),
        "estimate": int(round(carried * rate)),
    }


def _weigh(items: tuple[dict[str, Any], ...], weights: tuple[int, ...]) -> list[float]:
    """One weight per item, defaulting to equal weight when none was measured."""
    if len(weights) == len(items) and any(weights):
        return [float(max(1, weight)) for weight in weights]
    return [1.0] * len(items)
