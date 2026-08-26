"""Plan assembly and deterministic obligation checks.

`plan.assemble` is where balanced accounting is enforced: every scanned item leaves with
exactly one terminal state, or the plan is refused.  An item that did not resolve is
quarantined with a reason — it is never given a guessed name.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ..contracts import (
    ChangeAction,
    ChangeEntry,
    ChangePlan,
    Contract,
    ObligationResult,
    content_digest,
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
        raise AssemblyError(
            f"unbalanced plan: {len(plan.entries)} entries for {len(scanned)} scanned "
            f"and {len(skipped)} skipped items"
        )
    return plan


# --------------------------------------------------------------------------------------
# Obligation checkers
# --------------------------------------------------------------------------------------


class ObligationInput(Contract):
    obligation_id: str
    check: str
    payload: dict[str, Any] = Field(default_factory=dict)


def _result(obligation_id: str, passed: bool, **detail: Any) -> ObligationResult:
    return ObligationResult(obligation_id=obligation_id, passed=passed, detail=detail)


def check_balanced(obligation_id: str, payload: dict[str, Any]) -> ObligationResult:
    entries = payload.get("entries", [])
    scanned = int(payload.get("scanned", 0))
    ids = {entry["item_id"] for entry in entries}
    return _result(
        obligation_id,
        len(entries) == scanned and len(ids) == scanned,
        entries=len(entries),
        scanned=scanned,
        distinct=len(ids),
    )


def check_no_collisions(obligation_id: str, payload: dict[str, Any]) -> ObligationResult:
    targets = [
        entry["target_relpath"]
        for entry in payload.get("entries", [])
        if entry.get("target_relpath")
    ]
    lowered = [target.lower() for target in targets]
    duplicates = sorted({item for item in lowered if lowered.count(item) > 1})
    return _result(obligation_id, not duplicates, duplicates=duplicates[:20])


def check_resolved_or_quarantined(obligation_id: str, payload: dict[str, Any]) -> ObligationResult:
    """No item may be `changed` without a target, and none may be silently dropped."""
    offenders = [
        entry["item_id"]
        for entry in payload.get("entries", [])
        if entry["action"] == ChangeAction.CHANGED and not entry.get("target_relpath")
    ]
    missing_reason = [
        entry["item_id"]
        for entry in payload.get("entries", [])
        if entry["action"] == ChangeAction.UNRESOLVED and not entry.get("reason")
    ]
    return _result(
        obligation_id,
        not offenders and not missing_reason,
        without_target=offenders[:20],
        without_reason=missing_reason[:20],
    )


def check_every_item_has_text_or_reason(
    obligation_id: str, payload: dict[str, Any]
) -> ObligationResult:
    offenders = [
        item["item_id"]
        for item in payload.get("extractions", [])
        if not item.get("text") and not item.get("unavailable_reason")
    ]
    return _result(obligation_id, not offenders, without_text_or_reason=offenders[:20])


def check_resolution_accounted(obligation_id: str, payload: dict[str, Any]) -> ObligationResult:
    """Every inventoried item must come out of resolution either named or quarantined.

    Without this an item could silently vanish between the renderer and the plan, and the
    balance check downstream would never see it because it was never offered.
    """
    inventory = len(payload.get("items", []))
    named = len(payload.get("candidates", []))
    quarantined = len(payload.get("unrendered", []))
    return _result(
        obligation_id,
        named + quarantined == inventory,
        items=inventory,
        named=named,
        quarantined=quarantined,
    )


def check_spec_frozen(obligation_id: str, payload: dict[str, Any]) -> ObligationResult:
    """The frozen convention must match the digest recorded when it was frozen."""
    spec = payload.get("spec")
    declared = payload.get("digest")
    actual = content_digest(spec) if spec is not None else None
    return _result(
        obligation_id,
        bool(declared) and declared == actual,
        declared=declared,
        actual=actual,
    )


def check_rendered_mechanically(obligation_id: str, payload: dict[str, Any]) -> ObligationResult:
    """Every named item must have come out of the renderer.

    The structural guarantee is that `plan.assemble` refuses a hand-written `planned`
    parameter. This is the observable counterpart: a plan entry that names a file must
    correspond to a render candidate, so a name cannot appear from anywhere else.

    Collision resolution may rewrite a candidate's name, so the check is on identity
    rather than on the string — comparing names would fail every legitimate suffix.
    """
    rendered = {item["item_id"] for item in payload.get("candidates", [])}
    unaccounted = [
        entry["item_id"]
        for entry in payload.get("entries", [])
        if entry.get("target_relpath") and entry["item_id"] not in rendered
    ]
    return _result(
        obligation_id,
        not unaccounted,
        unaccounted=unaccounted[:20],
        rendered=len(rendered),
    )


def check_structure_mirrors(obligation_id: str, payload: dict[str, Any]) -> ObligationResult:
    expected = sorted(payload.get("expected_dirs", []))
    actual = sorted(payload.get("actual_dirs", []))
    return _result(
        obligation_id,
        expected == actual,
        missing=[item for item in expected if item not in actual][:20],
        unexpected=[item for item in actual if item not in expected][:20],
    )


#: Closed allowlist.  An obligation names one of these; it can never name arbitrary code.
CHECKS = {
    "balanced": check_balanced,
    "no_collisions": check_no_collisions,
    "resolved_or_quarantined": check_resolved_or_quarantined,
    "every_item_has_text_or_reason": check_every_item_has_text_or_reason,
    "resolution_accounted": check_resolution_accounted,
    "spec_frozen": check_spec_frozen,
    "rendered_mechanically": check_rendered_mechanically,
    "structure_mirrors": check_structure_mirrors,
}


#: The payload keys each checker needs.  A missing key means the evidence was never
#: produced, and an unevaluated obligation must never read as a satisfied one.
CHECK_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "balanced": ("entries", "scanned"),
    "no_collisions": ("entries",),
    "resolved_or_quarantined": ("entries",),
    "every_item_has_text_or_reason": ("extractions",),
    "resolution_accounted": ("items", "candidates", "unrendered"),
    "spec_frozen": ("spec", "digest"),
    "rendered_mechanically": ("entries", "candidates"),
    "structure_mirrors": ("expected_dirs", "actual_dirs"),
}


def missing_evidence(check: str, payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(key for key in CHECK_REQUIREMENTS.get(check, ()) if key not in payload)


def run_check(obligation_id: str, check: str, payload: dict[str, Any]) -> ObligationResult:
    try:
        checker = CHECKS[check]
    except KeyError as exc:
        raise AssemblyError(f"unknown obligation check: {check}") from exc
    absent = missing_evidence(check, payload)
    if absent:
        return _result(obligation_id, False, missing_evidence=list(absent))
    return checker(obligation_id, payload)


# --------------------------------------------------------------------------------------
# Scheduling
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
    reserve: float = 0.6,
    growth: float = 2.0,
    backoff: float = 0.5,
) -> dict[str, Any]:
    """Size the next batch of work from what the last ones actually cost.

    Returns `needed: False` when everything left already fits, and the caller then hands
    the whole set over undivided.

    A fixed batch size is only right when every item costs the same, which invoices do
    not: a one-line receipt and a thirty-line statement differ by an order of magnitude.
    So `cost_per_item` is a starting estimate, and each observation replaces it with
    measurement. `observations` carry `items`, `completion_tokens`, `truncated` and
    `failed` — the runtime's record of what each batch actually cost.

    The adjustment is deliberately asymmetric. Cost rises fast and is corrected slowly:
    an estimate never falls below what was just measured, a truncation proves the true
    cost is at least the whole ceiling divided across that batch, and any batch that hit
    a ceiling caps every later one. Growth is capped at `growth`× the last batch, so one
    cheap batch cannot undo the caution earned by an expensive one.
    """
    usable = max(1, int(capacity * reserve))
    estimate = float(max(1, cost_per_item))
    last_size = 0
    #: A batch size proven too large. Everything after it stays below.
    ceiling: int | None = None

    for observation in observations:
        size = int(observation.get("items") or 0)
        if size <= 0:
            continue
        last_size = size
        spent = int(observation.get("completion_tokens") or 0)
        if observation.get("truncated"):
            # It ran out of room, so the real cost is at least the ceiling spread over
            # the batch — more than whatever we had estimated.
            estimate = max(estimate, capacity / size)
            ceiling = size if ceiling is None else min(ceiling, size)
        elif spent > 0:
            measured = spent / size
            # Half the weight on history, but never estimate below what was just seen.
            estimate = max((estimate + measured) / 2, measured)
        if observation.get("failed"):
            ceiling = size if ceiling is None else min(ceiling, size)

    size = max(1, int(usable // estimate))
    if ceiling is not None:
        size = min(size, max(1, int(ceiling * backoff)))
    if last_size:
        size = min(size, max(1, int(last_size * growth)))
    size = min(size, len(remaining))

    return {
        "needed": size < len(remaining),
        "batch": list(remaining[:size]),
        "batch_size": size,
        "remaining_count": max(0, len(remaining) - size),
        "estimate": int(round(estimate)),
    }
