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
    planned: tuple[PlannedName, ...],
    operator_versions: dict[str, str] | None = None,
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
                    action=ChangeAction.UNRESOLVED,
                    reason="no_decision",
                    source_sha256=item.sha256,
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

        target = f"{decision.directory}/{decision.name}" if decision.directory else decision.name
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

    plan = ChangePlan(
        run_id=run_id,
        workflow_id=workflow_id,
        workflow_digest=workflow_digest,
        decision_digest=decision_digest,
        operator_versions=operator_versions or {},
        entries=tuple(entries),
    )
    if not plan.balanced(len(scanned)):
        raise AssemblyError(
            f"unbalanced plan: {len(plan.entries)} entries for {len(scanned)} scanned items"
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
        for item in payload.get("items", [])
        if not item.get("text") and not item.get("unavailable_reason")
    ]
    return _result(obligation_id, not offenders, without_text_or_reason=offenders[:20])


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
    """Every target must be reproducible by re-rendering from the frozen spec.

    This is the obligation that makes naming consistent regardless of the route an agent
    took: it is verified by digest rather than trusted.
    """
    mismatches = [
        entry["item_id"]
        for entry in payload.get("comparisons", [])
        if entry.get("expected") != entry.get("actual")
    ]
    return _result(obligation_id, not mismatches, mismatches=mismatches[:20])


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
    "spec_frozen": check_spec_frozen,
    "rendered_mechanically": check_rendered_mechanically,
    "structure_mirrors": check_structure_mirrors,
}


def run_check(obligation_id: str, check: str, payload: dict[str, Any]) -> ObligationResult:
    try:
        checker = CHECKS[check]
    except KeyError as exc:
        raise AssemblyError(f"unknown obligation check: {check}") from exc
    return checker(obligation_id, payload)
