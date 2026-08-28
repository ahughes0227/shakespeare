"""The deterministic half of a gate.

A check is verification, not behaviour: it reads the evidence a run has produced and says
whether an obligation holds. It lived beside the plan operators because that is where the
first few were written, which put the thing being verified and the thing verifying it in
one module.

Every check is total and takes the same shape — an obligation id and a payload in, an
`ObligationResult` out — so a workflow can name one in a gate without the runtime knowing
what it does.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ..contracts import ChangeAction, Contract, ObligationResult, content_digest


class CheckError(ValueError):
    """A gate named a check that does not exist, or one whose evidence is absent."""


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


def check_convention_followed(obligation_id: str, payload: dict[str, Any]) -> ObligationResult:
    """Every name must be the one the frozen convention produces from the stored values.

    `spec.freeze` and the convention goal make a convention evidence; nothing made it
    binding at the point of use. name.render also accepts a `template` and `fields`
    directly, for workflows that have no frozen spec — so a capability could author its
    own convention at render time and the run would commit names in a format nobody asked
    for, with every gate green. A live run did exactly that: sixty files named
    `2024-04-01, ...` under a convention that says `%Y%m`.

    So the names are recomputed here from the frozen spec and the values that produced
    them, and must match. This compares the pre-collision render on both sides, because
    collision resolution legitimately rewrites a name and re-rendering would not.
    """
    from ..components.pure_transform import naming

    spec_payload = payload.get("spec")
    rows = payload.get("results") or []
    if not spec_payload or not rows:
        return _result(obligation_id, False, reason="no frozen spec or no render results")

    spec = naming.NamingSpec.model_validate(spec_payload)
    divergent: list[dict[str, str]] = []
    checked = 0
    for index, row in enumerate(rows):
        claimed = row.get("rendered")
        if not claimed:
            continue
        checked += 1
        expected = naming.render(
            item_id=str(row.get("item_id", "")),
            template=spec.template,
            fields=spec.fields,
            values=row.get("values") or {},
            policy=spec.policy,
            extension=row.get("extension", ""),
            sequence=index + 1,
            confidences=row.get("confidences") or {},
            floor=spec.confidence_floor,
        )
        if expected.rendered != claimed:
            divergent.append(
                {"item_id": str(row.get("item_id", "")), "named": claimed,
                 "convention": str(expected.rendered)}
            )
    # Divergence is the only thing this check is about. A corpus where every document is
    # unreadable is legitimately all quarantine and renders nothing, and demanding that
    # something was named would fail a run that behaved correctly — `resolution_accounted`
    # is what makes sure every item is named or quarantined.
    return _result(
        obligation_id,
        not divergent,
        checked=checked,
        divergent=divergent[:10],
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
    "convention_followed": check_convention_followed,
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
    "convention_followed": ("spec", "results"),
    "structure_mirrors": ("expected_dirs", "actual_dirs"),
}


def missing_evidence(check: str, payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(key for key in CHECK_REQUIREMENTS.get(check, ()) if key not in payload)


def run_check(obligation_id: str, check: str, payload: dict[str, Any]) -> ObligationResult:
    try:
        checker = CHECKS[check]
    except KeyError as exc:
        raise CheckError(f"unknown obligation check: {check}") from exc
    absent = missing_evidence(check, payload)
    if absent:
        return _result(obligation_id, False, missing_evidence=list(absent))
    return checker(obligation_id, payload)


# --------------------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------------------
