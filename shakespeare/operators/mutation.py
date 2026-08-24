"""The only operators that write.

Execute writes into a staging tree; nothing user-visible changes until Review passes and
`commit` performs one atomic move.  Every write records a reversal, so `undo` is a
replay of recorded facts rather than a guess.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

from ..contracts import ChangeAction, ChangePlan, ReversalRecord
from .filesystem import digest_file


class MutationError(RuntimeError):
    pass


def _guard(root: Path, relpath: str) -> Path:
    """Refuse any path that escapes its root.

    `..` segments and absolute paths in a plan are the obvious attack, but a symlinked
    parent is the subtle one, so the check is on the resolved path.
    """
    if os.path.isabs(relpath):
        raise MutationError(f"absolute path in plan: {relpath}")
    candidate = (root / relpath).resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise MutationError(f"path escapes its root: {relpath}")
    return candidate


def stage_plan(
    *,
    plan: ChangePlan,
    input_root: Path,
    staging_root: Path,
    quarantine_dirname: str = "_unresolved",
) -> tuple[ReversalRecord, ...]:
    """Materialise a plan into staging by copying.

    Copying rather than moving is what keeps the source tree untouched, so an aborted run
    costs nothing but disk.
    """
    input_root = input_root.resolve()
    staging_root = staging_root.resolve()
    staging_root.mkdir(parents=True, exist_ok=True)

    reversals: list[ReversalRecord] = []
    for entry in plan.entries:
        source = _guard(input_root, entry.source_ref)
        if not source.is_file():
            raise MutationError(f"plan references a missing source: {entry.source_ref}")

        if entry.action is ChangeAction.UNRESOLVED:
            # Quarantined items keep their original name and structure so a human can
            # find them; they are never given a guessed name.
            target = _guard(staging_root, f"{quarantine_dirname}/{entry.source_ref}")
        else:
            target_relpath = getattr(entry, "target_relpath", None) or entry.source_ref
            target = _guard(staging_root, target_relpath)

        if target.exists():
            raise MutationError(f"staging target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

        reversals.append(
            ReversalRecord(
                mutation_id=uuid4().hex,
                operation="stage_write",
                payload={
                    "target": str(target),
                    "item_id": entry.item_id,
                    "after_digest": digest_file(target),
                },
            )
        )
    return tuple(reversals)


def commit(*, staging_root: Path, output_root: Path) -> ReversalRecord:
    """Move the verified staging tree into place in one atomic operation."""
    staging_root = staging_root.resolve()
    output_root = Path(output_root).resolve()
    if not staging_root.is_dir():
        raise MutationError(f"staging root does not exist: {staging_root}")
    if output_root.exists():
        raise MutationError(f"output root already exists, refusing to overwrite: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staging_root, output_root)
    except OSError as exc:
        # os.replace is only atomic within a filesystem.  Falling back to a copy would be
        # silently non-atomic, so we refuse instead and let the caller relocate staging.
        raise MutationError(
            f"cannot atomically commit across filesystems: {exc}. "
            f"Place the staging root on the same filesystem as {output_root}."
        ) from exc

    return ReversalRecord(
        mutation_id=uuid4().hex,
        operation="commit",
        payload={"output_root": str(output_root), "staging_root": str(staging_root)},
    )


def discard(staging_root: Path) -> None:
    """Roll back an uncommitted run.  Nothing user-visible was ever created."""
    shutil.rmtree(staging_root, ignore_errors=True)


def reverse(record: ReversalRecord) -> None:
    """Undo one recorded mutation."""
    if record.operation == "commit":
        output_root = Path(record.payload["output_root"])
        if output_root.exists():
            shutil.rmtree(output_root)
        return
    if record.operation == "stage_write":
        target = Path(record.payload["target"])
        target.unlink(missing_ok=True)
        return
    raise MutationError(f"no reversal is defined for operation: {record.operation}")


def verify_tree(
    *,
    plan: ChangePlan,
    staging_root: Path,
    quarantine_dirname: str = "_unresolved",
) -> dict[str, object]:
    """Re-scan staging and compare it against the plan before any commit."""
    staging_root = staging_root.resolve()
    missing: list[str] = []
    mismatched: list[str] = []

    for entry in plan.entries:
        if entry.action is ChangeAction.UNRESOLVED:
            expected = staging_root / quarantine_dirname / entry.source_ref
        else:
            target_relpath = getattr(entry, "target_relpath", None) or entry.source_ref
            expected = staging_root / target_relpath
        if not expected.is_file():
            missing.append(entry.item_id)
            continue
        source_digest = entry.digests.get("source") or getattr(entry, "source_sha256", None)
        if source_digest and digest_file(expected) != source_digest:
            mismatched.append(entry.item_id)

    staged = sum(1 for path in staging_root.rglob("*") if path.is_file())
    return {
        "ok": not missing and not mismatched and staged == len(plan.entries),
        "missing": missing,
        "mismatched": mismatched,
        "staged_files": staged,
        "planned_entries": len(plan.entries),
    }
