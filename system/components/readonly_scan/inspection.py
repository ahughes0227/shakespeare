"""Read-only inspection of an input tree.

Ordering is deterministic and symlinks are never followed: a plan is portable data, so
two scans of the same tree must produce the same inventory in the same order.
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from ..pure_transform.plans import ScannedItem

_CHUNK = 1 << 20
#: Skipped everywhere: these are never user documents and would pollute accounting.
_SKIP_NAMES = frozenset({".DS_Store", "Thumbs.db", ".gitkeep"})


class ScanError(RuntimeError):
    pass


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def media_type_of(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def scan(
    root: Path,
    *,
    depth_limit: int = 32,
    include_hidden: bool = False,
) -> tuple[tuple[ScannedItem, ...], tuple[dict[str, str], ...]]:
    """Return (items, skipped).

    Unreadable files are *reported*, never dropped: accounting must balance, so an item
    the scan could not read still has to appear somewhere.
    """
    root = root.resolve()
    if not root.is_dir():
        raise ScanError(f"input root is not a directory: {root}")

    items: list[ScannedItem] = []
    skipped: list[dict[str, str]] = []

    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relpath = path.relative_to(root).as_posix()
        parts = relpath.split("/")

        if path.is_symlink():
            # Following a symlink could escape the input root entirely.
            skipped.append({"relpath": relpath, "reason": "symlink"})
            continue
        if not include_hidden and any(part.startswith(".") for part in parts):
            continue
        if path.is_dir():
            continue
        if path.name in _SKIP_NAMES:
            continue
        if len(parts) > depth_limit:
            skipped.append({"relpath": relpath, "reason": "depth_limit_exceeded"})
            continue

        try:
            stat = path.stat()
            sha256 = digest_file(path)
        except (OSError, PermissionError) as exc:
            skipped.append({"relpath": relpath, "reason": f"unreadable:{type(exc).__name__}"})
            continue

        items.append(
            ScannedItem(
                # Identity is the file, not its contents. Two byte-identical invoices in
                # different quarters are two things to rename, and a pure content address
                # made them one: a plan then carried fewer entries than there were files
                # and failed its own balance check, reporting "5 entries for 5 scanned"
                # without saying that four of the five shared an id.
                item_id=hashlib.sha256(
                    f"{relpath}\0{sha256}".encode()
                ).hexdigest()[:16],
                relpath=relpath,
                sha256=sha256,
                media_type=media_type_of(path),
                size_bytes=stat.st_size,
            )
        )

    return tuple(items), tuple(skipped)


def directories(root: Path) -> tuple[str, ...]:
    root = root.resolve()
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir() and not path.is_symlink()
        )
    )
