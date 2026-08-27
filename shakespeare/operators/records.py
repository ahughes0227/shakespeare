"""A durable per-item record store, so a model's reading does not have to be carried.

Everything a model reads from a document had to survive inside its own response until the
next operator could be handed it. That is why a sixty-invoice run needed batch sizing at
all: the response was the transport. A record written the moment it is read needs no
transport, survives a failed batch, survives a failed attempt, and can be read back by
deterministic code that never involves a model again.

Parquet rather than JSON because the thing being built is a table, and because reading it
back column-wise is what the renaming loop actually wants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STORE_DIRNAME = "records"


def _table_path(workspace: Path, table: str) -> Path:
    """Where a table lives. Contained to the run workspace by construction.

    The name is checked rather than sanitised: a table name is configuration, and a
    separator or a parent reference in one is a mistake worth refusing rather than
    quietly rewriting.
    """
    if not table or any(part in table for part in ("/", "\\", "..")) or table.startswith("."):
        raise ValueError(f"{table!r} is not a table name")
    directory = workspace / STORE_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{table}.parquet"


def append(
    *, workspace: Path, table: str, rows: tuple[dict[str, Any], ...], key: str = "item_id"
) -> dict[str, Any]:
    """Add rows, replacing any that share a key with one already stored.

    Replacing rather than appending blindly is what makes a re-read idempotent: a document
    read twice leaves one record, so a retried batch cannot double-count and a corrected
    reading supersedes the one it corrects.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = _table_path(workspace, table)
    stored: list[dict[str, Any]] = []
    if path.exists():
        stored = pq.read_table(path).to_pylist()

    merged: dict[str, dict[str, Any]] = {
        str(row[key]): row for row in stored if key in row
    }
    added, replaced = 0, 0
    for row in rows:
        if key not in row:
            raise ValueError(f"every row needs {key!r}")
        identifier = str(row[key])
        if identifier in merged:
            replaced += 1
        else:
            added += 1
        merged[identifier] = _flatten(dict(row))

    ordered = sorted(merged.values(), key=lambda row: str(row[key]))
    pq.write_table(pa.Table.from_pylist(ordered), path)
    return {
        "table": table,
        "stored": len(ordered),
        "added": added,
        "replaced": replaced,
        "path": str(path),
    }


def read(*, workspace: Path, table: str) -> dict[str, Any]:
    """Every stored row, in key order. Empty rather than absent when nothing is stored yet."""
    import pyarrow.parquet as pq

    path = _table_path(workspace, table)
    if not path.exists():
        return {"table": table, "records": [], "stored": 0}
    rows = [_restore(row) for row in pq.read_table(path).to_pylist()]
    return {"table": table, "records": rows, "stored": len(rows)}


#: Columns whose values are mappings are stored as JSON text. A table wants columns, and
#: a model's `values` and `confidences` are open-ended maps whose keys come from the
#: naming convention rather than from a schema.
_ENCODED = ("values", "confidences")


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    for column in _ENCODED:
        if isinstance(row.get(column), dict):
            row[column] = json.dumps(row[column], sort_keys=True)
    return row


def _restore(row: dict[str, Any]) -> dict[str, Any]:
    for column in _ENCODED:
        if isinstance(row.get(column), str):
            try:
                row[column] = json.loads(row[column])
            except json.JSONDecodeError:
                pass
    return row
