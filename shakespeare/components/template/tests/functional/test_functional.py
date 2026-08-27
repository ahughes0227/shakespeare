"""Functional tier: the declared runner operation exists in the family allowlist."""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_runner_operation_is_vetted():
    from shakespeare.components.runners import allowlist

    manifest = json.loads((ROOT / "operator.json").read_text())
    operations = allowlist(manifest["family"])
    assert manifest["runner_operation"] in operations, (
        f"{manifest['runner_operation']} is not a vetted operation for "
        f"{manifest['family']}; a human must add it to shakespeare/runners.py"
    )
