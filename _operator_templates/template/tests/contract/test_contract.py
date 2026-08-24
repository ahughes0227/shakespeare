"""Contract tier: the package declares what the registry requires."""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _manifest():
    return json.loads((ROOT / "operator.json").read_text())


def test_manifest_is_complete():
    manifest = _manifest()
    for field in ("name", "version", "family", "entrypoint", "runner_operation"):
        assert manifest.get(field), f"missing manifest field: {field}"


def test_entrypoint_matches_family_runner():
    from shakespeare.contracts import OperatorFamily
    from shakespeare.registry import FAMILY_RUNNERS

    manifest = _manifest()
    expected = FAMILY_RUNNERS[OperatorFamily(manifest["family"])]
    assert manifest["entrypoint"] == expected
