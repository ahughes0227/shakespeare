"""Regression tier: the rendered package is reproducible."""

import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_marker_matches_manifest():
    marker = yaml.safe_load((ROOT / ".operator-template.yml").read_text())
    manifest = json.loads((ROOT / "operator.json").read_text())
    assert marker["family"] == manifest["family"]
    assert str(marker["revision"]) == str(manifest["template_revision"])
