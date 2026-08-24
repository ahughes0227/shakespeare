"""Containment tier: the package carries no executable behaviour."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_implementation_defines_no_callable():
    tree = ast.parse((ROOT / "implementation.py").read_text())
    offenders = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert not offenders, f"generated package must contain no callable: {offenders}"


def test_implementation_imports_nothing():
    tree = ast.parse((ROOT / "implementation.py").read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not imports, "a declarative package has no reason to import"
