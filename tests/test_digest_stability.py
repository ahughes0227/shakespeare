"""Digests must not depend on the process that computed them.

A `frozenset` field dumps to a list in iteration order, and Python randomises string
hashing per process — so the workflow digest differed on every run. That silently made
`replay` impossible (it refuses on a digest mismatch) and turned every digest recorded in
the audit log into a number that could never be compared with anything.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from system.contracts import (
    Composition,
    DomainSpec,
    Invocation,
    OperatorFamily,
    OperatorSpec,
    canonical_json,
    content_digest,
)

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import sys
sys.path.insert(0, {root!r})
from system.capabilities import CapabilityRegistry
from system.workflows import WorkflowRegistry
from system.components.builtin import build_registry
capabilities = CapabilityRegistry()
registry = WorkflowRegistry(capabilities=capabilities, operators=build_registry())
print(registry.get("rename_files").digest())
"""


class TestAcrossProcesses:
    def test_the_workflow_digest_is_identical_in_fresh_processes(self) -> None:
        """The check that would have caught it: separate interpreters, separate hash seeds."""
        digests = set()
        for _ in range(4):
            result = subprocess.run(
                [sys.executable, "-c", PROBE.format(root=str(ROOT))],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            assert result.returncode == 0, result.stderr
            digests.add(result.stdout.strip())
        assert len(digests) == 1, f"workflow digest varies by process: {digests}"


class TestSetOrdering:
    def test_a_frozenset_field_digests_identically_whatever_its_order(self) -> None:
        first = DomainSpec(
            id="d", scope="s", catalog=frozenset({"a.one", "b.two", "c.three"})
        )
        second = DomainSpec(
            id="d", scope="s", catalog=frozenset({"c.three", "a.one", "b.two"})
        )
        assert first.digest() == second.digest()

    def test_nested_sets_are_normalised_too(self) -> None:
        def spec(features: frozenset[str]) -> OperatorSpec:
            return OperatorSpec(
                name="x",
                version="1.0.0",
                description="d",
                family=OperatorFamily.PURE_TRANSFORM,
                entrypoint="system.components.runners:pure_transform",
                features=features,
            )

        assert content_digest({"op": spec(frozenset({"a", "b"}))}) == content_digest(
            {"op": spec(frozenset({"b", "a"}))}
        )

    def test_a_bare_set_is_normalised(self) -> None:
        assert canonical_json({"k": {3, 1, 2}}) == canonical_json({"k": {2, 3, 1}})

    def test_list_order_is_still_significant(self) -> None:
        """Only sets are order-free. Reordering a composition changes what it does."""
        def composition(order: tuple[str, ...]) -> Composition:
            return Composition(
                domain_id="d",
                invocations=tuple(
                    Invocation(invocation_id=name, operator="text.normalize")
                    for name in order
                ),
            )

        assert composition(("a", "b")).digest() != composition(("b", "a")).digest()


class TestReplayDependsOnIt:
    def test_replay_accepts_a_digest_recomputed_in_another_process(self) -> None:
        """Replay refuses on a digest mismatch, so an unstable digest disables it."""
        from system.runtime.replay import assert_same_workflow

        result = subprocess.run(
            [sys.executable, "-c", PROBE.format(root=str(ROOT))],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, result.stderr
        recorded = result.stdout.strip()

        from system.capabilities import CapabilityRegistry
        from system.components.builtin import build_registry
        from system.workflows import WorkflowRegistry

        capabilities = CapabilityRegistry()
        current = WorkflowRegistry(capabilities=capabilities, operators=build_registry())
        assert_same_workflow(recorded, current.get("rename_files").digest())

    def test_a_genuinely_changed_workflow_is_still_refused(self) -> None:
        from system.runtime.replay import ReplayError, assert_same_workflow

        with pytest.raises(ReplayError):
            assert_same_workflow("a" * 64, "b" * 64)
