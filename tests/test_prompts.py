"""Prompt artifacts: pinned, digest-verified, and complete."""

from __future__ import annotations

from pathlib import Path

import pytest
from system.capabilities import CapabilityRegistry
from system.components.catalog import build_registry
from system.contracts import PromptArtifact
from system.planning.planner import (
    CAPABILITY_SIGNATURE,
    GOAL_SIGNATURE,
    JUDGE_SIGNATURE,
    ROUTE_SIGNATURE,
)
from system.prompt_store import PromptStore, PromptStoreError
from system.workflows import WorkflowRegistry


class TestCompleteness:
    def test_every_pinned_capability_prompt_exists(self) -> None:
        """A capability pins a prompt version; a missing artifact breaks it at run time."""
        store = PromptStore()
        capabilities = CapabilityRegistry()
        for capability_id in capabilities.ids():
            spec = capabilities.get(capability_id)
            assert store.load(capability_id, spec.prompt_version).instructions.strip()

    @pytest.mark.parametrize(
        "signature",
        [ROUTE_SIGNATURE, GOAL_SIGNATURE, CAPABILITY_SIGNATURE, JUDGE_SIGNATURE],
    )
    def test_planner_prompts_exist(self, signature: str) -> None:
        assert PromptStore().load(signature, "1.0.0").instructions.strip()

    def test_every_capability_a_goal_names_is_registered(self) -> None:
        """A goal naming an unregistered capability is unanswerable."""
        capabilities = CapabilityRegistry()
        registry = WorkflowRegistry(capabilities=capabilities, operators=build_registry())
        for workflow_id in registry.ids():
            for goal in registry.get(workflow_id).spec.goals:
                for name in goal.capabilities:
                    assert name in capabilities, f"{workflow_id}.{goal.id} -> {name}"


class TestPinning:
    def test_editing_an_artifact_in_place_is_rejected(self, tmp_path: Path) -> None:
        """A prompt version is part of the workflow digest.

        Editing one in place would change behaviour while claiming to be the same run,
        which would quietly break replay.
        """
        store = PromptStore(tmp_path)
        path = store.save(
            PromptArtifact(signature_id="x.y", version="1.0.0", instructions="original")
        )
        path.write_text(path.read_text().replace("original", "tampered"))
        with pytest.raises(PromptStoreError, match="bump the version"):
            store.load("x.y", "1.0.0")

    def test_a_new_version_is_the_supported_path(self, tmp_path: Path) -> None:
        store = PromptStore(tmp_path)
        store.save(PromptArtifact(signature_id="x.y", version="1.0.0", instructions="first"))
        store.save(PromptArtifact(signature_id="x.y", version="1.1.0", instructions="second"))
        assert store.versions("x.y") == ("1.0.0", "1.1.0")
        assert store.load("x.y", "1.0.0").instructions == "first"

    def test_a_changed_goal_changes_the_workflow_digest(self) -> None:
        """The digest pins the intent, so altering a gate is a visible change."""
        capabilities = CapabilityRegistry()
        registry = WorkflowRegistry(capabilities=capabilities, operators=build_registry())
        original = registry.get("rename_files")
        before = original.digest()

        goals = list(original.spec.goals)
        goals[0] = goals[0].model_copy(update={"statement": "something else entirely"})
        after = original.__class__(
            spec=original.spec.model_copy(update={"goals": tuple(goals)}),
            card=original.card,
        ).digest()
        assert before != after

    def test_a_capability_pins_its_prompt_version(self) -> None:
        capabilities = CapabilityRegistry()
        for capability_id in capabilities.ids():
            assert capabilities.get(capability_id).prompt_version
