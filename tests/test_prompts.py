"""Prompt artifacts: pinned, digest-verified, and complete."""

from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.contracts import PromptArtifact
from shakespeare.operators.builtin import build_registry
from shakespeare.planner import PLAN_SIGNATURE, REVIEW_SIGNATURE, ROUTE_SIGNATURE
from shakespeare.prompts import PromptStore, PromptStoreError
from shakespeare.stages import StageRegistry
from shakespeare.workflows import WorkflowRegistry


class TestCompleteness:
    def test_every_pinned_domain_prompt_exists(self) -> None:
        """A stage pins a prompt version; a missing artifact breaks the run at that stage."""
        store = PromptStore()
        stages = StageRegistry()
        for ref in stages.refs():
            for domain in stages.get(ref).domains:
                artifact = store.load(domain.id, domain.prompt_version)
                assert artifact.instructions.strip()

    @pytest.mark.parametrize("signature", [ROUTE_SIGNATURE, PLAN_SIGNATURE, REVIEW_SIGNATURE])
    def test_planner_prompts_exist(self, signature: str) -> None:
        assert PromptStore().load(signature, "1.0.0").instructions.strip()

    def test_domain_ids_are_unique_within_a_spine(self) -> None:
        """Prompts resolve by domain id, so a duplicate would be ambiguous."""
        stages = StageRegistry()
        registry = WorkflowRegistry(stages=stages, operators=build_registry())
        for workflow_id in registry.ids():
            ids = [
                domain.id
                for stage in registry.get(workflow_id).stages
                for domain in stage.domains
            ]
            assert len(ids) == len(set(ids)), workflow_id


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

    def test_prompt_version_participates_in_the_workflow_digest(self) -> None:
        stages = StageRegistry()
        registry = WorkflowRegistry(stages=stages, operators=build_registry())
        original = registry.get("rename_files")
        before = original.digest()

        bumped_stages = tuple(
            stage.model_copy(
                update={
                    "domains": tuple(
                        domain.model_copy(update={"prompt_version": "9.9.9"})
                        for domain in stage.domains
                    )
                }
            )
            for stage in original.stages
        )
        after = original.__class__(
            spec=original.spec, card=original.card, stages=bumped_stages
        ).digest()
        assert before != after, "promoting a prompt must change the workflow digest"
