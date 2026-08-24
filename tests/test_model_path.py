"""The production model path, exercised offline.

test_rename_files.py fakes the *agents*; this fakes only the *gateway*, so the real
ModelPlanner and ModelDomainAgent run: prompts are loaded and digest-checked, messages
are built, and responses are parsed and validated against their contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shakespeare.agent import CompositionDraft, ModelDomainAgent
from shakespeare.audit import AuditStore
from shakespeare.contracts import (
    DomainGoal,
    RequestContract,
    RouteDecision,
    StagePlan,
    StageVerdict,
)
from shakespeare.executor import Executor
from shakespeare.gateway import FakeGateway, GatewayError, ModelProfile
from shakespeare.operators.builtin import build_registry
from shakespeare.planner import ModelPlanner
from shakespeare.prompts import PromptStore
from shakespeare.stages import StageRegistry
from shakespeare.verifier import Verifier
from shakespeare.workflows import WorkflowRegistry

PROFILE = ModelProfile(profile_id="test", model="openrouter/openai/gpt-5-mini")


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def planner(gateway: FakeGateway) -> ModelPlanner:
    return ModelPlanner(gateway=gateway, profile=PROFILE, prompts=PromptStore())


class TestPlannerPath:
    def test_route_uses_the_pinned_prompt_and_the_workflow_cards(
        self, gateway: FakeGateway, planner: ModelPlanner
    ) -> None:
        gateway.queue(
            RouteDecision,
            {"workflow_id": "rename_files", "supported": True, "rationale": "renaming"},
        )
        stages = StageRegistry()
        registry = WorkflowRegistry(stages=stages, operators=build_registry())
        decision, usage = planner.select_workflow(
            RequestContract(
                request_id="r", prompt="rename my invoices", input_root="/in", output_root="/out"
            ),
            registry.routing_catalog(),
        )
        assert decision.workflow_id == "rename_files"
        assert usage is not None and usage.requested_model == PROFILE.model

        # The routing catalog, and nothing else about a workflow, reaches the prompt.
        _, messages = gateway.calls[0]
        payload = messages[-1]["content"]
        assert "rename_files" in payload
        assert "mirroring the input" in payload  # from workflow-context.yml

    def test_stage_plan_prompt_carries_scope_and_skippability(
        self, gateway: FakeGateway, planner: ModelPlanner
    ) -> None:
        gateway.queue(
            StagePlan,
            {
                "activated": [
                    {
                        "domain_id": "content_acquisition",
                        "goal": "get text for every file",
                        "success_criterion": "every item has text or a reason",
                    }
                ],
                "skipped": [],
            },
        )
        stage = StageRegistry().get("extract@1.0.0")
        plan, _ = planner.plan_stage(
            stage,
            RequestContract(request_id="r", prompt="p", input_root="/in", output_root="/out"),
            {"items": []},
        )
        assert plan.activated[0].domain_id == "content_acquisition"
        _, messages = gateway.calls[0]
        assert "skippable" in messages[-1]["content"]

    def test_a_malformed_response_is_a_permanent_error(
        self, gateway: FakeGateway, planner: ModelPlanner
    ) -> None:
        """Retrying the same prompt would produce the same shape, so this is not transient."""
        gateway.queue(StageVerdict, {"decision": "not-a-decision"})
        with pytest.raises(GatewayError) as caught:
            planner.review_stage(
                StageRegistry().get("extract@1.0.0"),
                StagePlan(),
                (),
                {},
                attempts_remaining=1,
            )
        assert caught.value.code.value == "model_permanent"


class TestDomainAgentPath:
    def test_agent_cannot_claim_another_domain(self, gateway: FakeGateway) -> None:
        """The model returns a draft; the domain is stamped by the runtime."""
        gateway.queue(
            CompositionDraft,
            {
                "invocations": [
                    {"invocation_id": "scan", "operator": "fs.scan", "inputs": ["root"]}
                ],
                "rationale": "walk the tree",
            },
        )
        agent = ModelDomainAgent(gateway=gateway, profile=PROFILE, prompts=PromptStore())
        domain = StageRegistry().get("intake@1.0.0").domain("file_validity")
        composition, _ = agent.compose(
            domain=domain,
            goal=DomainGoal(domain_id="file_validity", goal="g", success_criterion="c"),
            stage_inputs={"root": "/in"},
            catalog_summary={},
        )
        assert composition.domain_id == "file_validity"

    def test_prompt_lists_only_the_granted_surface(self, gateway: FakeGateway) -> None:
        gateway.queue(CompositionDraft, {"invocations": []})
        agent = ModelDomainAgent(gateway=gateway, profile=PROFILE, prompts=PromptStore())
        domain = StageRegistry().get("extract@1.0.0").domain("content_acquisition")
        agent.compose(
            domain=domain,
            goal=DomainGoal(domain_id="content_acquisition", goal="g", success_criterion="c"),
            stage_inputs={"items": [1, 2, 3]},
            catalog_summary={"extract": ["auto_chain", "pdf_text"]},
        )
        _, messages = gateway.calls[0]
        payload = messages[-1]["content"]
        assert "doc.extract" in payload
        assert "fs.commit" not in payload, "a domain must never be shown a mutation operator"

    def test_a_composition_outside_the_catalog_is_still_refused(
        self, gateway: FakeGateway
    ) -> None:
        """The prompt is guidance; the verifier is the control."""
        gateway.queue(
            CompositionDraft,
            {"invocations": [{"invocation_id": "c", "operator": "fs.commit"}]},
        )
        agent = ModelDomainAgent(gateway=gateway, profile=PROFILE, prompts=PromptStore())
        domain = StageRegistry().get("extract@1.0.0").domain("content_acquisition")
        composition, _ = agent.compose(
            domain=domain,
            goal=DomainGoal(domain_id="content_acquisition", goal="g", success_criterion="c"),
            stage_inputs={},
            catalog_summary={},
        )
        from shakespeare.verifier import Denial

        with pytest.raises(Denial):
            Verifier(build_registry()).verify_composition(composition, domain)


class TestWiring:
    def test_bootstrap_assembles_with_a_fake_gateway(self, tmp_path: Path) -> None:
        """The production wiring must be constructible without a network."""
        from shakespeare.bootstrap import build_runtime

        services = build_runtime(
            state_root=tmp_path,
            gateway=FakeGateway(),
            profile=PROFILE,
        )
        assert "rename_files" in services.workflows.ids()
        assert isinstance(services.runtime.executor, Executor)
        assert isinstance(services.runtime.verifier, Verifier)
        assert isinstance(services.audit, AuditStore)
        services.audit.close()
