"""The production model path, exercised offline.

Other tests fake the capability agents; this fakes only the *gateway*, so the real
ModelGoalPlanner and ModelCapabilityAgent run: prompts load and digest-check, messages
build, and responses parse against their contracts.
"""

from __future__ import annotations

import pytest
from shakespeare.agent import ModelCapabilityAgent
from shakespeare.capabilities import CapabilityRegistry
from shakespeare.capabilities.runner import Organization
from shakespeare.contracts import RequestContract, RouteDecision
from shakespeare.gateway import FakeGateway, GatewayError, ModelProfile
from shakespeare.operators.builtin import build_registry
from shakespeare.planner import CapabilityChoice, GoalChoice, Judgment, ModelGoalPlanner
from shakespeare.prompts import PromptStore
from shakespeare.verifier import Denial, Verifier
from shakespeare.workflows import WorkflowRegistry

PROFILE = ModelProfile(profile_id="test", model="openrouter/openai/gpt-5-mini")


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def planner(gateway: FakeGateway) -> ModelGoalPlanner:
    return ModelGoalPlanner(gateway=gateway, profile=PROFILE, prompts=PromptStore())


def _graph():
    capabilities = CapabilityRegistry()
    registry = WorkflowRegistry(capabilities=capabilities, operators=build_registry())
    return registry, registry.get("rename_files").spec.graph


class TestPlannerPath:
    def test_route_reads_only_the_workflow_cards(
        self, gateway: FakeGateway, planner: ModelGoalPlanner
    ) -> None:
        gateway.queue(
            RouteDecision,
            {"workflow_id": "rename_files", "supported": True, "rationale": "renaming"},
        )
        registry, _ = _graph()
        decision, usage = planner.select_workflow(
            RequestContract(
                request_id="r", prompt="rename my invoices", input_root="/in", output_root="/out"
            ),
            registry.routing_catalog(),
        )
        assert decision.workflow_id == "rename_files"
        assert usage is not None and usage.requested_model == PROFILE.model
        payload = gateway.calls[0][1][-1]["content"]
        assert "mirroring the input" in payload, "the card, and only the card"

    def test_goal_selection_is_shown_open_goals_and_evidence(
        self, gateway: FakeGateway, planner: ModelGoalPlanner
    ) -> None:
        gateway.queue(GoalChoice, {"goal_id": "convention_frozen", "rationale": "ready"})
        _, graph = _graph()
        chosen = planner.select_goal(
            graph.open_goals(frozenset({"inventoried"})),
            [{"kind": "FileInventory", "quality": "complete"}],
        )
        assert chosen == "convention_frozen"
        payload = gateway.calls[0][1][-1]["content"]
        assert "readable" in payload and "convention_frozen" in payload

    def test_capability_selection_names_only_the_permitted_ones(
        self, gateway: FakeGateway, planner: ModelGoalPlanner
    ) -> None:
        gateway.queue(CapabilityChoice, {"capability_id": "acquire", "rationale": "reads files"})
        _, graph = _graph()
        chosen = planner.select_capability(
            graph.goal("readable"), [{"id": "acquire", "standing_goal": "read files"}]
        )
        assert chosen.capability_id == "acquire"

    def test_the_judge_is_given_the_rubric_not_the_documents(
        self, gateway: FakeGateway, planner: ModelGoalPlanner
    ) -> None:
        gateway.queue(Judgment, {"satisfied": True, "rationale": "nothing more would change it"})
        _, graph = _graph()
        satisfied, rationale = planner.judge(
            goal=graph.goal("readable"),
            rubric=graph.goal("readable").gate.rubric,
            artifacts=[{"kind": "ExtractedContent", "quality": "partial"}],
            evidence={"items": 3},
        )
        assert satisfied and rationale
        payload = gateway.calls[0][1][-1]["content"]
        assert "Would reading more" in payload, "the goal's own rubric reaches the judge"
        assert "invoice body" not in payload, "a judge weighs sufficiency, not documents"

    def test_a_malformed_response_is_a_permanent_error(
        self, gateway: FakeGateway, planner: ModelGoalPlanner
    ) -> None:
        """Retrying the same prompt would produce the same shape."""
        gateway.queue(Judgment, {"satisfied": "definitely"})
        _, graph = _graph()
        with pytest.raises(GatewayError) as caught:
            planner.judge(
                goal=graph.goal("readable"), rubric="r", artifacts=[], evidence={}
            )
        assert caught.value.code.value == "model_permanent"


class TestCapabilityAgentPath:
    def _agent(self, gateway: FakeGateway) -> ModelCapabilityAgent:
        return ModelCapabilityAgent(gateway=gateway, profile=PROFILE, prompts=PromptStore())

    def _organize(self, gateway: FakeGateway, capability_id: str = "survey"):
        capability = CapabilityRegistry().get(capability_id)
        return self._agent(gateway).organize(
            capability=capability,
            request="inventory the tree",
            artifacts=[],
            context={"root": "/in"},
            prior=[],
            catalog_summary={
                "components": {name: {} for name in sorted(capability.catalog)},
                "config": {},
            },
        )

    def test_an_organization_parses_into_its_contract(self, gateway: FakeGateway) -> None:
        gateway.queue(
            Organization,
            {
                "invocations": [
                    {"invocation_id": "scan", "operator": "fs.scan", "inputs": ["root"]}
                ],
                "intent": "walk it",
                "sufficient": True,
                "publishes": "FileInventory",
            },
        )
        organization, _ = self._organize(gateway)
        assert organization.publishes == "FileInventory"
        assert organization.sufficient

    def test_the_prompt_lists_only_the_granted_surface(self, gateway: FakeGateway) -> None:
        gateway.queue(Organization, {"invocations": [], "sufficient": True})
        self._organize(gateway)
        payload = gateway.calls[0][1][-1]["content"]
        assert "fs.scan" in payload
        assert "fs.commit" not in payload, "a capability is never shown a mutation component"

    def test_prior_rounds_reach_the_prompt(self, gateway: FakeGateway) -> None:
        """Without its own history a capability restarts rather than continues."""
        gateway.queue(Organization, {"invocations": [], "sufficient": True})
        capability = CapabilityRegistry().get("acquire")
        self._agent(gateway).organize(
            capability=capability,
            request="read them",
            artifacts=[],
            context={},
            prior=[{"round": 1, "intent": "first slice", "succeeded": True}],
            catalog_summary={"components": {}, "config": {}},
        )
        payload = gateway.calls[0][1][-1]["content"]
        assert "first slice" in payload

    def test_a_component_outside_the_catalog_is_still_refused(
        self, gateway: FakeGateway
    ) -> None:
        """The prompt is guidance; the verifier is the control."""
        from shakespeare.capabilities.runner import _as_domain
        from shakespeare.contracts import Composition

        gateway.queue(
            Organization,
            {
                "invocations": [{"invocation_id": "c", "operator": "fs.commit"}],
                "sufficient": True,
            },
        )
        organization, _ = self._organize(gateway)
        composition = Composition(domain_id="survey", invocations=organization.invocations)
        with pytest.raises(Denial):
            Verifier(build_registry()).verify_composition(
                composition, _as_domain(CapabilityRegistry().get("survey"))
            )


class TestWiring:
    def test_bootstrap_assembles_without_a_network(self, tmp_path) -> None:
        from shakespeare.bootstrap import build_runtime

        services = build_runtime(
            state_root=tmp_path, gateway=FakeGateway(), profile=PROFILE
        )
        assert "rename_files" in services.workflows.ids()
        assert services.capabilities.ids()
        services.audit.close()
