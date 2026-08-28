"""Capability-level meta-organization.

The framework puts adaptive organization inside the bounded capability (§8). This is the
layer that was missing: a capability that cannot see its own results cannot decide it has
done twenty of sixty items and should carry on, which is how a live sixty-invoice run
failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from system.capabilities import CapabilityRunner, CapabilitySpec
from system.capabilities.runner import Organization, ScriptedCapabilityAgent
from system.components.catalog import build_registry
from system.contracts import BudgetEnvelope, Invocation, SemanticCard
from system.runtime.artifacts import ArtifactStore, Quality
from system.runtime.executor import Budget, Executor
from system.runtime.verifier import Verifier

SURVEY = CapabilitySpec(
    id="survey",
    version="1.0.0",
    standing_goal="Inventory a tree and describe what is in it.",
    catalog=frozenset({"fs.scan", "fs.dirs", "batch.window"}),
    produces=("FileInventory",),
    max_rounds=4,
)


def card(purpose: str) -> SemanticCard:
    filler = "declared for the test harness"
    return SemanticCard(
        purpose=purpose, lifecycle=filler, contracts=filler, allowed_configuration=filler,
        side_effects=filler, risks=filler, failure_modes=filler, resource_limits=filler,
        examples=filler, provenance=filler,
    )


def scan_round(**overrides) -> Organization:
    options = {
        "invocations": (
            Invocation(invocation_id="scan", operator="fs.scan", inputs=("root",)),
        ),
        "intent": "walk the tree",
        "sufficient": True,
        "publishes": "FileInventory",
        "summary": {"items": 2},
    }
    options.update(overrides)
    return Organization(**options)


@pytest.fixture
def harness(tmp_path: Path):
    source = tmp_path / "in"
    (source / "sub").mkdir(parents=True)
    (source / "a.pdf").write_bytes(b"one")
    (source / "sub" / "b.pdf").write_bytes(b"two")

    operators = build_registry()
    verifier = Verifier(operators)
    store = ArtifactStore(root=tmp_path / "artifacts", run_id="r")
    agent = ScriptedCapabilityAgent()
    runner = CapabilityRunner(
        executor=Executor(operators, verifier), agents={"*": agent}, artifacts=store
    )
    return runner, agent, store, source


def run(harness, *plans: Organization, capability: CapabilitySpec = SURVEY, rounds: int = 4):
    runner, agent, store, source = harness
    agent.queue(capability.id, *plans)
    return runner.run(
        capability=capability.model_copy(update={"max_rounds": rounds}),
        request="inventory the tree",
        context={"root": str(source)},
        budget=Budget(envelope=BudgetEnvelope(), items=0),
        workspace=Path(store.root).parent / "work",
        goal_id="inventoried",
    )


class TestSingleRound:
    def test_a_capability_that_finishes_first_time_runs_once(self, harness) -> None:
        outcome = run(harness, scan_round())
        assert outcome.sufficient
        assert len(outcome.rounds) == 1
        assert not outcome.exhausted

    def test_it_publishes_the_artifact_it_declared(self, harness) -> None:
        outcome = run(harness, scan_round())
        assert [item.kind for item in outcome.artifacts] == ["FileInventory"]
        assert outcome.artifacts[0].produced_by == "survey"
        assert outcome.artifacts[0].summary == {"items": 2}

    def test_components_still_run_through_the_executor(self, harness) -> None:
        outcome = run(harness, scan_round())
        assert [item.operator for item in outcome.rounds[0].results] == ["fs.scan"]
        assert outcome.context["count"] == 2


class TestMetaOrganization:
    def test_a_capability_sees_its_own_prior_rounds(self, harness) -> None:
        """The whole difference: without this it cannot carry on, only restart."""
        runner, agent, store, source = harness
        run(
            harness,
            scan_round(sufficient=False, publishes=None, intent="first look"),
            scan_round(intent="finish up"),
        )
        assert agent.seen_prior[0] == [], "the first round has no history"
        assert len(agent.seen_prior[1]) == 1
        assert agent.seen_prior[1][0]["intent"] == "first look"
        assert agent.seen_prior[1][0]["succeeded"] is True

    def test_it_keeps_going_until_it_says_it_is_finished(self, harness) -> None:
        outcome = run(
            harness,
            scan_round(sufficient=False, publishes=None),
            scan_round(sufficient=False, publishes=None),
            scan_round(),
        )
        assert len(outcome.rounds) == 3
        assert outcome.sufficient

    def test_partial_work_is_published_as_partial(self, harness) -> None:
        """How a capability says 'correct so far, and there is more' without failing."""
        outcome = run(
            harness,
            scan_round(sufficient=False, quality=Quality.PARTIAL, summary={"done": 20, "of": 60}),
            scan_round(),
        )
        first = outcome.artifacts[0]
        assert first.quality is Quality.PARTIAL
        assert first.summary == {"done": 20, "of": 60}

    def test_rounds_are_bounded(self, harness) -> None:
        """A capability that never converges still terminates."""
        outcome = run(harness, scan_round(sufficient=False, publishes=None), rounds=3)
        assert outcome.exhausted
        assert len(outcome.rounds) == 3
        assert not outcome.sufficient


class TestContainmentIsUnchanged:
    def test_a_component_outside_the_catalog_is_still_refused(self, harness) -> None:
        outcome = run(
            harness,
            Organization(
                invocations=(Invocation(invocation_id="x", operator="plan.assemble"),),
                intent="reach outside",
                sufficient=True,
            ),
        )
        assert outcome.rounds[0].denial is not None
        assert "outside the catalog" in outcome.rounds[0].denial

    def test_a_refused_round_publishes_nothing(self, harness) -> None:
        outcome = run(
            harness,
            Organization(
                invocations=(Invocation(invocation_id="c", operator="fs.commit"),),
                intent="write",
                sufficient=True,
                publishes="FileInventory",
            ),
        )
        assert outcome.artifacts == ()

    def test_a_refusal_reaches_the_next_round(self, harness) -> None:
        """A capability must be able to see why it was refused, or it repeats itself."""
        runner, agent, store, source = harness
        run(
            harness,
            Organization(
                invocations=(Invocation(invocation_id="x", operator="fs.commit"),),
                intent="write",
                sufficient=False,
            ),
            scan_round(),
        )
        assert agent.seen_prior[1][0]["denial"] is not None


class TestEvidenceNotPayloads:
    def test_the_capability_is_shown_shape_not_content(self, harness) -> None:
        runner, agent, store, source = harness
        store.put(kind="ExtractedContent", payload={"text": "SENSITIVE"}, produced_by="x")
        run(harness, scan_round())
        described = str(agent.seen_prior) + str(store.describe())
        assert "SENSITIVE" not in described


class TestUnusableResponses:
    """A response that violates its contract is a failed round, not a failed run.

    The capability has rounds precisely so it can correct itself, and it can only do that
    if the reason reaches the next one.
    """

    class _Malformed:
        """An agent that returns something unusable, then something valid."""

        def __init__(self, then: Organization) -> None:
            self.then = then
            self.calls = 0
            self.seen_prior: list[list[dict]] = []

        def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
            from system.contracts import ErrorCode
            from system.model_access import GatewayError

            self.calls += 1
            self.seen_prior.append(prior)
            if self.calls == 1:
                raise GatewayError(
                    "model response does not satisfy Organization", ErrorCode.MODEL_PERMANENT
                )
            return self.then, None

    def test_the_run_survives_and_the_capability_retries(self, tmp_path: Path) -> None:
        from system.capabilities import CapabilityRunner
        from system.components.catalog import build_registry
        from system.runtime.artifacts import ArtifactStore
        from system.runtime.executor import Executor
        from system.runtime.verifier import Verifier

        source = tmp_path / "in"
        source.mkdir()
        (source / "a.pdf").write_bytes(b"x")

        agent = self._Malformed(scan_round())
        operators = build_registry()
        verifier = Verifier(operators)
        store = ArtifactStore(root=tmp_path / "artifacts", run_id="r")
        runner = CapabilityRunner(
            executor=Executor(operators, verifier), agents={"*": agent}, artifacts=store
        )
        outcome = runner.run(
            capability=SURVEY,
            request="inventory it",
            context={"root": str(source)},
            budget=Budget(envelope=BudgetEnvelope(), items=0),
            workspace=tmp_path / "work",
        )
        assert outcome.sufficient, "the capability recovered on its second round"
        assert len(outcome.rounds) == 2
        assert outcome.rounds[0].denial is not None

    def test_the_reason_reaches_the_next_round(self, tmp_path: Path) -> None:
        from system.capabilities import CapabilityRunner
        from system.components.catalog import build_registry
        from system.runtime.artifacts import ArtifactStore
        from system.runtime.executor import Executor
        from system.runtime.verifier import Verifier

        source = tmp_path / "in"
        source.mkdir()
        agent = self._Malformed(scan_round())
        operators = build_registry()
        verifier = Verifier(operators)
        runner = CapabilityRunner(
            executor=Executor(operators, verifier),
            agents={"*": agent},
            artifacts=ArtifactStore(root=tmp_path / "artifacts", run_id="r"),
        )
        runner.run(
            capability=SURVEY,
            request="inventory it",
            context={"root": str(source)},
            budget=Budget(envelope=BudgetEnvelope(), items=0),
            workspace=tmp_path / "work",
        )
        assert "does not satisfy Organization" in str(agent.seen_prior[1])


class TestEvidenceOutranksSelfReport:
    """A round says what it did; its results say what happened. The results win.

    A live run put an empty FileInventory in the store this way: every component in the
    round failed, the organization still declared itself complete, the deterministic gate
    saw the kind it required, and the next capability was asked to extract text from an
    inventory of nothing.
    """

    def test_a_round_whose_components_failed_publishes_nothing(self, harness) -> None:
        outcome = run(
            harness,
            scan_round(
                invocations=(
                    Invocation(
                        invocation_id="scan", operator="fs.scan", parameters={"root": "/nowhere"}
                    ),
                )
            ),
            rounds=1,
        )
        assert not any(item.succeeded for round_ in outcome.rounds for item in round_.results)
        assert outcome.artifacts == ()

    def test_a_round_whose_components_failed_cannot_declare_itself_finished(
        self, harness
    ) -> None:
        outcome = run(
            harness,
            scan_round(
                invocations=(
                    Invocation(
                        invocation_id="scan", operator="fs.scan", parameters={"root": "/nowhere"}
                    ),
                )
            ),
            rounds=1,
        )
        assert outcome.exhausted
        assert not outcome.sufficient
