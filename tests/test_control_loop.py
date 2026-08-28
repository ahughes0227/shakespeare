"""The control loop: goals, gates, and who answers them.

Replaces the old spine tests. The loop must not know the names of the things it drives —
it knows which goals are open, what evidence exists, and which capability could produce
what is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from system.agent import FakeCapabilityAgent
from system.contracts import ChangeAction, Invocation, RouteDecision
from system.planner import ScriptedGoalPlanner
from system.runtime.artifacts import Quality
from system.runtime.goals import GateOutcome

from harness import build, org


@pytest.fixture
def harness(tmp_path: Path):
    runtime, request, audit, recorder = build(tmp_path)
    yield runtime, request, audit, recorder
    audit.close()


class TestGoalDrivenRun:
    def test_every_goal_is_satisfied_and_the_plan_commits(self, harness) -> None:
        runtime, request, _, _ = harness
        result = runtime.run(request)
        assert result.outcome == "committed", result.detail
        assert result.satisfied == {
            "inventoried",
            "readable",
            "convention_frozen",
            "named",
            "planned",
            "reviewed",
        }

    def test_output_mirrors_the_input_structure(self, harness) -> None:
        runtime, request, _, _ = harness
        runtime.run(request)
        output = Path(request.output_root)
        assert sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        ) == [
            "2024/q1/202401, ACME Corporation, INV-99812, PO-44117.pdf",
            "2024/q1/202401, ACME Corporation, INV-99813, PO-44118.pdf",
            "2024/q2/202404, Globex Ltd, INV-20001, PO-77310.pdf",
        ]

    def test_source_tree_is_never_mutated(self, harness) -> None:
        runtime, request, _, _ = harness
        before = sorted(p.name for p in Path(request.input_root).rglob("*") if p.is_file())
        runtime.run(request)
        assert sorted(
            p.name for p in Path(request.input_root).rglob("*") if p.is_file()
        ) == before

    def test_accounting_balances(self, harness) -> None:
        runtime, request, _, _ = harness
        result = runtime.run(request)
        assert result.plan is not None
        assert result.plan.balanced(3)
        assert result.plan.count(ChangeAction.CHANGED) == 3

    def test_a_dry_run_creates_no_output_root(self, harness) -> None:
        runtime, request, _, _ = harness
        result = runtime.run(request, commit=False)
        assert result.outcome == "planned"
        assert not Path(request.output_root).exists()


class TestIndependentWorkIsNotSequenced:
    def test_two_goals_open_together_once_their_dependency_is_met(self, harness) -> None:
        """Nothing forces the convention to wait for every file to be read (§3)."""
        runtime, _, _, _ = harness
        graph = runtime.workflows.get("rename_files").spec.graph
        assert [goal.id for goal in graph.open_goals(frozenset())] == ["inventoried"]
        opened = {goal.id for goal in graph.open_goals(frozenset({"inventoried"}))}
        assert opened == {"readable", "convention_frozen"}

    def test_the_planner_chooses_when_more_than_one_is_open(self, tmp_path: Path) -> None:
        planner = ScriptedGoalPlanner(
            route=RouteDecision(workflow_id="rename_files"),
            goal_order=["convention_frozen", "readable"],
        )
        runtime, request, audit, _ = build(tmp_path, planner=planner)
        result = runtime.run(request, commit=False)
        pursued = [attempt.goal_id for attempt in result.attempts]
        assert pursued.index("convention_frozen") < pursued.index("readable")
        assert "select_goal" in planner.calls
        audit.close()

    def test_no_choice_is_made_when_only_one_goal_is_open(self, tmp_path: Path) -> None:
        """Asking a model to pick between one option is exactly the call §13 avoids."""
        planner = ScriptedGoalPlanner(route=RouteDecision(workflow_id="rename_files"))
        runtime, request, audit, _ = build(tmp_path, planner=planner)
        runtime.run(request, commit=False)
        assert planner.calls.count("select_goal") == 1, "only the one ambiguous choice"
        audit.close()


class TestGates:
    def test_a_missing_artifact_blocks_rather_than_fails(self, tmp_path: Path) -> None:
        """The question cannot be asked yet — that is not the same as answered badly."""
        agents = {"*": FakeCapabilityAgent()}
        agents["*"].queue("survey", org(intent="do nothing", publishes=None))
        runtime, request, audit, _ = build(tmp_path, agents=agents)
        result = runtime.run(request, commit=False)
        assert result.outcome == "aborted"
        first = result.attempts[0]
        assert first.gate.outcome is GateOutcome.BLOCKED
        assert "FileInventory" in first.gate.missing_kinds
        audit.close()

    def test_a_failed_deterministic_check_is_insufficient(self, tmp_path: Path) -> None:
        from harness import rename_agent, seed_invoices, values_for

        # A resolve capability that names one of three: evidence exists, and it does not
        # account for the set. (Naming *none* is refused outright by the renderer now.)
        partial = values_for(seed_invoices(tmp_path / "in"))[:1]
        runtime, request, audit, _ = build(tmp_path, agents={"*": rename_agent(partial)})
        result = runtime.run(request, commit=False)
        named = [a for a in result.attempts if a.goal_id == "named"]
        assert named and named[-1].gate.outcome is GateOutcome.INSUFFICIENT
        assert "resolution_accounted" in named[-1].gate.failed_checks
        audit.close()

    def test_a_goal_that_cannot_be_satisfied_stops_the_run(self, tmp_path: Path) -> None:
        agents = {"*": FakeCapabilityAgent()}
        agents["*"].queue("survey", org(intent="produce nothing", publishes=None))
        runtime, request, audit, _ = build(tmp_path, agents=agents)
        result = runtime.run(request, commit=False)
        assert result.outcome == "aborted"
        assert "achieved nothing new" in result.detail
        assert "inventoried" in result.detail
        assert not Path(request.output_root).exists()
        audit.close()


class TestTheLoopIsGeneric:
    def test_no_goal_or_capability_name_appears_in_the_driver(self) -> None:
        """The loop must not learn the names of the things it drives (§12)."""
        import system

        root = Path(system.__file__).parent
        # Distinctive names only. "compose" is excluded deliberately: it is the Hydra
        # composition function, and a name collision would make this test lie.
        forbidden = (
            "rename_files",
            "inventoried",
            "convention_frozen",
            "survey",
            "acquire",
            "convene",
        )
        for module in (
            "runtime/control.py",
            "runtime/engine.py",
            "runtime/gating.py",
            "capabilities/runner.py",
        ):
            source = (root / module).read_text()
            for name in forbidden:
                assert name not in source, f"{module} names {name!r}"


class TestArtifacts:
    def test_each_capability_publishes_its_declared_evidence(self, harness) -> None:
        runtime, request, _, _ = harness
        result = runtime.run(request, commit=False)
        published = {
            artifact.kind
            for attempt in result.attempts
            for artifact in attempt.outcome.artifacts
        }
        assert published == {
            "FileInventory",
            "ExtractedContent",
            "NamingSpec",
            "ResolvedNames",
            "ChangePlan",
            "ReviewEvidence",
        }

    def test_partial_evidence_does_not_read_as_failure(self, tmp_path: Path) -> None:
        from harness import rename_agent, values_for

        runtime, request, audit, _ = build(tmp_path)
        items = values_for(Path(request.input_root))
        agent = rename_agent(items)
        agent.plans["acquire"] = [
            org(
                Invocation(
                    invocation_id="extract",
                    operator="doc.extract",
                    selections={"extract": "auto_chain"},
                    inputs=("root", "items"),
                ),
                publishes="ExtractedContent",
                quality=Quality.PARTIAL,
                summary={"usable": 2, "of": 3},
            )
        ]
        runtime.agents = {"*": agent}
        result = runtime.run(request, commit=False)
        assert "readable" in result.satisfied
        audit.close()


class TestTelemetry:
    def test_no_document_content_reaches_the_exporter(self, harness) -> None:
        runtime, request, _, recorder = harness
        runtime.run(request)
        shipped = recorder.serialized()
        for secret in ("ACME Corporation", "INV-99812", "Globex", "invoice body"):
            assert secret not in shipped


class TestARetryIsToldWhyItIsRetrying:
    """A retry that changes nothing is the same run again.

    Five live runs failed a goal three times each, and every attempt began with the same
    request and no idea what the gate had objected to. The verdict went to telemetry and
    the audit log — everywhere except the one place that could act on it.
    """

    @staticmethod
    def _contexts(tmp_path: Path) -> list[tuple[str, dict]]:
        from harness import rename_agent, seed_invoices, values_for

        seen: list[tuple[str, dict]] = []
        # Names one of three, so 'named' is rejected every time on its accounting.
        inner = rename_agent(values_for(seed_invoices(tmp_path / "in"))[:1])

        class Recording:
            def organize(self, *, capability, context, **rest):
                seen.append((capability.id, context))
                return inner.organize(capability=capability, context=context, **rest)

        runtime, request, audit, _ = build(tmp_path, agents={"*": Recording()})
        runtime.run(request, commit=False)
        audit.close()
        return seen

    def test_the_first_attempt_has_nothing_to_be_told(self, tmp_path: Path) -> None:
        resolve = [c for name, c in self._contexts(tmp_path) if name == "resolve"]
        assert resolve and "previous_attempt" not in resolve[0]

    def test_a_later_attempt_is_given_the_gate_verdict(self, tmp_path: Path) -> None:
        resolve = [c for name, c in self._contexts(tmp_path) if name == "resolve"]
        told = [c["previous_attempt"] for c in resolve if "previous_attempt" in c]
        assert told, "every attempt after the first should know why the last one failed"
        assert "resolution_accounted" in told[0]["failed_checks"]
        assert told[0]["attempt"] == 1

    def test_it_is_shown_in_full_rather_than_described(self, tmp_path: Path) -> None:
        """Describing the shape of a diagnosis informs nobody."""
        resolve = [c for name, c in self._contexts(tmp_path) if name == "resolve"]
        told = next(c["previous_attempt"] for c in resolve if "previous_attempt" in c)
        assert set(told) >= {"attempt", "failed_checks", "missing_evidence", "rationale"}

    def test_it_does_not_follow_the_run_into_another_goal(self, tmp_path: Path) -> None:
        """Each goal has its own history; inheriting another's would be noise at best."""
        others = [c for name, c in self._contexts(tmp_path) if name != "resolve"]
        assert others and not any("previous_attempt" in c for c in others)


class TestStagingPrecedesReview:
    """Stage, verify, then move — the review goal has to have something to review.

    A live run failed 'reviewed' three times with "none of its 60 entries are staged; all
    60 are missing". Nothing had been staged because staging waited for the loop to
    finish, and the goal that asks whether the staged tree matches the plan runs inside it.
    """

    def test_the_tree_is_staged_before_the_goal_that_reviews_it(self, harness) -> None:
        runtime, request, audit, _ = harness
        staged_at: list[int] = []

        original = runtime._stage_when_planned

        def record(**kwargs):
            original(**kwargs)
            staged_at.append(len(kwargs["context"].get("plan", {}).get("entries", [])))

        runtime._stage_when_planned = record  # type: ignore[method-assign]
        result = runtime.run(request, commit=False)
        assert result.outcome == "planned"
        assert any(count > 0 for count in staged_at), "staged while the loop was running"
        audit.close()

    def test_a_plan_is_staged_once_however_often_it_is_offered(self, harness) -> None:
        """Every satisfied goal offers the context; only the first one with a plan acts."""
        runtime, request, audit, _ = harness
        result = runtime.run(request, commit=False)
        assert result.outcome == "planned"
        staged = list(Path(runtime.workspace_root).glob("**/staging/**/*.pdf"))
        assert staged, "the plan was materialised"
        assert len(staged) == len(result.plan.entries)
        audit.close()


class TestAttemptsAreBoundedByProgressNotACount:
    """ADR 0003 left max_goal_attempts as a chosen number rather than a measured one.

    The number was never the right control. An attempt that achieved nothing will not
    achieve anything next time either, and one that is still making progress deserves
    another go — a fixed count gets both cases wrong in opposite directions.
    """

    def test_a_goal_still_making_progress_is_allowed_to_continue(
        self, tmp_path: Path
    ) -> None:
        """Progress means the retry is worth paying for, whatever the attempt number."""
        from system.runtime.control import Controller

        seen = []
        verdict = _verdict("still_short")
        context: dict = {"candidates": []}
        for _ in range(5):
            context["candidates"] = context["candidates"] + [{"item_id": "x"}]
            seen.append(_achievement_of(context, verdict))
        assert len(set(seen)) == len(seen), "each attempt reads as a different position"
        assert Controller.max_goal_attempts > 3, "and the ceiling leaves room for them"

    def test_two_attempts_that_reach_the_same_place_are_the_same_attempt(self) -> None:
        stuck = _verdict("balanced")
        context = {"candidates": [{"item_id": "x"}], "unrendered": []}
        assert _achievement_of(context, stuck) == _achievement_of(dict(context), stuck)

    def test_a_different_failure_is_not_the_same_place(self) -> None:
        """Same evidence, different objection: the run has learned something."""
        context = {"candidates": [{"item_id": "x"}]}
        assert _achievement_of(context, _verdict("balanced")) != _achievement_of(
            context, _verdict("collisions")
        )

    def test_runtime_bookkeeping_is_not_mistaken_for_progress(self) -> None:
        """Underscored keys are the runtime talking to itself, not work achieved."""
        verdict = _verdict("balanced")
        bare = {"candidates": [{"item_id": "x"}]}
        assert _achievement_of(bare, verdict) == _achievement_of(
            {**bare, "_completed": ["x", "y", "z"]}, verdict
        )


def _verdict(*failed_checks: str):
    return type(
        "Verdict",
        (),
        {"failed_checks": failed_checks, "missing_kinds": (), "rationale": "short"},
    )()


def _achievement_of(context, result):
    from system.runtime.control import _achievement

    return _achievement(context, result)
