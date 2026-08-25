"""Windowing: a stage too large for one model response finishes anyway.

This is the system recognising a limit and scheduling around it, rather than a limit
being worked around by hand outside the system.
"""

from __future__ import annotations

from pathlib import Path

from shakespeare.agent import FakeDomainAgent
from shakespeare.contracts import (
    Composition,
    Invocation,
    ObligationResult,
    StageDecision,
    StageVerdict,
)
from shakespeare.operators.planning import next_window
from shakespeare.planner import constrain

from test_rename_files import INVOICES, SPEC, _values, build, build_agents, seed_invoices


class TestWindowOperator:
    def _items(self, count: int) -> tuple[dict[str, str], ...]:
        return tuple({"item_id": f"i{n}"} for n in range(count))

    def test_takes_the_first_slice(self) -> None:
        result = next_window(items=self._items(50), window_size=20)
        assert result["window_size"] == 20
        assert result["remaining"] == 30
        assert not result["exhausted"]

    def test_skips_what_is_already_done(self) -> None:
        done = tuple(f"i{n}" for n in range(20))
        result = next_window(items=self._items(50), completed=done, window_size=20)
        assert [item["item_id"] for item in result["window"]][0] == "i20"
        assert result["completed_count"] == 20

    def test_accepts_records_as_well_as_ids(self) -> None:
        """The accumulator holds records; projecting ids out would be pure friction."""
        done = tuple({"item_id": f"i{n}", "name": "x"} for n in range(20))
        result = next_window(items=self._items(50), completed=done, window_size=20)
        assert result["completed_count"] == 20
        assert [item["item_id"] for item in result["window"]][0] == "i20"

    def test_reports_exhaustion_on_the_final_slice(self) -> None:
        result = next_window(items=self._items(10), window_size=20)
        assert result["exhausted"]
        assert result["remaining"] == 0

    def test_is_stateless_so_a_window_can_be_replayed(self) -> None:
        first = next_window(items=self._items(50), window_size=20)
        again = next_window(items=self._items(50), window_size=20)
        assert first == again

    def test_a_window_is_never_empty_while_work_remains(self) -> None:
        assert next_window(items=self._items(5), window_size=0)["window_size"] == 1


class TestContinueIsNotRetry:
    def test_a_continuation_may_stand_over_an_unmet_obligation(self) -> None:
        """The work is not finished yet, so the check failing is expected, not wrong."""
        unmet = (ObligationResult(obligation_id="resolution_accounted", passed=False),)
        verdict = StageVerdict(met=False, decision=StageDecision.CONTINUE, rationale="30 left")
        assert constrain(verdict, unmet).decision is StageDecision.CONTINUE

    def test_an_accept_over_an_unmet_obligation_is_still_refused(self) -> None:
        unmet = (ObligationResult(obligation_id="resolution_accounted", passed=False),)
        verdict = StageVerdict(met=True, decision=StageDecision.ACCEPT, rationale="fine")
        assert constrain(verdict, unmet).decision is StageDecision.RERUN


class TestStageWindowing:
    """The runtime loop: continuations accumulate and do not spend attempts."""

    def _windowed_agents(self, tmp_path: Path):
        source = seed_invoices(tmp_path / "in", INVOICES)
        items = _values(source, INVOICES)
        agents = build_agents(items)
        # Two windows of items, produced on successive continuations.
        agents["field_resolution"] = FakeDomainAgent()
        for chunk in (items[:2], items[2:]):
            agents["field_resolution"].compositions.setdefault("field_resolution", []).append(
                Composition(
                    domain_id="field_resolution",
                    invocations=(
                        Invocation(
                            invocation_id="render",
                            operator="name.render",
                            inputs=("spec",),
                            parameters={"items": list(chunk), "spec": SPEC},
                        ),
                    ),
                )
            )
        return agents

    def test_continuations_accumulate_into_one_result(self, tmp_path: Path) -> None:
        from test_rename_files import build_planner

        planner = build_planner()
        planner.queue_verdict(
            "resolve",
            StageVerdict(met=False, decision=StageDecision.CONTINUE, rationale="more remain"),
            StageVerdict(met=True, decision=StageDecision.ACCEPT),
        )
        runtime, request, audit, _ = build(
            tmp_path, planner=planner, agents=self._windowed_agents(tmp_path)
        )
        result = runtime.run(request, commit=False)
        assert result.outcome == "planned", result.detail
        assert result.plan is not None
        assert result.plan.balanced(3), "every item resolved across two windows"
        audit.close()

    def test_a_continuation_does_not_spend_an_attempt(self, tmp_path: Path) -> None:
        from test_rename_files import build_planner

        planner = build_planner()
        planner.queue_verdict(
            "resolve",
            StageVerdict(met=False, decision=StageDecision.CONTINUE, rationale="more"),
            StageVerdict(met=True, decision=StageDecision.ACCEPT),
        )
        runtime, request, audit, _ = build(
            tmp_path, planner=planner, agents=self._windowed_agents(tmp_path)
        )
        result = runtime.run(request, commit=False)
        resolve = next(o for o in result.stages if o.stage.name == "resolve")
        assert resolve.attempts == 1, "a window is progress, so attempts stay unspent"
        audit.close()

    def test_endless_continuation_still_terminates(self, tmp_path: Path) -> None:
        """max_windows bounds a stage that never says it is finished."""
        from test_rename_files import build_planner

        planner = build_planner()
        planner.queue_verdict(
            "resolve",
            StageVerdict(met=False, decision=StageDecision.CONTINUE, rationale="forever"),
        )
        runtime, request, audit, _ = build(
            tmp_path, planner=planner, agents=self._windowed_agents(tmp_path)
        )
        # The workflow holds its own stage tuple from registration, so bound that one.
        registered = runtime.workflows.get("rename_files")
        runtime.workflows._workflows["rename_files"] = registered.__class__(
            spec=registered.spec,
            card=registered.card,
            stages=tuple(
                stage.model_copy(update={"max_windows": 3})
                if stage.name == "resolve"
                else stage
                for stage in registered.stages
            ),
        )

        result = runtime.run(request, commit=False)
        assert result.outcome == "aborted"
        assert "max_windows" in result.detail
        audit.close()


class TestStagePackageDeclaresIt:
    def test_resolve_accumulates_and_bounds_its_windows(self) -> None:
        from shakespeare.stages import StageRegistry

        stage = StageRegistry().get("resolve@1.2.0")
        assert stage.accumulates == ("candidates", "unrendered")
        assert stage.max_windows > 1
        assert "batch.window" in stage.domain("field_resolution").catalog

    def test_windowing_is_never_implicit(self) -> None:
        """A stage that declares nothing accumulates nothing."""
        from shakespeare.stages import StageRegistry

        assert StageRegistry().get("intake@1.0.0").accumulates == ()
