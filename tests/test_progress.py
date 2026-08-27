"""Windows must advance without the capability threading its own progress.

A live run took the same first twenty items eight times over: the model never bound
`completed`, so `batch.window` had no idea anything had been done. The trace showed it as
`completed_count: 0` on every round. Scheduling cannot depend on a prompt being followed.
"""

from __future__ import annotations

from pathlib import Path

from shakespeare.capabilities import CapabilityRunner, CapabilitySpec
from shakespeare.capabilities.runner import Organization, _record_progress
from shakespeare.components.builtin import build_registry
from shakespeare.components.runners import pure_transform
from shakespeare.contracts import BudgetEnvelope, Invocation
from shakespeare.runtime.artifacts import ArtifactStore, Quality
from shakespeare.runtime.executor import Budget, Executor
from shakespeare.runtime.verifier import Verifier

SLICER = CapabilitySpec(
    id="slicer",
    version="1.0.0",
    standing_goal="Take work in slices and process each one.",
    catalog=frozenset({"batch.window", "fs.scan", "doc.extract"}),
    config_groups=frozenset({"extract", "schedule"}),
    produces=("ExtractedContent",),
    max_rounds=6,
)


class TestProgressRecord:
    def test_item_ids_accumulate_across_rounds(self) -> None:
        working: dict = {}
        working["candidates"] = [{"item_id": "a"}, {"item_id": "b"}]
        _record_progress(working)
        working["candidates"] = [{"item_id": "c"}]
        _record_progress(working)
        assert working["_completed"] == ["a", "b", "c"]

    def test_the_same_item_is_not_counted_twice(self) -> None:
        working: dict = {"candidates": [{"item_id": "a"}]}
        _record_progress(working)
        _record_progress(working)
        assert working["_completed"] == ["a"]

    def test_quarantined_items_count_as_dealt_with(self) -> None:
        """An item that could not be resolved is finished with, not still pending."""
        working: dict = {"unrendered": [{"item_id": "q", "reason": "missing_field"}]}
        _record_progress(working)
        assert working["_completed"] == ["q"]

    def test_it_stays_out_of_prompts(self) -> None:
        from shakespeare.capabilities.runner import _summarise

        assert "_completed" not in _summarise({"_completed": ["a"], "items": []})


class TestWindowAdvances:
    def _window(self, items, **extra):
        return pure_transform(
            {"operation": "next_window", "items": items, "window_size": 20, **extra},
            Path("/tmp"),
        )

    def _items(self, count: int):
        return [{"item_id": f"i{n}"} for n in range(count)]

    def test_the_runtime_record_is_used_when_the_agent_omits_it(self) -> None:
        """The fix: scheduling no longer depends on the agent remembering."""
        done = [f"i{n}" for n in range(20)]
        result = self._window(self._items(60), _completed=done)
        assert result["completed_count"] == 20
        assert result["remaining"] == 20
        assert result["window"][0]["item_id"] == "i20"

    def test_an_explicit_binding_still_wins(self) -> None:
        result = self._window(
            self._items(60),
            completed=[f"i{n}" for n in range(40)],
            _completed=[f"i{n}" for n in range(20)],
        )
        assert result["completed_count"] == 40

    def test_without_either_it_starts_from_the_beginning(self) -> None:
        assert self._window(self._items(60))["completed_count"] == 0


class TestEndToEndAdvance:
    def test_windows_advance_across_rounds_without_the_agent_threading_them(
        self, tmp_path: Path
    ) -> None:
        """The whole point: a set larger than one response finishes anyway.

        The agent deliberately never binds `completed`, which is exactly what the live
        model did. Progress must come from the runtime's own record instead.
        """
        source = tmp_path / "in"
        source.mkdir()
        for n in range(50):
            (source / f"f{n:03d}.txt").write_text(f"document number {n}")

        class Slicer:
            def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
                return (
                    Organization(
                        invocations=(
                            Invocation(
                                invocation_id="scan", operator="fs.scan", inputs=("root",)
                            ),
                            Invocation(
                                invocation_id="win",
                                operator="batch.window",
                                inputs=("scan",),
                                parameters={"window_size": 20},
                            ),
                            Invocation(
                                invocation_id="ext",
                                operator="doc.extract",
                                selections={"extract": "auto_chain"},
                                inputs=("root", "win"),
                                bindings={"items": "window"},
                            ),
                        ),
                        intent="slice and read",
                        sufficient=len(prior) >= 2,
                        publishes="ExtractedContent",
                        quality=Quality.COMPLETE if len(prior) >= 2 else Quality.PARTIAL,
                    ),
                    None,
                )

        operators = build_registry()
        verifier = Verifier(operators)
        runner = CapabilityRunner(
            executor=Executor(operators, verifier),
            agents={"*": Slicer()},
            artifacts=ArtifactStore(root=tmp_path / "artifacts", run_id="r"),
        )
        outcome = runner.run(
            capability=SLICER,
            request="read them all",
            context={"root": str(source)},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="200"), items=50),
            workspace=tmp_path / "work",
        )

        def offset_of(round_) -> int | None:
            for result in round_.results:
                if result.operator == "batch.window" and result.output:
                    return int(result.output["completed_count"])
            return None

        offsets = [n for n in (offset_of(r) for r in outcome.rounds) if n is not None]
        assert offsets == [0, 20, 40], f"each window starts where the last stopped: {offsets}"
        assert outcome.sufficient
