"""Scheduling: an operator the runtime calls, and only when the work does not fit.

Sizing a batch is arithmetic over measured cost, so it is answered by code rather than by
a model deciding its own slicing every round — which is what a live sixty-invoice run
spent thirteen of twenty-one rounds failing to do. It stays an operator so the decision
is verified, journalled and traced; the runtime invokes it so no capability can schedule
itself.

The size is not fixed. A fixed size is only right when every item costs the same, and
invoices do not — so each batch is measured and the next one is sized from what the last
ones actually spent.
"""

from __future__ import annotations

from pathlib import Path

from system.capabilities import CapabilityRegistry, CapabilityRunner, CapabilitySpec
from system.capabilities.runner import Organization
from system.components.catalog import RUNTIME_ONLY, build_registry
from system.components.pure_transform.plans import plan_batch
from system.contracts import BudgetEnvelope, ErrorCode, Invocation
from system.model_access import GatewayError
from system.runtime.artifacts import ArtifactStore, Quality
from system.runtime.executor import Budget, Executor
from system.runtime.verifier import Verifier

DIVISIBLE = CapabilitySpec(
    id="divisible",
    version="1.0.0",
    standing_goal="Report something about every item.",
    catalog=frozenset({"text.normalize"}),
    produces=("ExtractedContent",),
    max_rounds=3,
    cost_per_item=674,
    divides="items",
)
WHOLE_SET = DIVISIBLE.model_copy(update={"id": "whole_set", "cost_per_item": None})


def items(count: int) -> list[dict[str, str]]:
    return [{"item_id": f"i{n}"} for n in range(count)]


def handed_over(context: dict) -> int:
    """How many items a capability was actually given this round."""
    value = context.get("items")
    return len(value) if isinstance(value, list) else int((value or {}).get("count", 0))


def spent(items_in_batch: int, tokens_each: int) -> dict[str, object]:
    return {"items": items_in_batch, "completion_tokens": items_in_batch * tokens_each}


class TestOnlyWhenNeeded:
    def test_a_set_that_fits_needs_no_scheduling(self) -> None:
        result = plan_batch(remaining=tuple(items(5)), capacity=16384, cost_per_item=674)
        assert result["needed"] is False
        assert result["batch_size"] == 5

    def test_a_set_that_does_not_fit_is_divided(self) -> None:
        result = plan_batch(remaining=tuple(items(60)), capacity=16384, cost_per_item=674)
        assert result["needed"] is True
        assert 0 < result["batch_size"] < 60
        assert result["remaining_count"] == 60 - result["batch_size"]

    def test_the_batch_is_taken_from_the_front(self) -> None:
        """So the caller can advance by simply dropping what it handed over."""
        result = plan_batch(remaining=tuple(items(60)), capacity=16384, cost_per_item=674)
        assert result["batch"] == items(60)[: result["batch_size"]]

    def test_headroom_is_left_for_structure(self) -> None:
        """A batch sized to the exact ceiling truncates, and a truncated round is wasted."""
        exact = 16384 // 674
        assert (
            plan_batch(remaining=tuple(items(200)), capacity=16384, cost_per_item=674)[
                "batch_size"
            ]
            < exact
        )

    def test_an_expensive_item_gets_a_smaller_batch(self) -> None:
        cheap = plan_batch(remaining=tuple(items(200)), capacity=16384, cost_per_item=100)
        dear = plan_batch(remaining=tuple(items(200)), capacity=16384, cost_per_item=2000)
        assert cheap["batch_size"] > dear["batch_size"]

    def test_a_batch_is_never_empty(self) -> None:
        result = plan_batch(remaining=tuple(items(3)), capacity=10, cost_per_item=100_000)
        assert result["batch_size"] >= 1


class TestItLearnsWhatTheWorkCosts:
    """The declared cost is a starting estimate. Measurement replaces it."""

    def test_cheap_items_speed_it_up(self) -> None:
        start = plan_batch(remaining=tuple(items(200)), capacity=16384, cost_per_item=674)
        faster = plan_batch(
            remaining=tuple(items(200)),
            capacity=16384,
            cost_per_item=674,
            observations=(spent(start["batch_size"], 200),),
        )
        assert faster["batch_size"] > start["batch_size"]

    def test_expensive_items_slow_it_down(self) -> None:
        start = plan_batch(remaining=tuple(items(200)), capacity=16384, cost_per_item=674)
        slower = plan_batch(
            remaining=tuple(items(200)),
            capacity=16384,
            cost_per_item=674,
            observations=(spent(start["batch_size"], 2000),),
        )
        assert slower["batch_size"] < start["batch_size"]

    def test_it_never_estimates_below_what_it_just_measured(self) -> None:
        """Averaging a spike away is how a run walks back into the same truncation."""
        result = plan_batch(
            remaining=tuple(items(200)),
            capacity=16384,
            cost_per_item=100,
            observations=(spent(14, 50), spent(14, 4000)),
        )
        assert result["estimate"] >= 4000

    def test_a_truncated_batch_forces_a_smaller_one(self) -> None:
        result = plan_batch(
            remaining=tuple(items(200)),
            capacity=16384,
            cost_per_item=674,
            observations=({"items": 14, "truncated": True},),
        )
        assert result["batch_size"] < 14

    def test_a_failed_batch_is_not_repeated_at_the_same_size(self) -> None:
        """Whatever went wrong, handing over the same amount again is not a new attempt."""
        result = plan_batch(
            remaining=tuple(items(200)),
            capacity=16384,
            cost_per_item=674,
            observations=({"items": 14, "completion_tokens": 0, "failed": True},),
        )
        assert result["batch_size"] < 14

    def test_repeated_truncation_keeps_shrinking(self) -> None:
        history: list[dict[str, object]] = []
        sizes = []
        for _ in range(3):
            size = plan_batch(
                remaining=tuple(items(200)),
                capacity=16384,
                cost_per_item=674,
                observations=tuple(history),
            )["batch_size"]
            sizes.append(size)
            history.append({"items": size, "truncated": True})
        assert sizes == sorted(sizes, reverse=True) and sizes[-1] < sizes[0]

    def test_growth_is_capped_so_one_cheap_batch_cannot_undo_caution(self) -> None:
        result = plan_batch(
            remaining=tuple(items(500)),
            capacity=16384,
            cost_per_item=674,
            observations=({"items": 3, "truncated": True}, spent(3, 5)),
        )
        assert result["batch_size"] <= 6

    def test_a_ceiling_once_proven_still_binds(self) -> None:
        """A cheap batch after a truncation does not license going back over the line."""
        result = plan_batch(
            remaining=tuple(items(500)),
            capacity=16384,
            cost_per_item=674,
            observations=(
                {"items": 20, "truncated": True},
                spent(10, 20),
                spent(10, 20),
                spent(10, 20),
            ),
        )
        assert result["batch_size"] <= 10


class TestTheRuntimeSchedules:
    def _runner(self, tmp_path: Path, agent):
        operators = build_registry()
        return CapabilityRunner(
            executor=Executor(operators, Verifier(operators)),
            agents={"*": agent},
            artifacts=ArtifactStore(root=tmp_path / "artifacts", run_id="r"),
            capacity=16384,
        )

    class _Recorder:
        """Records how many items it was handed each time it was asked."""

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
            self.batch_sizes.append(handed_over(context))
            return (
                Organization(
                    invocations=(
                        Invocation(
                            invocation_id="n",
                            operator="text.normalize",
                            parameters={"values": {"v": "x"}},
                        ),
                    ),
                    intent="handle this batch",
                    sufficient=True,
                    publishes="ExtractedContent",
                    quality=Quality.PARTIAL,
                ),
                None,
            )

    def test_a_large_set_is_handed_over_one_batch_at_a_time(self, tmp_path: Path) -> None:
        agent = self._Recorder()
        outcome = self._runner(tmp_path, agent).run(
            capability=DIVISIBLE,
            request="report on them",
            context={"items": items(60)},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="200"), items=60),
            workspace=tmp_path / "work",
        )
        assert len(agent.batch_sizes) > 1, "the work was divided"
        assert sum(agent.batch_sizes) == 60, "and every item was handed over exactly once"
        assert outcome.sufficient

    def test_a_small_set_is_handed_over_whole(self, tmp_path: Path) -> None:
        """One batch, and the capability is not told it is one of many."""
        agent = self._Recorder()
        self._runner(tmp_path, agent).run(
            capability=DIVISIBLE,
            request="report on them",
            context={"items": items(5)},
            budget=Budget(envelope=BudgetEnvelope(), items=5),
            workspace=tmp_path / "work",
        )
        assert agent.batch_sizes == [5]

    def test_a_whole_set_capability_is_never_divided(self, tmp_path: Path) -> None:
        """Collision resolution and plan assembly need the whole set at once."""
        agent = self._Recorder()
        self._runner(tmp_path, agent).run(
            capability=WHOLE_SET,
            request="do it all at once",
            context={"items": items(500)},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="200"), items=500),
            workspace=tmp_path / "work",
        )
        assert agent.batch_sizes == [500]

    def test_rounds_correct_mistakes_rather_than_advance_work(self, tmp_path: Path) -> None:
        """Progress is the runtime's job; a round exists to fix a bad response."""

        class Stumbling:
            def __init__(self) -> None:
                self.calls = 0

            def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
                self.calls += 1
                # Fail the first round of the first batch, then behave.
                bad = self.calls == 1
                return (
                    Organization(
                        invocations=(
                            Invocation(
                                invocation_id="n",
                                operator="fs.commit" if bad else "text.normalize",
                                parameters={} if bad else {"values": {"v": "x"}},
                            ),
                        ),
                        intent="retry" if not bad else "reach outside",
                        sufficient=not bad,
                        publishes=None if bad else "ExtractedContent",
                    ),
                    None,
                )

        agent = Stumbling()
        outcome = self._runner(tmp_path, agent).run(
            capability=DIVISIBLE,
            request="report",
            context={"items": items(5)},
            budget=Budget(envelope=BudgetEnvelope(), items=5),
            workspace=tmp_path / "work",
        )
        assert outcome.sufficient, "the second round corrected the first"
        assert outcome.rounds[0].denial is not None


    def test_a_batch_cut_off_is_retried_smaller(self, tmp_path: Path) -> None:
        """The whole point of the feedback: too big is a recoverable mistake."""

        class Truncating:
            def __init__(self) -> None:
                self.sizes: list[int] = []

            def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
                size = handed_over(context)
                self.sizes.append(size)
                if size > 4:
                    raise GatewayError(
                        "cut off at the limit", ErrorCode.MODEL_PERMANENT, truncated=True
                    )
                return (
                    Organization(
                        invocations=(
                            Invocation(
                                invocation_id="n",
                                operator="text.normalize",
                                parameters={"values": {"v": "x"}},
                            ),
                        ),
                        intent="handle this batch",
                        sufficient=True,
                        publishes="ExtractedContent",
                    ),
                    None,
                )

        agent = Truncating()
        outcome = self._runner(tmp_path, agent).run(
            capability=DIVISIBLE,
            request="report",
            context={"items": items(24)},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="400"), items=24),
            workspace=tmp_path / "work",
        )
        assert min(agent.sizes) <= 4, "it backed off after being cut off"
        assert outcome.sufficient, "and then finished the work"

    def test_a_batch_that_keeps_failing_is_given_up_on(self, tmp_path: Path) -> None:
        """Each retry is billed, so backing off cannot go on forever."""

        class AlwaysTruncating:
            def __init__(self) -> None:
                self.calls = 0

            def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
                self.calls += 1
                raise GatewayError("cut off", ErrorCode.MODEL_PERMANENT, truncated=True)

        agent = AlwaysTruncating()
        outcome = self._runner(tmp_path, agent).run(
            capability=DIVISIBLE,
            request="report",
            context={"items": items(24)},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="400"), items=24),
            workspace=tmp_path / "work",
        )
        assert outcome.exhausted
        assert agent.calls < 40, "it stopped rather than retrying indefinitely"


class TestContainment:
    def test_no_capability_can_schedule_itself(self) -> None:
        """The runtime divides the work; a capability decides what the answer is."""
        registry = CapabilityRegistry()
        for capability_id in registry.ids():
            assert "schedule.plan" not in registry.get(capability_id).catalog

    def test_scheduling_is_not_a_mutation(self) -> None:
        assert "schedule.plan" not in RUNTIME_ONLY

    def test_only_a_measured_capability_is_divided(self) -> None:
        """cost_per_item is measured, so an unset one means 'this is not divisible'."""
        registry = CapabilityRegistry()
        divisible = {
            capability_id
            for capability_id in registry.ids()
            if registry.get(capability_id).cost_per_item is not None
        }
        assert divisible == {"acquire", "resolve", "transcribe"}


class TestBatchingIsInvisibleDownstream:
    """A capability may be asked in pieces; nothing after it should be able to tell.

    Both of these were live-run failures. The gate was handed the last batch as though it
    were the whole inventory and accepted "30 of 30" for a sixty-file run, and the second
    batch's extractions silently replaced the first's.
    """

    def _runner(self, tmp_path: Path, agent):
        operators = build_registry()
        return CapabilityRunner(
            executor=Executor(operators, Verifier(operators)),
            agents={"*": agent},
            artifacts=ArtifactStore(root=tmp_path / "artifacts", run_id="r"),
            capacity=16384,
        )

    class _Extractor:
        """Writes a per-item result for exactly the items it was handed."""

        def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
            return (
                Organization(
                    invocations=(
                        Invocation(
                            invocation_id="n",
                            operator="text.normalize",
                            parameters={"values": {"v": "x"}},
                        ),
                    ),
                    intent="extract this batch",
                    sufficient=True,
                    publishes="ExtractedContent",
                ),
                None,
            )

    def _run(self, tmp_path: Path, agent, count: int):
        return self._runner(tmp_path, agent).run(
            capability=DIVISIBLE,
            request="extract them",
            context={"items": items(count)},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="400"), items=count),
            workspace=tmp_path / "work",
        )

    def test_the_whole_set_is_restored_when_the_work_is_done(self, tmp_path: Path) -> None:
        """Otherwise the gate judges the last batch and calls it the inventory."""
        outcome = self._run(tmp_path, self._Extractor(), 60)
        assert len(outcome.context["items"]) == 60

    def test_the_whole_set_is_restored_even_when_it_fails(self, tmp_path: Path) -> None:
        class Refusing:
            def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
                return (
                    Organization(
                        invocations=(
                            Invocation(
                                invocation_id="n", operator="fs.commit", parameters={}
                            ),
                        ),
                        intent="reach outside",
                        sufficient=False,
                    ),
                    None,
                )

        outcome = self._run(tmp_path, Refusing(), 60)
        assert outcome.exhausted
        assert len(outcome.context["items"]) == 60

    def test_per_item_results_accumulate_across_batches(self, tmp_path: Path) -> None:
        """An operator's output replaces its key, which across batches erases the last."""

        class PerItem:
            def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
                return (
                    Organization(
                        invocations=(
                            Invocation(
                                invocation_id="n",
                                operator="text.normalize",
                                parameters={"values": {"v": "x"}},
                            ),
                        ),
                        intent="extract",
                        sufficient=True,
                        publishes="ExtractedContent",
                    ),
                    None,
                )

        runner = self._runner(tmp_path, PerItem())
        # Stand in for an operator that reports one row per item of the batch it saw.
        original = runner.executor.execute

        def execute(composition, domain, **kwargs):
            results = original(composition, domain, **kwargs)
            batch = kwargs["stage_inputs"].get("items") or []
            scheduling = composition.invocations[0].operator == "schedule.plan"
            for item in results:
                if item.output is not None and not scheduling:
                    item.output["extractions"] = [
                        {"item_id": row["item_id"], "text": "x"} for row in batch
                    ]
            return results

        runner.executor.execute = execute  # type: ignore[method-assign]
        outcome = runner.run(
            capability=DIVISIBLE,
            request="extract them",
            context={"items": items(60)},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="400"), items=60),
            workspace=tmp_path / "work",
        )
        extracted = {row["item_id"] for row in outcome.context["extractions"]}
        assert extracted == {row["item_id"] for row in items(60)}

    def test_the_scheduling_decision_is_journalled(self, tmp_path: Path) -> None:
        """The audit log must show why a capability was asked what it was asked."""
        outcome = self._run(tmp_path, self._Extractor(), 60)
        assert outcome.scheduling
        operators = {
            invocation.operator
            for composition, _ in outcome.scheduling
            for invocation in composition.invocations
        }
        assert operators == {"schedule.plan"}


class TestTheBatchIsShownNotDescribed:
    """A capability that must read the content has to be given the content.

    Three live attempts died on this: "the available context provides only aggregate item
    and extraction counts, not the individual item IDs, paths, extensions, or extracted
    invoice text". The batch is sized to fit one response — this is what it was sized for.
    """

    def _seen(self, tmp_path: Path, capability: CapabilitySpec, context: dict):
        seen: list[dict] = []

        class Watcher:
            def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
                seen.append(context)
                return (
                    Organization(
                        invocations=(
                            Invocation(
                                invocation_id="n",
                                operator="text.normalize",
                                parameters={"values": {"v": "x"}},
                            ),
                        ),
                        intent="read it",
                        sufficient=True,
                        publishes="ExtractedContent",
                    ),
                    None,
                )

        operators = build_registry()
        CapabilityRunner(
            executor=Executor(operators, Verifier(operators)),
            agents={"*": Watcher()},
            artifacts=ArtifactStore(root=tmp_path / "artifacts", run_id="r"),
            capacity=16384,
        ).run(
            capability=capability,
            request="read them",
            context=context,
            budget=Budget(envelope=BudgetEnvelope(operator_calls="400"), items=60),
            workspace=tmp_path / "work",
        )
        return seen

    def test_the_batch_arrives_in_full(self, tmp_path: Path) -> None:
        seen = self._seen(tmp_path, DIVISIBLE, {"items": items(60)})
        assert all(isinstance(context["items"], list) for context in seen)
        assert sum(len(context["items"]) for context in seen) == 60

    def test_the_evidence_for_those_items_comes_with_them(self, tmp_path: Path) -> None:
        """Rows for other batches would be noise, and the whole set would not fit."""
        text = [{"item_id": row["item_id"], "text": "body"} for row in items(60)]
        seen = self._seen(tmp_path, DIVISIBLE, {"items": items(60), "extractions": text})
        first = seen[0]
        assert {row["item_id"] for row in first["extractions"]} == {
            row["item_id"] for row in first["items"]
        }

    def test_an_undivided_capability_is_still_only_described(self, tmp_path: Path) -> None:
        """Nothing sized the set for it, so handing it over whole could not be safe."""
        seen = self._seen(tmp_path, WHOLE_SET, {"items": items(500)})
        assert seen[0]["items"] == {"kind": "list", "count": 500}


class TestItWeighsItemsRatherThanCountingThem:
    """A one-line receipt and a forty-line statement are both one item.

    ADR 0003 left this open: batch cost was measured per capability, so a corpus with
    real variance got every batch sized for the average and the heavy ones truncated.
    """

    @staticmethod
    def _weighted(sizes: list[int]) -> tuple[tuple, tuple]:
        rows = tuple({"item_id": f"i{n}"} for n in range(len(sizes)))
        return rows, tuple(sizes)

    def test_a_heavy_run_of_items_gets_a_smaller_batch(self) -> None:
        light, weights_light = self._weighted([10] * 100)
        heavy, weights_heavy = self._weighted([1000] * 100)
        thin = plan_batch(
            remaining=light, weights=weights_light, capacity=16384, cost_per_item=674
        )
        thick = plan_batch(
            remaining=heavy, weights=weights_heavy, capacity=16384, cost_per_item=674
        )
        # Same declared cost, same count, same ceiling: only the material differs.
        assert thin["batch_size"] == thick["batch_size"], "no history yet, so no evidence"
        # Once one batch has been measured, weight is what the next one is sized by.
        seen = ({"items": 10, "weight": 10 * 1000, "completion_tokens": 8000},)
        assert (
            plan_batch(
                remaining=heavy,
                weights=weights_heavy,
                capacity=16384,
                cost_per_item=674,
                observations=seen,
            )["batch_size"]
            < plan_batch(
                remaining=light,
                weights=weights_light,
                capacity=16384,
                cost_per_item=674,
                observations=seen,
            )["batch_size"]
        )

    def test_a_mixed_set_takes_more_light_items_than_heavy_ones(self) -> None:
        """The batch is filled by material, so its size varies with what it meets."""
        mixed, weights = self._weighted([50] * 20 + [5000] * 20)
        first = plan_batch(
            remaining=mixed,
            weights=weights,
            capacity=16384,
            cost_per_item=674,
            observations=({"items": 5, "weight": 250, "completion_tokens": 500},),
        )
        # It should clear a good part of the light run before the heavy ones stop it.
        assert first["batch_size"] > 5
        assert first["batch_size"] <= 21, "and it must stop once the heavy items begin"

    def test_the_batch_reports_what_it_carried(self) -> None:
        rows, weights = self._weighted([100] * 30)
        plan = plan_batch(
            remaining=rows, weights=weights, capacity=16384, cost_per_item=674
        )
        assert plan["batch_weight"] == plan["batch_size"] * 100

    def test_no_weights_is_the_old_count_behaviour_exactly(self) -> None:
        rows = tuple({"item_id": f"i{n}"} for n in range(200))
        assert (
            plan_batch(remaining=rows, capacity=16384, cost_per_item=674)["batch_size"]
            == plan_batch(
                remaining=rows, weights=(1,) * 200, capacity=16384, cost_per_item=674
            )["batch_size"]
        )

    def test_one_item_over_the_allowance_is_still_attempted(self) -> None:
        """Refusing to hand it over would stall the run; the backoff is what handles it."""
        rows, weights = self._weighted([10_000_000])
        assert plan_batch(
            remaining=rows, weights=weights, capacity=16384, cost_per_item=674
        )["batch_size"] == 1


class TestProgressSurvivesAcrossAttempts:
    """A second attempt must start from where the first got to, not behind it.

    A live calibration run watched a goal reach forty-six of sixty items and then start
    its next attempt with forty-six still to do: each batch was assigning its own results
    back over everything earlier attempts had accumulated.
    """

    def _runner(self, tmp_path: Path, agent):
        operators = build_registry()
        return CapabilityRunner(
            executor=Executor(operators, Verifier(operators)),
            agents={"*": agent},
            artifacts=ArtifactStore(root=tmp_path / "artifacts", run_id="r"),
            capacity=16384,
        )

    class _Reporter:
        def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
            return (
                Organization(
                    invocations=(
                        Invocation(
                            invocation_id="n",
                            operator="text.normalize",
                            parameters={"values": {"v": "x"}},
                        ),
                    ),
                    intent="report",
                    sufficient=True,
                    publishes="ExtractedContent",
                ),
                None,
            )

    def test_earlier_rows_are_not_replaced_by_a_later_batch(self, tmp_path: Path) -> None:
        earlier = [{"item_id": f"i{n}", "text": "done"} for n in range(20)]
        outcome = self._runner(tmp_path, self._Reporter()).run(
            capability=DIVISIBLE,
            request="carry on",
            context={"items": items(60), "extractions": earlier},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="400"), items=60),
            workspace=tmp_path / "work",
        )
        kept = {row["item_id"] for row in outcome.context["extractions"]}
        assert kept >= {row["item_id"] for row in earlier}, "what was already done is still done"

    def test_a_second_attempt_does_not_redo_a_finished_one(self, tmp_path: Path) -> None:
        """The property that actually matters: attempts converge rather than oscillate."""
        runner = self._runner(tmp_path, self._Reporter())
        original = runner.executor.execute

        def execute(composition, domain, **kwargs):
            results = original(composition, domain, **kwargs)
            batch = kwargs["stage_inputs"].get("items") or []
            if composition.invocations[0].operator != "schedule.plan":
                for item in results:
                    if item.output is not None:
                        item.output["extractions"] = [
                            {"item_id": row["item_id"], "text": "x"} for row in batch
                        ]
            return results

        runner.executor.execute = execute  # type: ignore[method-assign]

        handed: list[int] = []
        plan_batch_of = runner._plan_batch

        def counting(capability, remaining, weights, observations, **kwargs):
            handed.append(len(remaining))
            return plan_batch_of(capability, remaining, weights, observations, **kwargs)

        runner._plan_batch = counting  # type: ignore[method-assign]

        # Progress is judged by what this capability's own components produce, so the
        # catalog has to contain the one that produces it.
        extracting = DIVISIBLE.model_copy(
            update={"catalog": frozenset({"text.normalize", "doc.extract"})}
        )

        def attempt(context: dict):
            outcome = runner.run(
                capability=extracting,
                request="carry on",
                context=context,
                budget=Budget(envelope=BudgetEnvelope(operator_calls="400"), items=60),
                workspace=tmp_path / "work",
            )
            context.update(outcome.context)
            return outcome

        context: dict = {"items": items(60)}
        assert attempt(context).sufficient
        assert handed and handed[0] == 60
        scheduled = len(handed)

        assert attempt(context).sufficient, "it still answers"
        assert len(handed) == scheduled, "but it schedules nothing, having nothing left to do"
        assert len(context["items"]) == 60, "and the set it hands on is still whole"


class TestABatchIsFinishedWhenItsItemsAre:
    """Saying a batch is answered is not the same as answering it.

    A round with no components succeeds trivially, so a capability could declare its batch
    done having produced nothing for it. Five live attempts each said yes to fourteen items
    and moved none of them, and the goal failed six times without advancing once.
    """

    EXTRACTING = DIVISIBLE.model_copy(
        update={"catalog": frozenset({"text.normalize", "doc.extract"})}
    )

    def _runner(self, tmp_path: Path, agent):
        """A runner whose components report one row per item they were told to handle."""
        operators = build_registry()
        runner = CapabilityRunner(
            executor=Executor(operators, Verifier(operators)),
            agents={"*": agent},
            artifacts=ArtifactStore(root=tmp_path / "artifacts", run_id="r"),
            capacity=16384,
        )
        original = runner.executor.execute

        def execute(composition, domain, **kwargs):
            results = original(composition, domain, **kwargs)
            invocation = composition.invocations[0]
            if invocation.operator == "schedule.plan":
                return results
            handled = invocation.parameters.get("handled")
            if handled is None:
                handled = kwargs["stage_inputs"].get("items") or []
            for item in results:
                if item.output is not None:
                    item.output["extractions"] = [
                        {"item_id": row["item_id"], "text": "x"} for row in handled
                    ]
            return results

        runner.executor.execute = execute  # type: ignore[method-assign]
        return runner

    @staticmethod
    def _extract(handled: list | None = None) -> Organization:
        parameters: dict = {"values": {"v": "x"}}
        if handled is not None:
            parameters["handled"] = handled
        return Organization(
            invocations=(
                Invocation(
                    invocation_id="x", operator="text.normalize", parameters=parameters
                ),
            ),
            intent="extract",
            sufficient=True,
            publishes="ExtractedContent",
        )

    class _Claiming:
        """Declares every batch answered while producing nothing for any of them."""

        def __init__(self) -> None:
            self.calls = 0

        def organize(self, *, capability, request, artifacts, context, prior, catalog_summary):
            self.calls += 1
            return (
                Organization(intent="already done", sufficient=True, publishes=None),
                None,
            )

    def _run(self, tmp_path: Path, agent, runner=None):
        return (runner or self._runner(tmp_path, agent)).run(
            capability=self.EXTRACTING,
            request="extract them",
            context={"items": items(60), "root": str(tmp_path)},
            budget=Budget(envelope=BudgetEnvelope(operator_calls="400"), items=60),
            workspace=tmp_path / "work",
        )

    def test_an_empty_claim_does_not_finish_a_batch(self, tmp_path: Path) -> None:
        outcome = self._run(tmp_path, self._Claiming())
        assert outcome.exhausted, "the runtime checked rather than taking its word"
        assert not outcome.sufficient

    def test_it_gives_up_rather_than_claiming_its_way_through(self, tmp_path: Path) -> None:
        """Bounded, because every attempt at an unaccounted batch is billed."""
        agent = self._Claiming()
        self._run(tmp_path, agent)
        assert agent.calls <= 20

    def test_a_batch_that_does_the_work_is_finished(self, tmp_path: Path) -> None:
        class Working:
            def organize(inner, *, capability, request, artifacts, context, prior, catalog_summary):
                return self._extract(), None

        outcome = self._run(tmp_path, Working())
        assert outcome.sufficient
        assert len(outcome.context["extractions"]) == 60

    def test_only_the_unaccounted_items_are_offered_again(self, tmp_path: Path) -> None:
        """Redoing the part of a batch that worked would spend the run's budget twice."""

        class Partial:
            """Accounts for everything except the last item of whatever it is handed."""

            def organize(inner, *, capability, request, artifacts, context, prior, catalog_summary):
                batch = list(context.get("items") or [])
                return self._extract(batch[:-1] if len(batch) > 1 else []), None

        runner = self._runner(tmp_path, Partial())
        offered: list[int] = []
        original = runner._plan_batch

        def counting(capability, remaining, weights, observations, **kwargs):
            offered.append(len(remaining))
            return original(capability, remaining, weights, observations, **kwargs)

        runner._plan_batch = counting  # type: ignore[method-assign]
        self._run(tmp_path, Partial(), runner=runner)
        assert offered == sorted(offered, reverse=True), "the outstanding set only shrinks"
        assert len(set(offered)) == len(offered), "and it never stands still"
