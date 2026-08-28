"""What runs measured, and which declared constant that evidence supports.

Three numbers decide how the system behaves and none is measured: `cost_per_item` in a
capability manifest, the confidence floor in a config group, and `max_goal_attempts` in a
field default. Each run produces the evidence that would settle one of them, uses it once,
and throws it away at the run boundary.

These tests pin two things. That the evidence survives the boundary — and that it enters
a run only through a manifest a person edited, because a run whose behaviour depends on
unpinned state is no longer determined by its journal, and `replay` would stop meaning
what it says.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from system import measurements as memory
from system.contracts import Bound, Measurement, MeasurementKind
from system.runtime.audit import AuditStore


@pytest.fixture
def store(tmp_path: Path) -> Any:
    audit = AuditStore(tmp_path / "audit.sqlite3")
    for run_id in ("run-1", "run-2", "run-3"):
        audit.record_run(
            run_id=run_id,
            workflow_id="rename_files",
            workflow_version="1.0.0",
            workflow_digest="d",
            request_digest="q",
            input_root_digest="i",
        )
    yield audit
    audit.close()


def cost(
    *,
    value: float,
    weight: float,
    count: int,
    outcome: bool = True,
    bound: Bound | None = None,
    model: str = "gpt-5-mini",
) -> Measurement:
    return Measurement(
        kind=MeasurementKind.SCHEDULE_COST,
        subject="resolve@1.0.0",
        resolved_model=model,
        value=value,
        weight=weight,
        count=count,
        outcome=outcome,
        bound=bound,
    )


def rows(store: AuditStore, measurements: list[Measurement], run_id: str = "run-1") -> list[dict]:
    store.record_measurements(run_id=run_id, measurements=measurements)
    return store.measurements(kind=MeasurementKind.SCHEDULE_COST)


def spread_over_runs(store: AuditStore, times: int = 12, **kwargs: Any) -> list[dict[str, Any]]:
    """The same observation repeated across three runs, which is the minimum evidence."""
    for index in range(times):
        store.record_measurements(
            run_id=f"run-{index % 3 + 1}", measurements=[cost(**kwargs)]
        )
    return store.measurements(kind=MeasurementKind.SCHEDULE_COST)


class TestTheLedgerIsFacts:
    def test_a_measurement_records_what_was_observed_not_what_it_implies(
        self, store: AuditStore
    ) -> None:
        """A rate is derived on read, so changing the derivation does not need a time machine."""
        recorded = rows(store, [cost(value=9000, weight=30, count=15)])[0]
        assert recorded["value"] == 9000
        assert recorded["weight"] == 30
        assert recorded["count"] == 15
        assert "rate" not in recorded

    def test_a_truncated_batch_is_marked_as_a_limit_rather_than_a_measurement(
        self, store: AuditStore
    ) -> None:
        """It never reported what it would have cost; it proved the cost is at least this."""
        recorded = rows(
            store, [cost(value=16384, weight=30, count=15, outcome=False, bound=Bound.LOWER)]
        )[0]
        assert recorded["bound"] == "lower"

    def test_the_ledger_is_append_only_like_the_rest_of_the_log(
        self, store: AuditStore
    ) -> None:
        import sqlite3

        rows(store, [cost(value=9000, weight=30, count=15)])
        connection = sqlite3.connect(store.path)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE measurements SET value = 1")
        connection.close()

    def test_the_model_is_part_of_the_identity(self, store: AuditStore) -> None:
        """A cost measured under one model is not evidence about another."""
        rows(store, [cost(value=9000, weight=30, count=15, model="a")])
        rows(store, [cost(value=200, weight=30, count=15, model="b")])
        assert len(store.measurements(kind=MeasurementKind.SCHEDULE_COST, resolved_model="a")) == 1


class TestWhichObservationsCount:
    def test_a_completed_batch_counts(self) -> None:
        assert memory.usable([{"weight": 30, "count": 15, "outcome": True, "bound": None}])

    def test_a_truncated_batch_counts_because_it_bounds_the_cost(self) -> None:
        assert memory.usable([{"weight": 30, "count": 15, "outcome": False, "bound": "lower"}])

    def test_a_batch_that_failed_for_some_other_reason_says_nothing_about_cost(self) -> None:
        """It is evidence about the capability, not about the arithmetic.

        Averaging it in would drag the estimate toward whatever a failure happened to
        spend, which is the mistake `plan_batch` avoids by letting a failure set a ceiling
        without moving the rate.
        """
        assert not memory.usable([{"weight": 30, "count": 15, "outcome": False, "bound": None}])


class TestEvidenceAboutTheThingDeclaredNow:
    """A capability version and a pinned prompt both change what a capability spends.

    Left unfiltered, bumping a capability to 1.1.0 would quietly propose 1.0.0's measured
    cost as 1.1.0's declared one — the same mistake as averaging two models together, in a
    place that looks like bookkeeping rather than like evidence.
    """

    def rows(self, subject: str, prompt_version: str) -> dict[str, Any]:
        return {
            "subject": subject,
            "prompt_version": prompt_version,
            "value": 9000,
            "weight": 30,
            "count": 15,
            "outcome": True,
            "bound": None,
            "run_id": "run-1",
            "resolved_model": "gpt-5-mini",
        }

    def test_an_older_capability_version_is_set_aside(self) -> None:
        found = memory.applicable(
            [self.rows("resolve@1.0.0", "1.0.0"), self.rows("resolve@1.1.0", "1.0.0")],
            subject="resolve@1.1.0",
            prompt_version="1.0.0",
        )
        assert len(found.rows) == 1
        assert found.set_aside == {"resolve@1.0.0": 1}

    def test_an_older_prompt_is_set_aside(self) -> None:
        found = memory.applicable(
            [self.rows("resolve@1.0.0", "1.0.0"), self.rows("resolve@1.0.0", "1.1.0")],
            subject="resolve@1.0.0",
            prompt_version="1.1.0",
        )
        assert len(found.rows) == 1
        assert "prompt 1.0.0" in found.summary

    def test_what_was_set_aside_is_reported_rather_than_dropped_quietly(self) -> None:
        """A proposal built from a third of the ledger looks like one built from all of it."""
        found = memory.applicable(
            [self.rows("resolve@1.0.0", "1.0.0")] * 4,
            subject="resolve@2.0.0",
            prompt_version="1.0.0",
        )
        assert found.rows == []
        assert found.set_aside == {"resolve@1.0.0": 4}

    def test_stale_evidence_cannot_reach_a_proposal(self, store: AuditStore) -> None:
        """The whole point: twelve observations about 1.0.0 say nothing about 1.1.0."""
        recorded = spread_over_runs(store, value=9000, weight=30, count=15)
        current = memory.applicable(
            recorded, subject="resolve@1.1.0", prompt_version="1.0.0"
        )
        proposal = memory.cost_proposal(current.rows, incumbent=674)
        assert proposal.verdict is memory.Verdict.INSUFFICIENT
        assert proposal.candidate is None


class TestTheCostItSupports:
    def test_too_few_observations_proposes_nothing(self, store: AuditStore) -> None:
        proposal = memory.cost_proposal(
            rows(store, [cost(value=9000, weight=30, count=15)]), incumbent=674
        )
        assert proposal.verdict is memory.Verdict.INSUFFICIENT
        assert proposal.candidate is None

    def test_one_run_cannot_promote_itself(self, store: AuditStore) -> None:
        """One unusual corpus is not evidence about the work."""
        store.record_measurements(
            run_id="run-1",
            measurements=[cost(value=9000, weight=30, count=15) for _ in range(12)],
        )
        proposal = memory.cost_proposal(
            store.measurements(kind=MeasurementKind.SCHEDULE_COST), incumbent=674
        )
        assert proposal.verdict is memory.Verdict.INSUFFICIENT
        assert "same run" in proposal.rationale

    def test_a_settled_estimate_is_reported_as_supported(self, store: AuditStore) -> None:
        # 9000 tokens over 30 units of material, 15 items: 300 per unit, 2 units per item.
        proposal = memory.cost_proposal(
            spread_over_runs(store, value=9000, weight=30, count=15), incumbent=674
        )
        assert proposal.verdict is memory.Verdict.SUPPORTED
        assert proposal.candidate == 600
        assert proposal.runs == 3

    def test_an_estimate_matching_what_is_declared_proposes_no_edit(
        self, store: AuditStore
    ) -> None:
        proposal = memory.cost_proposal(
            spread_over_runs(store, value=9000, weight=30, count=15), incumbent=600
        )
        assert proposal.verdict is memory.Verdict.INSUFFICIENT
        assert proposal.candidate == 600

    def test_the_estimate_leans_high_because_the_two_errors_do_not_cost_the_same(
        self, store: AuditStore
    ) -> None:
        """Too small wastes a whole call, billed; too large wastes part of one.

        So the estimate sits above the typical batch rather than on it. Half these
        batches cost five times the other half: an estimate at the middle would truncate
        on every expensive one, and the run would pay to rediscover what is already known.
        """
        for index in range(12):
            store.record_measurements(
                run_id=f"run-{index % 3 + 1}",
                measurements=[
                    cost(value=3000 if index % 2 else 15000, weight=30, count=15)
                ],
            )
        recorded = store.measurements(kind=MeasurementKind.SCHEDULE_COST)
        proposal = memory.cost_proposal(recorded, incumbent=200)
        middle = memory.quantile([row["value"] / row["weight"] for row in recorded], 0.5)
        assert proposal.candidate is not None
        assert proposal.detail["rate"] > middle

    def test_a_truncation_raises_the_estimate_rather_than_being_discarded(
        self, store: AuditStore
    ) -> None:
        settled = memory.cost_proposal(
            spread_over_runs(store, value=3000, weight=30, count=15), incumbent=200
        )
        store.record_measurements(
            run_id="run-2",
            measurements=[
                cost(value=16384, weight=30, count=15, outcome=False, bound=Bound.LOWER)
                for _ in range(4)
            ],
        )
        after = memory.cost_proposal(
            store.measurements(kind=MeasurementKind.SCHEDULE_COST), incumbent=200
        )
        assert settled.candidate is not None and after.candidate is not None
        assert after.candidate > settled.candidate

    def test_a_corpus_whose_items_differ_wildly_refuses_to_pretend_one_number_fits(
        self, store: AuditStore
    ) -> None:
        """ADR 0003's open item, surfaced rather than averaged away.

        Cost is measured per capability, not per item. Where item weight is uniform that
        distinction is invisible; where it is not, one declared per-item number describes
        neither end, and saying so is worth more than a confident average.
        """
        for index in range(6):
            store.record_measurements(
                run_id=f"run-{index % 3 + 1}",
                measurements=[
                    cost(value=600, weight=2, count=2),  # one-line receipts
                    cost(value=12000, weight=40, count=2),  # itemised statements
                ],
            )
        proposal = memory.cost_proposal(
            store.measurements(kind=MeasurementKind.SCHEDULE_COST), incumbent=674
        )
        assert proposal.verdict is memory.Verdict.REVIEW
        assert "varies" in proposal.rationale
        assert proposal.detail["item_weight_spread"] > 3

    def test_mixed_models_are_refused_rather_than_averaged(self, store: AuditStore) -> None:
        for index in range(12):
            store.record_measurements(
                run_id=f"run-{index % 3 + 1}",
                measurements=[
                    cost(value=9000, weight=30, count=15, model="a" if index % 2 else "b")
                ],
            )
        proposal = memory.cost_proposal(
            store.measurements(kind=MeasurementKind.SCHEDULE_COST), incumbent=674
        )
        assert proposal.verdict is memory.Verdict.INSUFFICIENT
        assert "more than one model" in proposal.rationale

    def test_nothing_declared_yet_is_a_human_decision(self, store: AuditStore) -> None:
        proposal = memory.cost_proposal(
            spread_over_runs(store, value=9000, weight=30, count=15), incumbent=None
        )
        assert proposal.verdict is memory.Verdict.REVIEW


class TestTheFloorItSupports:
    def claims(self, store: AuditStore, pairs: list[tuple[float, bool]]) -> list[dict[str, Any]]:
        for index, (confidence, correct) in enumerate(pairs):
            store.record_measurements(
                run_id=f"run-{index % 3 + 1}",
                measurements=[
                    Measurement(
                        kind=MeasurementKind.CONFIDENCE,
                        subject="vendor",
                        resolved_model="gpt-5-mini",
                        value=confidence,
                        outcome=correct,
                    )
                ],
            )
        return store.measurements(kind=MeasurementKind.CONFIDENCE)

    def test_a_floor_is_never_supported_on_its_own(self, store: AuditStore) -> None:
        """Which error to prefer is a judgment about the work, not about the data.

        Too high quarantines files a person renames by hand; too low produces a
        confidently wrong name. Unlike a batch size, neither direction is the cheap one.
        """
        pairs = [(0.95, True)] * 10 + [(0.4, False)] * 4
        proposal = memory.floor_proposal(self.claims(store, pairs), incumbent=0.7)
        assert proposal.verdict is memory.Verdict.REVIEW
        assert proposal.candidate is not None

    def test_it_proposes_the_lowest_floor_the_evidence_supports(
        self, store: AuditStore
    ) -> None:
        """Every point above what is required is a file somebody renames by hand."""
        pairs = [(0.95, True)] * 10 + [(0.5, False)] * 5
        proposal = memory.floor_proposal(
            self.claims(store, pairs), incumbent=0.7, precision=0.99
        )
        assert proposal.candidate is not None
        assert 0.5 < proposal.candidate <= 0.95

    def test_claims_that_are_worthless_are_reported_as_worthless(
        self, store: AuditStore
    ) -> None:
        """No floor rescues a model whose confidence means nothing."""
        pairs = [(0.99, index % 2 == 0) for index in range(20)]
        proposal = memory.floor_proposal(
            self.claims(store, pairs), incumbent=0.7, precision=0.99
        )
        assert proposal.candidate is None
        assert "not worth anything" in proposal.rationale

    def test_too_few_claims_describe_an_afternoon_rather_than_the_work(
        self, store: AuditStore
    ) -> None:
        proposal = memory.floor_proposal(self.claims(store, [(0.9, True)]), incumbent=0.7)
        assert proposal.verdict is memory.Verdict.INSUFFICIENT


class TestWhetherARetryWasWorthMaking:
    def test_it_answers_from_the_log_without_new_measurement(self, store: AuditStore) -> None:
        """ADR 0003 left `max_goal_attempts` chosen rather than measured.

        Nothing new has to be recorded to settle it: whether an attempt ever recovered
        has been a fact of the audit log since before the question was asked.
        """
        found = memory.recovery({"resolved": [(1, False), (2, True), (1, True)]})
        assert found[0].by_attempt == {1: (2, 1), 2: (1, 1)}
        assert found[0].deepest_recovery == 2

    def test_attempts_beyond_the_deepest_recovery_are_reported_as_spent(self) -> None:
        found = memory.recovery({"resolved": [(1, True), (2, False), (3, False)]})
        assert found[0].deepest_recovery == 1
        assert found[0].wasted == 2

    def test_a_goal_that_never_recovered_is_named(self) -> None:
        found = memory.recovery({"reviewed": [(1, False), (2, False)]})
        assert found[0].deepest_recovery is None
        assert found[0].wasted == 2

    def test_it_reads_attempts_a_real_run_recorded(self, tmp_path: Path) -> None:
        """The query and the journal have to agree about what an attempt is."""
        from harness import build

        runtime, request, audit, _ = build(tmp_path)
        runtime.run(request, commit=False)
        attempts = audit.attempts_by_goal()
        assert attempts, "the run journalled its attempts"
        found = memory.recovery(attempts)
        assert {row.goal_id for row in found} == set(attempts)
        assert all(row.by_attempt for row in found)


class TestQuantile:
    def test_it_interpolates_rather_than_snapping_to_an_index(self) -> None:
        """A small sample should not hand the estimate to whichever value lands on it."""
        assert memory.quantile([0.0, 10.0], 0.5) == 5.0

    def test_a_single_observation_is_its_own_quantile(self) -> None:
        assert memory.quantile([7.0], 0.8) == 7.0

    def test_an_empty_sample_has_no_quantile_rather_than_a_wrong_one(self) -> None:
        assert memory.quantile([], 0.8) == 0.0


class Metered:
    """A scripted agent that reports what it spent, which a fake one does not.

    The distinction is the point of these tests: an offline run measures nothing, so it
    must contribute nothing, or the suite's arithmetic becomes evidence about a provider.
    """

    def __init__(self, inner: Any, completion_tokens: int = 4000) -> None:
        self.inner = inner
        self.completion_tokens = completion_tokens

    def organize(self, **kwargs: Any) -> Any:
        from system.model_access import ModelUsage

        organization, _ = self.inner.organize(**kwargs)
        return organization, ModelUsage(
            requested_model="openrouter/openai/gpt-5-mini",
            resolved_model="openai/gpt-5-mini-2026-04",
            completion_tokens=self.completion_tokens,
        )


class TestARunRemembersWhatItMeasured:
    def test_an_offline_run_measures_nothing_and_records_nothing(self, tmp_path: Path) -> None:
        """A fake agent reports no usage, so its batches cost zero as far as anyone knows.

        Recording those zeros would put the offline suite's arithmetic into the evidence
        for a live provider, which is worse than having no evidence at all.
        """
        from system.contracts import MeasurementKind

        from harness import build

        runtime, request, audit, _ = build(tmp_path)
        runtime.run(request, commit=False)
        assert audit.measurements(kind=MeasurementKind.SCHEDULE_COST) == []

    def test_a_run_that_spends_records_what_each_batch_cost(self, tmp_path: Path) -> None:
        from system.contracts import MeasurementKind

        from harness import build, rename_agent, seed_invoices, values_for

        source = seed_invoices(tmp_path / "probe")
        runtime, request, audit, _ = build(
            tmp_path, agents={"*": Metered(rename_agent(values_for(source)))}
        )
        runtime.run(request, commit=False)

        recorded = audit.measurements(kind=MeasurementKind.SCHEDULE_COST)
        assert recorded, "a run that spent tokens on divided work measured something"
        assert all(row["resolved_model"] == "openai/gpt-5-mini-2026-04" for row in recorded)
        # The ref, not the bare id: a capability version change invalidates the evidence.
        assert all("@" in row["subject"] for row in recorded)
        assert all(row["value"] > 0 and row["weight"] > 0 for row in recorded)

    def test_measurements_are_recorded_even_when_the_run_does_not_commit(
        self, tmp_path: Path
    ) -> None:
        """What a batch cost is true whether or not the plan was any good."""
        from system.contracts import MeasurementKind

        from harness import build, rename_agent, seed_invoices, values_for

        source = seed_invoices(tmp_path / "probe")
        runtime, request, audit, _ = build(
            tmp_path, agents={"*": Metered(rename_agent(values_for(source)))}
        )
        result = runtime.run(request, commit=False)
        assert result.outcome != "committed"
        assert audit.measurements(kind=MeasurementKind.SCHEDULE_COST)


class TestNothingReadsTheLedgerDuringARun:
    """The property that makes this memory safe rather than merely useful.

    A run whose behaviour depends on unpinned state is no longer determined by its
    journal, and `replay` would stop being a statement about what was recorded. So a
    measured constant reaches a run only by being written into the manifest that declares
    it — a versioned edit, visible in git, that a person made.
    """

    def test_a_ledger_full_of_contrary_evidence_does_not_change_what_a_run_does(
        self, tmp_path: Path
    ) -> None:
        from system.contracts import Measurement, MeasurementKind

        from harness import build, rename_agent, seed_invoices, values_for

        source = seed_invoices(tmp_path / "probe")
        runtime, request, audit, _ = build(
            tmp_path, agents={"*": Metered(rename_agent(values_for(source)))}
        )
        declared = runtime.capabilities.get("acquire").cost_per_item
        audit.record_run(
            run_id="earlier",
            workflow_id="rename_files",
            workflow_version="1.0.0",
            workflow_digest="d",
            request_digest="q",
            input_root_digest="i",
        )
        audit.record_measurements(
            run_id="earlier",
            measurements=[
                Measurement(
                    kind=MeasurementKind.SCHEDULE_COST,
                    subject="acquire@1.0.0",
                    resolved_model="openai/gpt-5-mini-2026-04",
                    value=99_000,
                    weight=1,
                    count=1,
                )
                for _ in range(40)
            ],
        )

        result = runtime.run(request, commit=False)
        scheduled = [
            invocation.parameters["cost_per_item"]
            for attempt in result.attempts
            if attempt.capability == "acquire"
            for composition, _ in attempt.outcome.scheduling
            for invocation in composition.invocations
            if "cost_per_item" in invocation.parameters
        ]
        assert scheduled, "the runtime sized this capability's work"
        # Forty observations screaming a different number, and the run still uses the one
        # a person wrote in the manifest.
        assert set(scheduled) == {declared}


def choice(
    *, subject: str, corpus: int, satisfied: bool, run_id: str = "run-1", candidates: int = 2
) -> Measurement:
    return Measurement(
        kind=MeasurementKind.SHAPE_CHOICE,
        subject=subject,
        resolved_model="gpt-5-mini",
        value=float(corpus),
        count=candidates,
        outcome=satisfied,
    )


class TestWhetherTheShapeChosenWasTheRightOne:
    """The planner now picks between capabilities from their *declared* per-item cost.

    Nothing recorded whether the pick was borne out — the same gap ADR 0003 found in the
    scheduler: arithmetic computed, shown to a model, and never checked against what
    happened.
    """

    def test_it_reports_how_often_each_shape_left_the_goal_satisfied(self) -> None:
        rows = [
            {**choice(subject="transcribe@1.0.0", corpus=60, satisfied=True).model_dump(),
             "run_id": f"run-{index}"}
            for index in range(4)
        ] + [
            {**choice(subject="transcribe@1.0.0", corpus=60, satisfied=False).model_dump(),
             "run_id": "run-9"}
        ]
        found = measurements.shapes(rows, costs={}, endings={})
        assert found[0].chosen == 5
        assert found[0].satisfied == 4
        assert found[0].rate == 0.8

    def test_it_reports_the_range_of_corpus_sizes_a_shape_was_chosen_at(self) -> None:
        """A shape that only ever won on small corpora has not been tested on large ones."""
        rows = [
            {**choice(subject="resolve@1.0.0", corpus=size, satisfied=True).model_dump(),
             "run_id": f"run-{size}"}
            for size in (3, 12, 60)
        ]
        found = measurements.shapes(rows, costs={}, endings={})
        assert found[0].corpus == (3, 60)

    def test_run_cost_is_the_median_rather_than_the_mean(self) -> None:
        """One pathological run is not the story, and a mean lets it be."""
        rows = [
            {**choice(subject="resolve@1.0.0", corpus=60, satisfied=True).model_dump(),
             "run_id": run}
            for run in ("a", "b", "c")
        ]
        found = measurements.shapes(
            rows, costs={"a": 0.08, "b": 0.09, "c": 9.00}, endings={}
        )
        assert found[0].median_run_cost == 0.09

    def test_a_run_that_never_finished_is_named_rather_than_counted_as_success(self) -> None:
        rows = [
            {**choice(subject="resolve@1.0.0", corpus=60, satisfied=True).model_dump(),
             "run_id": "a"}
        ]
        found = measurements.shapes(rows, costs={}, endings={})
        assert found[0].endings == {"unfinished": 1}

    def test_shapes_are_compared_separately_rather_than_pooled(self) -> None:
        rows = [
            {**choice(subject="resolve@1.0.0", corpus=60, satisfied=False).model_dump(),
             "run_id": "a"},
            {**choice(subject="transcribe@1.0.0", corpus=60, satisfied=True).model_dump(),
             "run_id": "b"},
        ]
        found = measurements.shapes(rows, costs={}, endings={})
        assert [shape.subject for shape in found] == ["resolve@1.0.0", "transcribe@1.0.0"]
        assert [shape.rate for shape in found] == [0.0, 1.0]


class TestOnlyARealChoiceIsRecorded:
    def test_a_goal_with_one_candidate_records_nothing(self, tmp_path: Path) -> None:
        """It was not chosen for, and a foregone conclusion is not evidence about judgment.

        The rename workflow gives most goals exactly one capability, so recording those
        would bury the handful of real decisions under a pile of settled ones.
        """
        from harness import build
        from system.contracts import MeasurementKind

        runtime, request, audit, _ = build(tmp_path)
        runtime.run(request, commit=False)
        assert audit.measurements(kind=MeasurementKind.SHAPE_CHOICE) == []

    def test_a_goal_with_several_candidates_records_the_choice(self, tmp_path: Path) -> None:
        from harness import build
        from system.contracts import MeasurementKind
        from system.runtime.control import Chosen

        runtime, request, audit, _ = build(tmp_path)
        original = runtime.__dict__.get("_probe")
        assert original is None  # nothing already patched

        # The rename workflow declares one capability per goal, so a second candidate has
        # to be simulated to exercise the recording path end to end.
        from system.runtime import control

        real = control.Controller._choose_capability

        def two_candidates(self: Any, goal: Any) -> Chosen:
            chosen = real(self, goal)
            return Chosen(
                capability=chosen.capability,
                impediment=chosen.impediment,
                candidates=2,
                corpus=chosen.corpus if chosen.corpus is not None else 3,
            )

        control.Controller._choose_capability = two_candidates  # type: ignore[method-assign]
        try:
            runtime.run(request, commit=False)
        finally:
            control.Controller._choose_capability = real  # type: ignore[method-assign]

        recorded = audit.measurements(kind=MeasurementKind.SHAPE_CHOICE)
        assert recorded, "a goal with a real choice was recorded"
        assert all(row["count"] == 2 for row in recorded)
        assert all(row["value"] > 0 for row in recorded)
        assert all("@" in row["subject"] for row in recorded)
