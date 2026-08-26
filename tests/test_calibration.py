"""Measuring what a reported confidence has been worth.

ADR 0001 recorded it and ADR 0003 repeated it: the floor is enforced and nothing has ever
checked whether a model's 0.8 means anything. A floor chosen rather than measured is a
guess with a number on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shakespeare.calibration import (
    accuracy_above,
    brier,
    buckets,
    expected_error,
    floor_for,
    observe,
    report,
)


def claims(*pairs: tuple[float, bool]) -> list[tuple[float, bool]]:
    return list(pairs)


class TestBands:
    def test_claims_are_grouped_by_the_confidence_reported(self) -> None:
        bands = buckets(claims((0.05, True), (0.95, True), (0.92, False)))
        assert [band.count for band in bands] == [1, 2]

    def test_an_unused_band_is_not_reported(self) -> None:
        """A band nobody claimed says nothing, and printing it invites reading noise."""
        assert all(band.count for band in buckets(claims((0.9, True))))

    def test_the_gap_is_positive_when_claims_run_ahead_of_reality(self) -> None:
        overconfident = buckets(claims((0.9, False), (0.9, False), (0.9, True)))
        assert overconfident[0].gap > 0

    def test_the_gap_is_negative_when_claims_are_too_modest(self) -> None:
        modest = buckets(claims((0.5, True), (0.5, True), (0.5, True)))
        assert modest[0].gap < 0


class TestScores:
    def test_a_perfect_forecaster_scores_zero(self) -> None:
        assert brier(claims((1.0, True), (0.0, False))) == 0.0

    def test_a_confidently_wrong_forecaster_scores_worst(self) -> None:
        assert brier(claims((1.0, False))) == 1.0

    def test_a_coin_flip_claimed_as_certainty_beats_nothing(self) -> None:
        honest = brier(claims((0.5, True), (0.5, False)))
        reckless = brier(claims((1.0, True), (1.0, False)))
        assert honest < reckless

    def test_mean_error_ignores_claims_nobody_made(self) -> None:
        assert expected_error(claims((0.9, True))) == pytest.approx(0.1)


class TestTheFloorIsDerivedFromEvidence:
    def test_it_finds_the_lowest_floor_that_reaches_the_target(self) -> None:
        # Everything at 0.8 and above is right; below it, half are wrong.
        evidence = claims(*[(0.9, True)] * 10, *[(0.8, True)] * 10, *[(0.5, False)] * 10)
        assert floor_for(evidence, 1.0) == 0.55

    def test_lowest_rather_than_safest(self) -> None:
        """Every point of floor above what the evidence needs is a file renamed by hand."""
        evidence = claims(*[(0.6, True)] * 20)
        assert floor_for(evidence, 0.95) == 0.0

    def test_no_floor_is_offered_when_claims_do_not_separate_right_from_wrong(self) -> None:
        evidence = claims(*[(0.9, True), (0.9, False)] * 10)
        assert floor_for(evidence, 0.99) is None

    def test_a_floor_reports_what_it_would_accept(self) -> None:
        evidence = claims((0.9, True), (0.5, False), (0.4, True))
        count, accuracy = accuracy_above(evidence, 0.8)
        assert (count, accuracy) == (1, 1.0)


class TestPairingAgainstTruth:
    def test_a_right_value_and_a_wrong_one_are_told_apart(self) -> None:
        per_field = observe(
            [
                {
                    "relpath": "a.pdf",
                    "values": {"vendor": "ACME", "invoice_number": "INV-1"},
                    "confidences": {"vendor": 0.9, "invoice_number": 0.6},
                }
            ],
            {"a.pdf": {"vendor": "ACME", "invoice_number": "INV-2"}},
        )
        assert per_field["vendor"] == [(0.9, True)]
        assert per_field["invoice_number"] == [(0.6, False)]

    def test_surrounding_whitespace_is_not_a_wrong_answer(self) -> None:
        per_field = observe(
            [{"relpath": "a.pdf", "values": {"v": " ACME "}, "confidences": {"v": 0.9}}],
            {"a.pdf": {"v": "ACME"}},
        )
        assert per_field["v"] == [(0.9, True)]

    def test_a_field_the_run_declined_to_report_is_not_a_wrong_claim(self) -> None:
        """Omission is the safe failure working, not an error to be scored."""
        per_field = observe(
            [{"relpath": "a.pdf", "values": {}, "confidences": {}}],
            {"a.pdf": {"vendor": "ACME"}},
        )
        assert per_field == {}

    def test_a_value_with_no_confidence_is_not_counted(self) -> None:
        """The floor is about claims; a value offered without one is a different question."""
        per_field = observe(
            [{"relpath": "a.pdf", "values": {"v": "x"}, "confidences": {}}],
            {"a.pdf": {"v": "x"}},
        )
        assert per_field == {}

    def test_an_item_missing_from_the_truth_file_is_skipped(self) -> None:
        per_field = observe(
            [{"relpath": "unknown.pdf", "values": {"v": "x"}, "confidences": {"v": 0.9}}],
            {"a.pdf": {"v": "x"}},
        )
        assert per_field == {}


class TestReport:
    def test_it_reports_per_field_and_overall(self) -> None:
        summary = report(
            {"vendor": claims((0.9, True), (0.9, True)), "po": claims((0.4, False))}
        )
        assert summary.observations == 3
        assert summary.fields["vendor"] == (2, 1.0)
        assert summary.fields["po"] == (1, 0.0)

    def test_overconfidence_is_named_because_it_is_the_direction_that_hurts(self) -> None:
        assert report({"v": claims(*[(0.95, False)] * 10)}).overconfident
        assert not report({"v": claims(*[(0.55, True)] * 10)}).overconfident


class TestTheRenderRecordsWhatWasClaimed:
    def test_values_and_confidences_survive_rendering(self) -> None:
        runner = _render_template()
        out = runner(
            {
                "template": "{vendor}",
                "fields": [{"name": "vendor"}],
                "items": [
                    {
                        "item_id": "i1",
                        "directory": "",
                        "extension": ".pdf",
                        "values": {"vendor": "ACME"},
                        "confidences": {"vendor": 0.83},
                    }
                ],
            },
            Path("."),
        )
        assert out["results"][0]["values"] == {"vendor": "ACME"}
        assert out["results"][0]["confidences"] == {"vendor": 0.83}

    def test_a_quarantined_item_still_records_what_it_claimed(self) -> None:
        """The near-misses are exactly the evidence a floor should be chosen from."""
        runner = _render_template()
        out = runner(
            {
                "template": "{vendor}",
                "fields": [{"name": "vendor", "confidence_floor": 0.9}],
                "items": [
                    {
                        "item_id": "i1",
                        "directory": "",
                        "extension": ".pdf",
                        "values": {"vendor": "ACME"},
                        "confidences": {"vendor": 0.5},
                    }
                ],
            },
            Path("."),
        )
        assert out["unrendered"]
        assert out["results"][0]["confidences"] == {"vendor": 0.5}


def _render_template():
    from shakespeare.runners import pure_transform

    def call(arguments, workspace):
        return pure_transform({**arguments, "operation": "render_template"}, workspace)

    return call
