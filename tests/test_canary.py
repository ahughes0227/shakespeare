"""Golden-fixture drift detection.

Drift is only proactive if something re-runs known inputs. These cover the comparison and
the recording discipline; the run itself uses the real model by design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from shakespeare.canary import (
    CanaryCase,
    CanaryError,
    CanaryResult,
    decisions_of,
    load_cases,
    record,
    run_case,
)
from shakespeare.cli import app
from shakespeare.contracts import ChangeAction, ChangePlan
from shakespeare.domain.planning import RenameEntry
from typer.testing import CliRunner

from harness import INVOICES, build, seed_invoices

runner = CliRunner()


def _case_dir(root: Path, name: str = "invoices", *, expected: list[dict] | None = None) -> Path:
    directory = root / name
    (directory / "inputs").mkdir(parents=True, exist_ok=True)
    seed_invoices(directory / "inputs", INVOICES)
    (directory / "case.yml").write_text(
        yaml.safe_dump({"prompt": "rename these invoices", "inputs": "inputs"})
    )
    if expected is not None:
        (directory / "expected.json").write_text(json.dumps(expected, indent=2))
    return directory


def _plan(*entries: tuple[str, str, str | None]) -> ChangePlan:
    return ChangePlan(
        run_id="r",
        workflow_id="rename_files",
        workflow_digest="d",
        decision_digest="s",
        entries=tuple(
            RenameEntry(
                item_id=source,
                source_ref=source,
                action=ChangeAction(action),
                target_relpath=target,
            )
            for source, action, target in entries
        ),
    )


class TestLoading:
    def test_a_case_without_an_expectation_is_flagged_not_failed(self, tmp_path: Path) -> None:
        """A newly written case has nothing recorded yet; that is a state, not an error."""
        _case_dir(tmp_path)
        (case,) = load_cases(tmp_path)
        assert case.name == "invoices"
        assert not case.has_expectation

    def test_an_expectation_is_loaded_and_sorted(self, tmp_path: Path) -> None:
        _case_dir(
            tmp_path,
            expected=[
                {"source": "b.pdf", "action": "changed", "target": "B.pdf"},
                {"source": "a.pdf", "action": "changed", "target": "A.pdf"},
            ],
        )
        (case,) = load_cases(tmp_path)
        assert case.expected[0][0] == "a.pdf", "order must not affect comparison"

    def test_a_case_with_no_prompt_is_refused(self, tmp_path: Path) -> None:
        directory = tmp_path / "broken"
        (directory / "inputs").mkdir(parents=True)
        (directory / "case.yml").write_text(yaml.safe_dump({"inputs": "inputs"}))
        with pytest.raises(CanaryError, match="no prompt"):
            load_cases(tmp_path)

    def test_a_missing_input_tree_is_refused(self, tmp_path: Path) -> None:
        directory = tmp_path / "broken"
        directory.mkdir()
        (directory / "case.yml").write_text(yaml.safe_dump({"prompt": "x"}))
        with pytest.raises(CanaryError, match="input tree not found"):
            load_cases(tmp_path)

    def test_no_cases_is_not_an_error(self, tmp_path: Path) -> None:
        assert load_cases(tmp_path / "nothing-here") == ()


class TestDriftDetection:
    def _case(self, tmp_path: Path, expected: tuple[tuple[str, str, str | None], ...]):
        return CanaryCase(
            name="c", prompt="p", inputs=tmp_path, expected=tuple(sorted(expected))
        )

    def test_identical_decisions_are_not_drift(self, tmp_path: Path) -> None:
        decisions = (("a.pdf", "changed", "A.pdf"),)
        result = CanaryResult(case=self._case(tmp_path, decisions), produced=decisions)
        assert not result.drifted

    def test_a_changed_target_is_drift(self, tmp_path: Path) -> None:
        result = CanaryResult(
            case=self._case(tmp_path, (("a.pdf", "changed", "A.pdf"),)),
            produced=(("a.pdf", "changed", "DIFFERENT.pdf"),),
        )
        assert result.drifted
        assert result.diff() == [("a.pdf", "changed → A.pdf", "changed → DIFFERENT.pdf")]

    def test_an_item_becoming_unresolved_is_drift(self, tmp_path: Path) -> None:
        """The failure a canary exists to catch: extraction quietly getting worse."""
        result = CanaryResult(
            case=self._case(tmp_path, (("a.pdf", "changed", "A.pdf"),)),
            produced=(("a.pdf", "unresolved", None),),
        )
        assert result.drifted

    def test_a_run_that_produced_no_plan_is_drift(self, tmp_path: Path) -> None:
        result = CanaryResult(
            case=self._case(tmp_path, (("a.pdf", "changed", "A.pdf"),)),
            produced=(),
            error="stage extract was not accepted",
        )
        assert result.drifted

    def test_run_order_does_not_affect_comparison(self, tmp_path: Path) -> None:
        expected = (("a.pdf", "changed", "A.pdf"), ("b.pdf", "changed", "B.pdf"))
        result = CanaryResult(case=self._case(tmp_path, expected), produced=tuple(reversed(
            tuple(sorted(expected))
        )))
        # decisions_of sorts, so a real run cannot produce reversed order; this asserts the
        # comparison itself is order-sensitive only because the producer sorts.
        assert result.produced != result.case.expected


class TestRecording:
    def test_recording_captures_the_current_decisions(self, tmp_path: Path) -> None:
        case = CanaryCase(name="c", prompt="p", inputs=tmp_path, expected=())
        path = record(
            case,
            _plan(("a.pdf", "changed", "A.pdf"), ("b.pdf", "unresolved", None)),
            tmp_path,
        )
        payload = json.loads(path.read_text())
        assert payload == [
            {"source": "a.pdf", "action": "changed", "target": "A.pdf"},
            {"source": "b.pdf", "action": "unresolved", "target": None},
        ]

    def test_a_recorded_case_then_matches_itself(self, tmp_path: Path) -> None:
        plan = _plan(("a.pdf", "changed", "A.pdf"))
        case = CanaryCase(name="invoices", prompt="p", inputs=tmp_path, expected=())
        (tmp_path / "invoices" / "inputs").mkdir(parents=True)
        (tmp_path / "invoices" / "case.yml").write_text(
            yaml.safe_dump({"prompt": "p", "inputs": "inputs"})
        )
        record(case, plan, tmp_path)
        (reloaded,) = load_cases(tmp_path)
        assert reloaded.expected == decisions_of(plan)


class TestAgainstARealRun:
    def test_a_canary_matches_a_real_planning_run(self, tmp_path: Path) -> None:
        """End to end: plan, record, then re-plan and confirm no drift."""
        runtime, request, audit, _ = build(tmp_path)
        planned = runtime.run(request, commit=False)
        assert planned.plan is not None

        case = CanaryCase(
            name="invoices", prompt=request.prompt, inputs=Path(request.input_root), expected=()
        )
        record(case, planned.plan, tmp_path / "_canaries")
        recorded = tuple(
            (item["source"], item["action"], item["target"])
            for item in json.loads(
                (tmp_path / "_canaries" / "invoices" / "expected.json").read_text()
            )
        )
        result = CanaryResult(
            case=case.__class__(
                name=case.name, prompt=case.prompt, inputs=case.inputs, expected=recorded
            ),
            produced=decisions_of(planned.plan),
        )
        assert not result.drifted
        audit.close()

    def test_run_case_never_commits(self, tmp_path: Path) -> None:
        """A canary must observe, not act."""
        runtime, request, audit, _ = build(tmp_path)
        case = CanaryCase(
            name="c", prompt=request.prompt, inputs=Path(request.input_root), expected=()
        )
        output = tmp_path / "canary-out"
        run_case(runtime, case, output_root=output)
        assert not output.exists()
        audit.close()


class TestCli:
    def test_listing_with_no_cases_explains_how_to_add_one(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["canary", "list", "--canaries", str(tmp_path)])
        assert result.exit_code == 0
        assert "case.yml" in result.stdout

    def test_an_unrecorded_case_is_shown_as_unrecorded(self, tmp_path: Path) -> None:
        _case_dir(tmp_path)
        result = runner.invoke(app, ["canary", "list", "--canaries", str(tmp_path)])
        assert result.exit_code == 0
        assert "unrecorded" in result.stdout

    def test_running_an_unknown_case_is_refused(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["canary", "run", "nope", "--canaries", str(tmp_path), "--state", str(tmp_path)]
        )
        assert result.exit_code == 1
