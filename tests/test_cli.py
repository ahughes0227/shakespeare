"""The command surface.

These exercise the commands a person actually types, including the paths that must refuse
to do something.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from system.cli import app
from system.components.admission import AdmissionService
from system.components.catalog import build_registry
from system.contracts import OperatorFamily, OperatorRequest, RequestKind
from system.prompt_store import PromptStore
from system.runtime.audit import AuditStore
from typer.testing import CliRunner

runner = CliRunner()


def invoke(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.stdout


@pytest.fixture
def state(tmp_path: Path) -> Path:
    return tmp_path / "state"


def _seed_request(
    state: Path,
    *,
    kind: RequestKind = RequestKind.VARIANT,
    dependencies: tuple[str, ...] = ("requests",),
    name: str = "text.tidy",
) -> str:
    """Record a pending request the way a run would."""
    from test_admission import StubRenderer, passing_tests

    audit = AuditStore(state / "audit.sqlite3")
    service = AdmissionService(
        registry=build_registry(),
        audit=audit,
        workspace=state / "candidates",
        renderer=StubRenderer(),
        test_runner=passing_tests,
    )
    report, _ = service.evaluate(
        OperatorRequest(
            request_id="req-1",
            run_id="run-1",
            domain_id="field_resolution",
            kind=kind,
            family=OperatorFamily.PURE_TRANSFORM,
            name=name,
            features=frozenset({"normalize"}),
            dependencies=dependencies,
            rationale="normalise vendor names before rendering",
        )
    )
    audit.close()
    return report.report_id


class TestInspection:
    def test_workflows_validate_shows_the_goal_graph(self) -> None:
        code, output = invoke("workflows", "validate")
        assert code == 0
        assert "rename_files" in output
        assert "inventoried" in output and "ReviewEvidence" in output

    def test_operators_marks_mutation_operators_runtime_only(self) -> None:
        code, output = invoke("operators")
        assert code == 0
        assert "runtime only" in output
        assert "fs.commit" in output

    def test_capabilities_shows_what_each_produces(self) -> None:
        code, output = invoke("capabilities")
        assert code == 0
        assert "survey" in output and "FileInventory" in output

    def test_prompts_list_shows_what_each_capability_pins(self) -> None:
        code, output = invoke("prompts", "list")
        assert code == 0
        assert "convene" in output
        assert "planner.judge_gate" in output


class TestRequests:
    def test_empty_queue_says_so(self, state: Path) -> None:
        code, output = invoke("requests", "list", "--state", str(state))
        assert code == 0
        assert "No operator requests are waiting" in output

    def test_a_pending_request_is_listed_with_its_computed_risk(self, state: Path) -> None:
        _seed_request(state)
        code, output = invoke("requests", "list", "--state", str(state))
        assert code == 0
        assert "text.tidy" in output
        assert "medium" in output, "a dependency-bearing request is medium risk"

    def test_review_shows_the_findings(self, state: Path) -> None:
        report_id = _seed_request(state)
        code, output = invoke("requests", "review", report_id[:12], "--state", str(state))
        assert code == 0
        assert "untrusted_dependency" in output

    def test_approving_removes_it_from_the_queue(self, state: Path) -> None:
        report_id = _seed_request(state)
        code, _ = invoke("requests", "approve", report_id[:12], "--state", str(state))
        assert code == 0
        _, output = invoke("requests", "list", "--state", str(state))
        assert "No operator requests are waiting" in output

    def test_denying_also_closes_it(self, state: Path) -> None:
        report_id = _seed_request(state)
        code, output = invoke("requests", "deny", report_id[:12], "--state", str(state))
        assert code == 0
        assert "Denied" in output

    def test_a_behaviour_request_cannot_be_approved(self, state: Path) -> None:
        """No approval can conjure a runner operation that does not exist."""
        report_id = _seed_request(state, kind=RequestKind.BEHAVIOUR, dependencies=())
        result = runner.invoke(
            app, ["requests", "approve", report_id[:12], "--state", str(state)]
        )
        assert result.exit_code == 1
        assert "runners.py" in result.stdout + str(result.stderr)

    def test_an_unknown_report_is_refused(self, state: Path) -> None:
        result = runner.invoke(app, ["requests", "review", "deadbeef", "--state", str(state)])
        assert result.exit_code == 1


class TestPromptPromotion:
    def _artifacts(self, tmp_path: Path) -> Path:
        """Write into a temporary root: a test must never mutate the repo's prompts."""
        from system.contracts import PromptArtifact

        root = tmp_path / "prompts"
        store = PromptStore(root)
        for version in ("1.0.0", "1.1.0"):
            store.save(
                PromptArtifact(
                    signature_id="planner.route",
                    version=version,
                    instructions="select a workflow",
                )
            )
        return root

    def test_a_clear_win_auto_promotes(self, state: Path, tmp_path: Path) -> None:
        root = self._artifacts(tmp_path)
        code, output = invoke(
            "prompts", "promote", "planner.route",
            "--candidate", "1.1.0", "--score", "0.95",
            "--incumbent", "1.0.0", "--incumbent-score", "0.80",
            "--state", str(state), "--prompts", str(root),
        )
        assert code == 0
        assert "auto_promote" in output
        assert "prompt_version" in output, "it should say how to pin the winner"

    def test_a_golden_regression_is_rejected_however_high_the_score(
        self, state: Path, tmp_path: Path
    ) -> None:
        root = self._artifacts(tmp_path)
        code, output = invoke(
            "prompts", "promote", "planner.route",
            "--candidate", "1.1.0", "--score", "0.99",
            "--incumbent", "1.0.0", "--incumbent-score", "0.80",
            "--regressed", "invoice_with_no_po",
            "--state", str(state), "--prompts", str(root),
        )
        assert code == 0
        assert "reject" in output

    def test_a_marginal_win_goes_to_a_human(self, state: Path, tmp_path: Path) -> None:
        root = self._artifacts(tmp_path)
        code, output = invoke(
            "prompts", "promote", "planner.route",
            "--candidate", "1.1.0", "--score", "0.805",
            "--incumbent", "1.0.0", "--incumbent-score", "0.80",
            "--state", str(state), "--prompts", str(root),
        )
        assert code == 0
        assert "human_review" in output


class TestHygiene:
    def test_the_suite_does_not_mutate_the_repository(self) -> None:
        """A test that writes into a system package leaves the repo dirty.

        This caught exactly that: the promotion tests were writing a real artifact into
        the repository's prompt tree.
        """
        import subprocess

        result = subprocess.run(
            ["git", "status", "--porcelain", "shakespeare", "conventions", "docs"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        untracked = [
            line for line in result.stdout.splitlines() if line.startswith("??")
        ]
        assert not untracked, f"tests left files behind: {untracked}"


class TestGuardRails:
    def test_new_operator_refuses_an_unvetted_operation(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "new-operator", "--family", "pure_transform", "--name", "evil.exec",
                "--operation", "run_shell", "--to", str(tmp_path / "ops"),
            ],
        )
        assert result.exit_code == 1
        assert "runners.py" in result.stdout + str(result.stderr)

    def test_planning_without_a_model_explains_how_to_set_one(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.delenv("SHAKESPEARE_MODEL", raising=False)
        source = tmp_path / "in"
        source.mkdir()
        (source / "a.pdf").write_bytes(b"x")
        result = runner.invoke(
            app,
            ["plan", "-p", "rename", "-i", str(source), "-o", str(tmp_path / "out"),
             "--state", str(tmp_path / "state")],
        )
        assert result.exit_code == 1
        assert "SHAKESPEARE_MODEL" in result.stdout + str(result.stderr)

    def test_replaying_an_unknown_run_is_refused(self, tmp_path: Path) -> None:
        source = tmp_path / "in"
        source.mkdir()
        result = runner.invoke(
            app,
            ["replay", "nope", "-i", str(source), "--state", str(tmp_path / "state")],
        )
        assert result.exit_code == 1
