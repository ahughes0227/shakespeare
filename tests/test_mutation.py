"""Two-phase commit: staging, verification, atomic commit, reversal."""

from __future__ import annotations

import pathlib

import pytest
from system.contracts import ChangeAction, ChangePlan
from system.domain.mutation import (
    MutationError,
    commit,
    discard,
    reverse,
    stage_plan,
    verify_tree,
)
from system.domain.planning import RenameEntry


def _tree(root: pathlib.Path) -> dict[str, str]:
    (root / "2024" / "q1").mkdir(parents=True)
    (root / "2024" / "q1" / "scan001.pdf").write_bytes(b"invoice one")
    (root / "2024" / "scan002.pdf").write_bytes(b"invoice two")
    (root / "loose.pdf").write_bytes(b"unreadable")
    import hashlib

    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _plan(digests: dict[str, str]) -> ChangePlan:
    return ChangePlan(
        run_id="r",
        workflow_id="rename_files",
        workflow_digest="wd",
        decision_digest="dd",
        entries=(
            RenameEntry(
                item_id="1",
                source_ref="2024/q1/scan001.pdf",
                action=ChangeAction.CHANGED,
                target_relpath="2024/q1/202401, ACME, INV-1.pdf",
                digests={"source": digests["2024/q1/scan001.pdf"]},
            ),
            RenameEntry(
                item_id="2",
                source_ref="2024/scan002.pdf",
                action=ChangeAction.CHANGED,
                target_relpath="2024/202402, ACME, INV-2.pdf",
                digests={"source": digests["2024/scan002.pdf"]},
            ),
            RenameEntry(
                item_id="3",
                source_ref="loose.pdf",
                action=ChangeAction.UNRESOLVED,
                reason="ocr_unavailable",
                digests={"source": digests["loose.pdf"]},
            ),
        ),
    )


@pytest.fixture
def staged(tmp_path: pathlib.Path):
    source = tmp_path / "in"
    source.mkdir()
    digests = _tree(source)
    plan = _plan(digests)
    staging = tmp_path / "staging"
    reversals = stage_plan(plan=plan, input_root=source, staging_root=staging)
    return source, staging, plan, reversals


class TestStaging:
    def test_mirrors_input_structure(self, staged) -> None:
        _, staging, _, _ = staged
        produced = sorted(
            path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()
        )
        assert produced == [
            "2024/202402, ACME, INV-2.pdf",
            "2024/q1/202401, ACME, INV-1.pdf",
            "_unresolved/loose.pdf",
        ]

    def test_source_tree_is_untouched(self, staged) -> None:
        source, _, _, _ = staged
        assert sorted(
            path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()
        ) == ["2024/q1/scan001.pdf", "2024/scan002.pdf", "loose.pdf"]

    def test_quarantined_item_keeps_its_original_name(self, staged) -> None:
        _, staging, _, _ = staged
        assert (staging / "_unresolved" / "loose.pdf").read_bytes() == b"unreadable"

    def test_every_write_records_a_reversal(self, staged) -> None:
        _, _, plan, reversals = staged
        assert len(reversals) == len(plan.entries)
        assert all(item.operation == "stage_write" for item in reversals)


class TestPathEscape:
    @pytest.mark.parametrize("target", ["../escape.pdf", "/etc/passwd", "a/../../escape.pdf"])
    def test_plan_cannot_write_outside_staging(
        self, tmp_path: pathlib.Path, target: str
    ) -> None:
        source = tmp_path / "in"
        source.mkdir()
        (source / "x.pdf").write_bytes(b"x")
        plan = ChangePlan(
            run_id="r",
            workflow_id="w",
            workflow_digest="wd",
            decision_digest="dd",
            entries=(
                RenameEntry(
                    item_id="1",
                    source_ref="x.pdf",
                    action=ChangeAction.CHANGED,
                    target_relpath=target,
                ),
            ),
        )
        with pytest.raises(MutationError, match="escapes its root|absolute path"):
            stage_plan(plan=plan, input_root=source, staging_root=tmp_path / "staging")


class TestVerification:
    def test_passes_for_a_faithful_staging_tree(self, staged) -> None:
        _, staging, plan, _ = staged
        assert verify_tree(plan=plan, staging_root=staging)["ok"] is True

    def test_detects_corrupted_content(self, staged) -> None:
        _, staging, plan, _ = staged
        (staging / "2024" / "202402, ACME, INV-2.pdf").write_bytes(b"tampered")
        report = verify_tree(plan=plan, staging_root=staging)
        assert report["ok"] is False
        assert report["mismatched"] == ["2"]

    def test_detects_a_missing_file(self, staged) -> None:
        _, staging, plan, _ = staged
        (staging / "_unresolved" / "loose.pdf").unlink()
        report = verify_tree(plan=plan, staging_root=staging)
        assert report["ok"] is False
        assert report["missing"] == ["3"]

    def test_detects_an_extra_file(self, staged) -> None:
        _, staging, plan, _ = staged
        (staging / "stowaway.pdf").write_bytes(b"x")
        assert verify_tree(plan=plan, staging_root=staging)["ok"] is False


class TestCommit:
    def test_commit_moves_staging_into_place(self, staged, tmp_path: pathlib.Path) -> None:
        _, staging, _, _ = staged
        output = tmp_path / "out"
        commit(staging_root=staging, output_root=output)
        assert (output / "2024" / "q1" / "202401, ACME, INV-1.pdf").is_file()
        assert not staging.exists()

    def test_refuses_to_overwrite_an_existing_output_root(
        self, staged, tmp_path: pathlib.Path
    ) -> None:
        _, staging, _, _ = staged
        output = tmp_path / "out"
        output.mkdir()
        with pytest.raises(MutationError, match="already exists"):
            commit(staging_root=staging, output_root=output)

    def test_discard_leaves_nothing_user_visible(self, staged, tmp_path: pathlib.Path) -> None:
        """A failed review must cost the user nothing."""
        _, staging, _, _ = staged
        discard(staging)
        assert not staging.exists()
        assert not (tmp_path / "out").exists()

    def test_reverse_removes_a_committed_output(self, staged, tmp_path: pathlib.Path) -> None:
        _, staging, _, _ = staged
        output = tmp_path / "out"
        record = commit(staging_root=staging, output_root=output)
        reverse(record)
        assert not output.exists()

    def test_unknown_operation_has_no_reversal(self) -> None:
        from system.contracts import ReversalRecord

        with pytest.raises(MutationError, match="no reversal is defined"):
            reverse(ReversalRecord(mutation_id="m", operation="invented"))


class TestUnreadableEntries:
    """An unreadable file is reported in the plan but cannot be copied anywhere."""

    def _plan(self) -> ChangePlan:
        return ChangePlan(
            run_id="r",
            workflow_id="w",
            workflow_digest="wd",
            decision_digest="dd",
            entries=(
                RenameEntry(
                    item_id="locked.pdf",
                    source_ref="locked.pdf",
                    action=ChangeAction.UNRESOLVED,
                    reason="unreadable:PermissionError",
                    digests={"unreadable": "true"},
                ),
            ),
        )

    def test_staging_does_not_try_to_copy_it(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "in"
        source.mkdir()
        staging = tmp_path / "staging"
        reversals = stage_plan(plan=self._plan(), input_root=source, staging_root=staging)
        assert reversals == ()

    def test_verification_does_not_expect_it_on_disk(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "in"
        source.mkdir()
        staging = tmp_path / "staging"
        plan = self._plan()
        stage_plan(plan=plan, input_root=source, staging_root=staging)
        report = verify_tree(plan=plan, staging_root=staging)
        assert report["ok"] is True
        assert report["unreadable"] == 1
