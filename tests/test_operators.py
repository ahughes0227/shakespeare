from __future__ import annotations

import pathlib

import pytest
from shakespeare.contracts import ChangeAction, OperatorFamily
from shakespeare.domain.extraction import Backend, extract
from shakespeare.domain.filesystem import scan
from shakespeare.domain.planning import (
    AssemblyError,
    PlannedName,
    ScannedItem,
    assemble_plan,
    run_check,
)
from shakespeare.operators.builtin import BUILTIN, RUNTIME_ONLY, build_registry
from shakespeare.runners import RunnerError, allowlist, pure_transform


class TestFamilyDiscipline:
    def test_every_builtin_operation_is_vetted(self) -> None:
        for name, (spec, operation) in BUILTIN.items():
            assert operation in allowlist(spec.family), f"{name} names an unvetted operation"

    def test_every_vetted_operation_belongs_to_exactly_one_family(self) -> None:
        seen: dict[str, OperatorFamily] = {}
        for family in OperatorFamily:
            for operation in allowlist(family):
                assert operation not in seen, (
                    f"{operation} is reachable from both {seen.get(operation)} and {family}"
                )
                seen[operation] = family

    def test_unknown_operation_is_refused(self) -> None:
        with pytest.raises(RunnerError, match="unsupported operation"):
            pure_transform({"operation": "arbitrary_code"}, pathlib.Path("."))

    def test_builtins_register_cleanly(self) -> None:
        assert len(build_registry().names()) == len(BUILTIN)


class TestWriteContainment:
    def test_only_the_mutation_family_declares_writes(self) -> None:
        for spec, _ in BUILTIN.values():
            writes = [item for item in spec.side_effects if item.startswith("write")]
            if writes:
                assert spec.family is OperatorFamily.FILESYSTEM_MUTATION, spec.name

    def test_every_mutation_operator_is_runtime_only(self) -> None:
        """Agents plan; the runtime commits.

        If a mutation operator ever became reachable from a domain catalog, an agent could
        write to disk before Review had a chance to reject the plan.
        """
        mutating = {
            spec.name
            for spec, _ in BUILTIN.values()
            if spec.family is OperatorFamily.FILESYSTEM_MUTATION
        }
        assert mutating == RUNTIME_ONLY


class TestScan:
    def test_orders_deterministically_and_skips_hidden(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "2.pdf").write_bytes(b"two")
        (tmp_path / "a.pdf").write_bytes(b"one")
        (tmp_path / ".secret").write_text("x")
        items, _ = scan(tmp_path)
        assert [item.relpath for item in items] == ["a.pdf", "b/2.pdf"]
        assert scan(tmp_path)[0] == items

    def test_symlinks_are_reported_not_followed(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "real.pdf").write_bytes(b"x")
        (tmp_path / "link.pdf").symlink_to(tmp_path / "real.pdf")
        items, skipped = scan(tmp_path)
        assert [item.relpath for item in items] == ["real.pdf"]
        assert skipped == ({"relpath": "link.pdf", "reason": "symlink"},)

    def test_identical_files_are_two_things_to_rename(self, tmp_path: pathlib.Path) -> None:
        """Identity is the file, not its contents.

        A pure content address made duplicates one item, and a live run on five
        byte-identical documents built a plan with one entry for five files and failed
        its own balance check. Duplicate scans are ordinary, and both copies need naming.
        """
        (tmp_path / "a.pdf").write_bytes(b"same")
        (tmp_path / "b.pdf").write_bytes(b"same")
        items, _ = scan(tmp_path)
        assert len({item.item_id for item in items}) == 2

    def test_duplicate_content_is_still_visible_as_duplicate(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Nothing is lost: the content digest is recorded beside the identity."""
        (tmp_path / "a.pdf").write_bytes(b"same")
        (tmp_path / "b.pdf").write_bytes(b"same")
        items, _ = scan(tmp_path)
        assert len({item.sha256 for item in items}) == 1

    def test_the_same_file_keeps_its_identity_across_scans(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Identity has to be stable, or a resumed run cannot tell what it already did."""
        (tmp_path / "a.pdf").write_bytes(b"same")
        first, _ = scan(tmp_path)
        second, _ = scan(tmp_path)
        assert first[0].item_id == second[0].item_id


class TestExtraction:
    def test_unsupported_media_type_gives_a_reason_not_empty_text(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "x.zip"
        path.write_bytes(b"PK")
        result = extract(item_id="1", path=path, media_type="application/zip")
        assert not result.usable
        assert result.unavailable_reason == "unsupported_media_type:application/zip"

    def test_missing_ocr_degrades_explicitly(self, tmp_path: pathlib.Path) -> None:
        """A missing tesseract must quarantine the item, never yield a nameless file."""
        path = tmp_path / "scan.png"
        path.write_bytes(b"not-a-real-png")
        result = extract(item_id="1", path=path, media_type="image/png", backend=Backend.IMAGE_OCR)
        assert not result.usable
        assert result.unavailable_reason is not None
        assert result.unavailable_reason.startswith(
            ("ocr_unavailable", "backend_unavailable", "unreadable")
        )


class TestPlanAssembly:
    def _scanned(self) -> tuple[ScannedItem, ...]:
        return (
            ScannedItem(item_id="1", relpath="a/x.pdf", sha256="d" * 64),
            ScannedItem(item_id="2", relpath="a/y.pdf", sha256="e" * 64),
        )

    def _assemble(self, planned: tuple[PlannedName, ...]):
        return assemble_plan(
            run_id="r",
            workflow_id="w",
            workflow_digest="wd",
            decision_digest="dd",
            scanned=self._scanned(),
            planned=planned,
        )

    def test_unresolved_item_keeps_a_reason_and_no_target(self) -> None:
        plan = self._assemble(
            (
                PlannedName(item_id="1", directory="a", name="new.pdf"),
                PlannedName(item_id="2", directory="a", name=None, reason="missing_field:vendor"),
            )
        )
        assert plan.balanced(2)
        unresolved = [e for e in plan.entries if e.action is ChangeAction.UNRESOLVED]
        assert len(unresolved) == 1
        assert unresolved[0].reason == "missing_field:vendor"

    def test_item_with_no_decision_is_quarantined_not_dropped(self) -> None:
        plan = self._assemble((PlannedName(item_id="1", directory="a", name="new.pdf"),))
        assert plan.balanced(2), "every scanned item must appear exactly once"
        assert plan.count(ChangeAction.UNRESOLVED) == 1

    def test_identical_target_is_unchanged_not_changed(self) -> None:
        plan = self._assemble((PlannedName(item_id="1", directory="a", name="x.pdf"),))
        entry = next(e for e in plan.entries if e.item_id == "1")
        assert entry.action is ChangeAction.UNCHANGED

    def test_decision_for_an_unscanned_item_is_refused(self) -> None:
        with pytest.raises(AssemblyError, match="unscanned"):
            self._assemble((PlannedName(item_id="ghost", directory="a", name="x.pdf"),))

    def test_plan_is_byte_identical_across_runs(self) -> None:
        planned = (PlannedName(item_id="1", directory="a", name="n.pdf"),)
        assert self._assemble(planned).digest() == self._assemble(planned).digest()


class TestObligations:
    def test_unbalanced_plan_fails(self) -> None:
        result = run_check("o", "balanced", {"entries": [{"item_id": "1"}], "scanned": 2})
        assert not result.passed

    def test_collision_is_detected_case_insensitively(self) -> None:
        result = run_check(
            "o",
            "no_collisions",
            {"entries": [{"target_relpath": "a/X.pdf"}, {"target_relpath": "a/x.pdf"}]},
        )
        assert not result.passed

    def test_changed_entry_without_a_target_fails(self) -> None:
        result = run_check(
            "o",
            "resolved_or_quarantined",
            {"entries": [{"item_id": "1", "action": "changed", "target_relpath": None}]},
        )
        assert not result.passed

    def test_unresolved_entry_without_a_reason_fails(self) -> None:
        result = run_check(
            "o",
            "resolved_or_quarantined",
            {"entries": [{"item_id": "1", "action": "unresolved", "reason": ""}]},
        )
        assert not result.passed

    def test_spec_frozen_compares_digests(self) -> None:
        from shakespeare.contracts import content_digest

        spec = {"template": "{vendor}"}
        assert run_check("o", "spec_frozen", {"spec": spec, "digest": content_digest(spec)}).passed
        assert not run_check("o", "spec_frozen", {"spec": spec, "digest": "wrong"}).passed

    def test_rendered_mechanically_catches_a_name_from_nowhere(self) -> None:
        """Defence in depth behind plan.assemble's refusal of hand-written names.

        A plan entry that names a file must correspond to a render candidate.
        """
        result = run_check(
            "o",
            "rendered_mechanically",
            {
                "entries": [{"item_id": "ghost", "target_relpath": "invented.pdf"}],
                "candidates": [{"item_id": "real"}],
            },
        )
        assert not result.passed
        assert result.detail["unaccounted"] == ["ghost"]

    def test_rendered_mechanically_tolerates_a_collision_suffix(self) -> None:
        """Collision resolution rewrites names, so the check is on identity not strings."""
        result = run_check(
            "o",
            "rendered_mechanically",
            {
                "entries": [{"item_id": "a", "target_relpath": "x (2).pdf"}],
                "candidates": [{"item_id": "a", "name": "x.pdf"}],
            },
        )
        assert result.passed

    def test_rendered_mechanically_ignores_quarantined_items(self) -> None:
        result = run_check(
            "o",
            "rendered_mechanically",
            {
                "entries": [{"item_id": "q", "target_relpath": None, "reason": "ocr_unavailable"}],
                "candidates": [],
            },
        )
        assert result.passed

    def test_unknown_check_is_refused(self) -> None:
        with pytest.raises(AssemblyError, match="unknown obligation check"):
            run_check("o", "rm -rf /", {})


class TestDeclaredOutputs:
    """Declared outputs are what an agent binds from, so a wrong one is a broken wire.

    A live model bound `planned` from `collide.planned` because nothing told it the
    operator produces `resolutions`.
    """

    def test_every_composable_operator_declares_its_outputs(self) -> None:
        from shakespeare.operators.builtin import RUNTIME_ONLY
        from shakespeare.operators.contracts import OUTPUT_KEYS

        composable = {name for name in BUILTIN if name not in RUNTIME_ONLY}
        assert composable == set(OUTPUT_KEYS)

    def test_declared_outputs_match_what_the_runner_returns(self, tmp_path: pathlib.Path) -> None:
        from shakespeare.operators.contracts import OUTPUT_KEYS
        from shakespeare.runners import pure_transform, readonly_scan

        source = tmp_path / "in"
        source.mkdir()
        (source / "a.pdf").write_bytes(b"x")

        produced = {
            "fs.scan": readonly_scan({"operation": "walk", "root": str(source)}, tmp_path),
            "fs.dirs": readonly_scan(
                {"operation": "directories", "root": str(source)}, tmp_path
            ),
            "text.normalize": pure_transform(
                {"operation": "normalize", "values": {"v": "x"}}, tmp_path
            ),
            "spec.freeze": pure_transform(
                {
                    "operation": "freeze_spec",
                    "spec": {"template": "{vendor}", "fields": [{"name": "vendor"}]},
                },
                tmp_path,
            ),
            "name.collide": pure_transform(
                {"operation": "collision_resolve", "candidates": []}, tmp_path
            ),
        }
        for name, output in produced.items():
            assert set(OUTPUT_KEYS[name]) == set(output), (
                f"{name} declares {OUTPUT_KEYS[name]} but returns {sorted(output)}"
            )


class TestUnreadableAccounting:
    """A file the scanner could not read must still end in a visible terminal state.

    Live, `locked.pdf` was in the source tree and appeared nowhere in the plan or the
    output — silently absent rather than reported.
    """

    def _plan(self, skipped: tuple[dict[str, str], ...]):
        return assemble_plan(
            run_id="r",
            workflow_id="w",
            workflow_digest="wd",
            decision_digest="dd",
            scanned=(ScannedItem(item_id="1", relpath="a.pdf", sha256="d" * 64),),
            planned=(PlannedName(item_id="1", directory="", name="A.pdf"),),
            skipped=skipped,
        )

    def test_a_skipped_file_appears_as_unresolved(self) -> None:
        plan = self._plan(({"relpath": "locked.pdf", "reason": "unreadable:PermissionError"},))
        entry = next(e for e in plan.entries if e.source_ref == "locked.pdf")
        assert entry.action is ChangeAction.UNRESOLVED
        assert entry.reason == "unreadable:PermissionError"

    def test_the_balance_counts_scanned_and_skipped(self) -> None:
        plan = self._plan(({"relpath": "locked.pdf", "reason": "unreadable"},))
        assert plan.balanced(2)
        assert len(plan.entries) == 2

    def test_no_skipped_files_changes_nothing(self) -> None:
        assert len(self._plan(()).entries) == 1
