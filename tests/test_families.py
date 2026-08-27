"""Family manifests.

`family.yml` and `family-context.yml` were written but never read, which meant
`allowed_features` bounded nothing and a family could ship an incomplete card unnoticed —
as `filesystem_mutation` had been doing since it was written.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from shakespeare.components import families
from shakespeare.components.registry import FAMILY_RUNNERS
from shakespeare.contracts import OperatorFamily, SemanticCard


class TestManifests:
    def test_every_family_has_a_manifest_and_a_complete_card(self) -> None:
        loaded = families.load_all()
        assert set(loaded) == set(OperatorFamily)
        for family, (manifest, card) in loaded.items():
            assert manifest.allowed_features, f"{family} declares no configuration slots"
            for field in SemanticCard.model_fields:
                assert getattr(card, field).strip(), f"{family} card field {field} is empty"

    def test_a_manifest_pins_its_trusted_runner(self) -> None:
        for family in OperatorFamily:
            assert families.manifest(family).entrypoint == FAMILY_RUNNERS[family]

    def test_an_incomplete_card_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "templates"
        for family in OperatorFamily:
            directory = root / family
            directory.mkdir(parents=True)
            (directory / "family.yml").write_text(
                yaml.safe_dump(
                    {"family": str(family), "revision": "1.0", "allowed_features": ["a"]}
                )
            )
            card = {name: "filled" for name in SemanticCard.model_fields}
            if family is OperatorFamily.PURE_TRANSFORM:
                card.pop("side_effects")  # the defect this check exists to catch
            (directory / "family-context.yml").write_text(yaml.safe_dump(card))

        families.load_all.cache_clear()
        with pytest.raises(families.FamilyError, match="all ten fields"):
            families.load_all(str(root))
        families.load_all.cache_clear()

    def test_a_manifest_in_the_wrong_directory_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "templates"
        for family in OperatorFamily:
            directory = root / family
            directory.mkdir(parents=True)
            declared = (
                OperatorFamily.READONLY_SCAN
                if family is OperatorFamily.PURE_TRANSFORM
                else family
            )
            (directory / "family.yml").write_text(
                yaml.safe_dump(
                    {"family": str(declared), "revision": "1.0", "allowed_features": ["a"]}
                )
            )
            (directory / "family-context.yml").write_text(
                yaml.safe_dump({name: "filled" for name in SemanticCard.model_fields})
            )
        families.load_all.cache_clear()
        with pytest.raises(families.FamilyError, match="but lives in"):
            families.load_all(str(root))
        families.load_all.cache_clear()


class TestFeatureBounds:
    def test_a_declared_slot_is_accepted(self) -> None:
        families.check_features(OperatorFamily.PURE_TRANSFORM, frozenset({"normalize"}))

    def test_an_undeclared_slot_is_refused(self) -> None:
        """Otherwise allowed_features is decoration and a request may name any slot."""
        with pytest.raises(families.FamilyError, match="does not allow"):
            families.check_features(OperatorFamily.PURE_TRANSFORM, frozenset({"exec_shell"}))

    def test_a_slot_from_another_family_is_refused(self) -> None:
        with pytest.raises(families.FamilyError, match="does not allow"):
            families.check_features(OperatorFamily.PURE_TRANSFORM, frozenset({"atomic_move"}))

    def test_no_features_is_allowed(self) -> None:
        families.check_features(OperatorFamily.READONLY_SCAN, frozenset())


class TestMarkerVerification:
    def _package(self, tmp_path: Path, family: str, revision: str) -> Path:
        package = tmp_path / "pkg"
        package.mkdir()
        (package / ".operator-template.yml").write_text(
            yaml.safe_dump({"family": family, "revision": revision})
        )
        return package

    def test_a_matching_marker_is_accepted(self, tmp_path: Path) -> None:
        package = self._package(tmp_path, "pure_transform", "1.0")
        families.verify_marker(package, OperatorFamily.PURE_TRANSFORM)

    def test_a_mismatched_family_is_refused(self, tmp_path: Path) -> None:
        """A package could otherwise claim any family it liked."""
        package = self._package(tmp_path, "pure_transform", "1.0")
        with pytest.raises(families.FamilyError, match="but was rendered as"):
            families.verify_marker(package, OperatorFamily.READONLY_SCAN)

    def test_a_stale_revision_is_refused(self, tmp_path: Path) -> None:
        package = self._package(tmp_path, "pure_transform", "0.9")
        with pytest.raises(families.FamilyError, match="revision"):
            families.verify_marker(package, OperatorFamily.PURE_TRANSFORM)

    def test_a_missing_marker_is_refused(self, tmp_path: Path) -> None:
        package = tmp_path / "empty"
        package.mkdir()
        with pytest.raises(families.FamilyError, match="no .operator-template.yml"):
            families.verify_marker(package, OperatorFamily.PURE_TRANSFORM)


class TestAdmissionEnforcesFeatures:
    def test_a_request_naming_an_undeclared_slot_escalates(self, tmp_path: Path) -> None:
        from shakespeare.admission import AdmissionService
        from shakespeare.components.builtin import build_registry
        from shakespeare.contracts import AdmissionDisposition, OperatorRequest, RequestKind
        from shakespeare.runtime.audit import AuditStore

        from test_admission import StubRenderer, passing_tests

        audit = AuditStore(tmp_path / "audit.sqlite3")
        service = AdmissionService(
            registry=build_registry(),
            audit=audit,
            workspace=tmp_path / "candidates",
            renderer=StubRenderer(),
            test_runner=passing_tests,
        )
        report, _ = service.evaluate(
            OperatorRequest(
                request_id="r",
                run_id="run",
                domain_id="d",
                kind=RequestKind.VARIANT,
                family=OperatorFamily.PURE_TRANSFORM,
                name="text.sneak",
                features=frozenset({"normalize", "exec_shell"}),
                rationale="needs a shell",
            )
        )
        assert report.disposition is AdmissionDisposition.HUMAN_REVIEW
        assert any(f.code == "feature_not_allowed" for f in report.findings)
        audit.close()


class TestAManifestGovernsWhatShips:
    """The manifest bounded only operators that were asked for, never the ones that ship.

    `families.py` is imported by admission and the CLI and by nothing on the built-in
    registration path, so `allowed_features` was checked for a requested operator and never
    for a registered one. It drifted: `pure_transform` declared `plan_batches` after that
    operation was renamed to `plan_batch`, and nothing noticed for a whole session.
    """

    def test_every_family_declares_its_own_operations(self) -> None:
        """Otherwise the family is unrequestable, not merely undocumented.

        A request selects its operation by naming it in `features`, and `check_features`
        rejects a name the manifest does not declare — so a manifest omitting its own
        operations refuses every request for that family before the operation is read.
        """
        from shakespeare.components.runners import allowlist

        for family in OperatorFamily:
            undeclared = sorted(set(allowlist(family)) - families.allowed_features(family))
            assert not undeclared, f"{family} runs {undeclared} but does not declare them"

    def test_a_declared_name_is_either_an_operation_or_a_slot(self) -> None:
        """Naming something that is neither is how `plan_batches` survived a rename."""
        from shakespeare.components.runners import allowlist

        slots = {
            "char_limit", "depth_limit", "digest", "docx", "email", "fallback_chain",
            "image_ocr", "include_hidden", "key", "mime_detect", "page_limit", "pdf_ocr",
            "pdf_text", "stable_sort", "stat", "table", "xlsx",
        }
        for family in OperatorFamily:
            unexplained = sorted(
                families.allowed_features(family) - set(allowlist(family)) - slots
            )
            assert not unexplained, f"{family} declares {unexplained}, neither operation nor slot"

    def test_the_template_offers_every_family(self) -> None:
        """A family the template cannot render is a family nothing can be admitted into."""
        import yaml

        copier = yaml.safe_load(
            (families.TEMPLATE_ROOT / "copier.yml").read_text()
        )
        offered = set(copier["operator_family"]["choices"])
        assert offered == {family.value for family in OperatorFamily}

    def test_a_request_naming_a_real_operation_passes_its_family_check(self) -> None:
        """The end the drift was breaking: readonly_scan and content_extract were
        unrequestable, because neither declared any operation a request could name."""
        from shakespeare.components.runners import allowlist

        for family in OperatorFamily:
            operation = sorted(allowlist(family))[0]
            families.check_features(family, frozenset({operation}))
