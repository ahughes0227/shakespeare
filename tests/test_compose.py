"""The Hydra guard: a selection may never become code execution or path traversal."""

from __future__ import annotations

from typing import Any

import pytest
from shakespeare.runtime.compose import CompositionError, catalog, compose, validate_parameters


class TestCatalog:
    def test_is_derived_from_disk(self) -> None:
        groups = catalog()
        assert set(groups) == {
            "extract",
            "naming",
            "collision",
            "confidence",
            "write",
            "schedule",
        }
        assert "auto_chain" in groups["extract"]
        assert groups["collision"] == frozenset({"suffix_n", "hash_suffix", "fail"})


class TestComposition:
    def test_selection_overrides_the_default(self) -> None:
        resolved = compose({"extract": "pdf_text"})
        assert resolved["extract"]["backend"] == "pdf_text"

    def test_unselected_groups_fall_back_to_defaults(self) -> None:
        resolved = compose({"extract": "pdf_text"})
        assert resolved["confidence"]["floor"] == 0.7
        assert resolved["collision"]["policy"] == "suffix_n"

    def test_is_deterministic(self) -> None:
        first = compose({"extract": "docx"}, {"page_limit": 3})
        second = compose({"extract": "docx"}, {"page_limit": 3})
        assert first == second

    def test_unknown_group_is_refused(self) -> None:
        with pytest.raises(CompositionError, match="unknown config group"):
            compose({"nonexistent": "x"})

    def test_unknown_choice_is_refused(self) -> None:
        with pytest.raises(CompositionError, match="unknown choice"):
            compose({"extract": "definitely_not_a_backend"})

    def test_group_outside_the_domain_grant_is_refused(self) -> None:
        with pytest.raises(CompositionError, match="not granted"):
            compose({"collision": "fail"}, allowed_groups=frozenset({"extract"}))


class TestInjectionGuard:
    @pytest.mark.parametrize(
        "parameters",
        [
            {"_target_": "os.system"},
            {"_partial_": True},
            {"key": "${env:HOME}"},
            {"key": "${oc.env:AWS_SECRET_ACCESS_KEY}"},
            {"key": "../../etc/passwd"},
            {"key": "/etc/passwd"},
            {"key": "a\\b"},
            {"key": "+override"},
            {"key": "~delete"},
            {"nested": {"_target_": "os.system"}},
            {"listed": ["${env:HOME}"]},
        ],
        ids=lambda value: str(value)[:40],
    )
    def test_rejects_hydra_syntax_in_parameters(self, parameters: dict[str, Any]) -> None:
        with pytest.raises(CompositionError):
            validate_parameters(parameters)

    @pytest.mark.parametrize(
        "choice", ["pdf_text/../../etc", "pdf text", "PDF_TEXT", "_hidden", "a;b"]
    )
    def test_rejects_unsafe_choice_syntax(self, choice: str) -> None:
        with pytest.raises(CompositionError):
            compose({"extract": choice})

    def test_accepts_ordinary_bounded_parameters(self) -> None:
        validate_parameters({"page_limit": 5, "char_limit": 1000, "strict": True, "floor": 0.9})


class TestDataVersusConfigKeys:
    """The guard's key rule is about *argument names*, not about data.

    Applying it at every depth rejected an alias map keyed by vendor names — legitimate
    data, refused as an unsafe key. The same conflation had already cost a path separator
    fix; this pins the distinction so it is not made a third time.
    """

    def test_a_top_level_key_must_look_like_an_argument_name(self) -> None:
        for key in ("Not An Arg", "_hidden", "9lives", "with-dash"):
            with pytest.raises(CompositionError, match="unsafe parameter key"):
                validate_parameters({key: "value"})

    def test_nested_data_keys_may_be_arbitrary_strings(self) -> None:
        validate_parameters(
            {
                "spec": {
                    "policy": {
                        "aliases": {
                            "Northwind Traders Ltd.": "Northwind",
                            "ACME Corp & Sons": "ACME",
                            "société générale": "SocGen",
                        }
                    }
                }
            }
        )

    def test_nested_values_keyed_by_document_content_are_allowed(self) -> None:
        validate_parameters({"items": [{"values": {"Invoice No.": "INV-1", "P.O. #": "PO-1"}}]})

    @pytest.mark.parametrize(
        "directive", ["_target_", "_partial_", "_args_", "_convert_", "_recursive_"]
    )
    def test_hydra_directive_keys_are_refused_at_any_depth(self, directive: str) -> None:
        with pytest.raises(CompositionError, match="unsafe parameter key"):
            validate_parameters({"spec": {"policy": {directive: "os.system"}}})

    def test_interpolation_is_still_refused_in_nested_data(self) -> None:
        with pytest.raises(CompositionError):
            validate_parameters({"spec": {"aliases": {"vendor": "${env:HOME}"}}})
