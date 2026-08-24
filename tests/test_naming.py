"""Golden tests for the determinism core.

Consistency in this program is a property of these functions, not of any agent, so these
tests are the ones that actually guarantee it.
"""

from __future__ import annotations

import random

import pytest
from shakespeare.operators.naming import (
    Candidate,
    CasePolicy,
    CollisionPolicy,
    FieldDecl,
    FieldKind,
    NamePolicy,
    render,
    resolve_collisions,
    sanitize,
)

INVOICE_FIELDS = (
    FieldDecl(name="invoice_date", kind=FieldKind.DATE, format="%Y%m"),
    FieldDecl(name="vendor"),
    FieldDecl(name="invoice_number"),
    FieldDecl(name="po_number", required=False),
)
TEMPLATE = "{invoice_date}, {vendor}, {invoice_number}, {po_number}"
VALUES = {
    "invoice_date": "2024-01-15",
    "vendor": "ACME Corporation",
    "invoice_number": "INV-99812",
    "po_number": "PO-44117",
}


def _render(values: dict[str, object], **kwargs: object):
    options: dict[str, object] = {
        "item_id": "1",
        "template": TEMPLATE,
        "fields": INVOICE_FIELDS,
        "values": values,
        "policy": NamePolicy(),
        "extension": ".pdf",
    }
    options.update(kwargs)
    return render(**options)  # type: ignore[arg-type]


class TestRender:
    def test_invoice_golden(self) -> None:
        assert _render(VALUES).rendered == "202401, ACME Corporation, INV-99812, PO-44117.pdf"

    def test_is_pure(self) -> None:
        assert _render(VALUES).rendered == _render(VALUES).rendered

    @pytest.mark.parametrize("illegal", ['<', '>', ':', '"', "/", "\\", "|", "?", "*"])
    def test_illegal_characters_are_replaced(self, illegal: str) -> None:
        result = _render({**VALUES, "vendor": f"AC{illegal}ME"})
        assert result.rendered is not None
        assert illegal not in result.rendered

    def test_control_characters_are_replaced(self) -> None:
        result = _render({**VALUES, "vendor": "AC\x07ME\x00"})
        assert result.rendered is not None
        assert all(ord(char) >= 32 for char in result.rendered)

    def test_unicode_is_preserved_and_normalised(self) -> None:
        # Composed and decomposed forms must render identically or two runs disagree.
        composed = _render({**VALUES, "vendor": "Café"}).rendered
        decomposed = _render({**VALUES, "vendor": "Café"}).rendered
        assert composed == decomposed
        assert composed is not None and "Café" in composed

    def test_extension_is_preserved_exactly(self) -> None:
        assert _render(VALUES, extension=".PDF").rendered.endswith(".PDF")

    def test_long_names_clamp_the_stem_not_the_extension(self) -> None:
        result = _render({**VALUES, "vendor": "V" * 400}, policy=NamePolicy(max_length=64))
        assert result.rendered is not None
        assert len(result.rendered) <= 64
        assert result.rendered.endswith(".pdf")

    def test_missing_required_field_is_reported_not_guessed(self) -> None:
        result = _render({"vendor": "ACME", "invoice_number": "INV-1"})
        assert result.rendered is None
        assert result.reason == "missing_field:invoice_date"

    def test_blank_required_field_is_treated_as_missing(self) -> None:
        result = _render({**VALUES, "vendor": "   "})
        assert result.rendered is None
        assert result.reason == "missing_field:vendor"

    def test_optional_field_may_be_absent(self) -> None:
        result = _render({k: v for k, v in VALUES.items() if k != "po_number"})
        assert result.rendered == "202401, ACME Corporation, INV-99812.pdf"

    def test_absent_optional_field_leaves_no_dangling_separator(self) -> None:
        result = _render(
            {**VALUES, "po_number": None},
            template="{vendor}, {po_number}, {invoice_number}",
        )
        assert result.rendered == "ACME Corporation, INV-99812.pdf"

    def test_comma_inside_a_value_survives_separator_tidying(self) -> None:
        # Only *runs* of separators collapse; a real comma in a vendor name must remain.
        result = _render({**VALUES, "vendor": "Smith, Jones Ltd"}, template="{vendor}")
        assert result.rendered == "Smith, Jones Ltd.pdf"

    def test_undeclared_placeholder_is_refused(self) -> None:
        result = _render(VALUES, template="{vendor}, {not_declared}")
        assert result.rendered is None
        assert result.reason == "undeclared_field:not_declared"

    def test_unformattable_date_is_reported(self) -> None:
        result = _render({**VALUES, "invoice_date": "not-a-date"})
        assert result.rendered is None
        assert result.reason is not None
        assert result.reason.startswith("unformattable_field:invoice_date")

    def test_windows_reserved_stem_is_escaped(self) -> None:
        result = _render({"vendor": "CON"}, template="{vendor}")
        assert result.rendered is not None
        assert result.rendered != "CON.pdf"

    @pytest.mark.parametrize(
        ("case", "expected"),
        [
            (CasePolicy.LOWER, "acme corporation"),
            (CasePolicy.UPPER, "ACME CORPORATION"),
            (CasePolicy.PRESERVE, "ACME Corporation"),
        ],
    )
    def test_case_policy(self, case: CasePolicy, expected: str) -> None:
        result = _render(VALUES, template="{vendor}", policy=NamePolicy(case=case))
        assert result.rendered == f"{expected}.pdf"

    def test_alias_map_canonicalises_a_vendor(self) -> None:
        policy = NamePolicy(aliases={"ACME Corporation": "ACME"})
        result = _render(VALUES, template="{vendor}", policy=policy)
        assert result.rendered == "ACME.pdf"


class TestSequence:
    def test_sequence_is_filled_from_scan_order(self) -> None:
        result = _render(VALUES, template="{seq:04d} - {vendor}", sequence=7)
        assert result.rendered == "0007 - ACME Corporation.pdf"

    def test_sequence_without_a_value_is_reported(self) -> None:
        result = _render(VALUES, template="{seq} - {vendor}", sequence=None)
        assert result.rendered is None
        assert result.reason == "missing_sequence"

    def test_pure_sequential_rename_needs_no_extracted_field(self) -> None:
        # The case that must cost zero per-file model calls.
        result = render(
            item_id="1",
            template="{seq:03d}",
            fields=(FieldDecl(name="seq", kind=FieldKind.SEQUENCE, format="03d"),),
            values={},
            policy=NamePolicy(),
            extension=".pdf",
            sequence=12,
        )
        assert result.rendered == "012.pdf"


class TestCollisions:
    def _candidates(self) -> list[Candidate]:
        return [
            Candidate(item_id="a", directory="inv", name="202401, ACME.pdf"),
            Candidate(item_id="b", directory="inv", name="202401, ACME.pdf"),
            Candidate(item_id="c", directory="inv", name="202401, ACME.pdf"),
            Candidate(item_id="d", directory="other", name="202401, ACME.pdf"),
        ]

    def test_suffixes_duplicates_within_a_directory(self) -> None:
        resolved = resolve_collisions(tuple(self._candidates()), CollisionPolicy.SUFFIX_N)
        names = {item.item_id: item.name for item in resolved}
        assert names["a"] == "202401, ACME.pdf"
        assert names["b"] == "202401, ACME (2).pdf"
        assert names["c"] == "202401, ACME (3).pdf"
        # A different directory is not a collision.
        assert names["d"] == "202401, ACME.pdf"

    def test_order_independence(self) -> None:
        """Shuffled input must produce byte-identical output.

        Otherwise two runs over the same files could disagree about which one keeps the
        unsuffixed name, and the plan would stop being reproducible.
        """
        baseline = resolve_collisions(tuple(self._candidates()), CollisionPolicy.SUFFIX_N)
        rng = random.Random(1729)
        for _ in range(10):
            shuffled = self._candidates()
            rng.shuffle(shuffled)
            assert resolve_collisions(tuple(shuffled), CollisionPolicy.SUFFIX_N) == baseline

    def test_case_insensitive_collision(self) -> None:
        # Two names differing only in case collide on macOS and Windows.
        resolved = resolve_collisions(
            (
                Candidate(item_id="a", directory="d", name="Invoice.pdf"),
                Candidate(item_id="b", directory="d", name="invoice.pdf"),
            ),
            CollisionPolicy.SUFFIX_N,
        )
        assert {item.name for item in resolved} == {"Invoice.pdf", "invoice (2).pdf"}

    def test_fail_policy_quarantines_rather_than_renaming(self) -> None:
        resolved = resolve_collisions(tuple(self._candidates()), CollisionPolicy.FAIL)
        unresolved = [item for item in resolved if item.name is None]
        assert len(unresolved) == 2
        assert all(item.reason is not None for item in unresolved)

    def test_hash_suffix_is_stable(self) -> None:
        first = resolve_collisions(tuple(self._candidates()), CollisionPolicy.HASH_SUFFIX)
        second = resolve_collisions(tuple(self._candidates()), CollisionPolicy.HASH_SUFFIX)
        assert first == second


class TestSanitize:
    def test_strips_trailing_dots_and_spaces(self) -> None:
        # Windows silently drops these, which would desynchronise plan from reality.
        assert sanitize("name. . ", NamePolicy()) == "name"

    def test_collapses_internal_whitespace(self) -> None:
        assert sanitize("a   \t b", NamePolicy()) == "a b"
