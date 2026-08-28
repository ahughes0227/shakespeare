"""Golden tests for the determinism core.

Consistency in this program is a property of these functions, not of any agent, so these
tests are the ones that actually guarantee it.
"""

from __future__ import annotations

import random

import pytest
from system.domain.naming import (
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


class TestConfidenceFloor:
    """A value present but uncertain must quarantine, exactly like a missing one.

    This is the system's central safety claim — 'never guessed at' — and until now the
    declared confidence_floor was dead code that nothing read.
    """

    def _fields(self, floor: float = 0.7):
        return (
            FieldDecl(name="vendor", confidence_floor=floor),
            FieldDecl(name="invoice_number", confidence_floor=floor),
            FieldDecl(name="po_number", required=False, confidence_floor=floor),
        )

    def _render(self, confidences: dict[str, float] | None, **kwargs):
        return render(
            item_id="1",
            template="{vendor}, {invoice_number}, {po_number}",
            fields=self._fields(),
            values={"vendor": "ACME", "invoice_number": "INV-1", "po_number": "PO-1"},
            policy=NamePolicy(),
            extension=".pdf",
            confidences=confidences,
            **kwargs,
        )

    def test_a_confident_value_renders(self) -> None:
        assert self._render({"vendor": 0.95}).rendered == "ACME, INV-1, PO-1.pdf"

    def test_a_low_confidence_required_field_quarantines(self) -> None:
        result = self._render({"vendor": 0.4})
        assert result.rendered is None
        assert result.reason is not None
        assert result.reason.startswith("low_confidence:vendor")

    def test_the_reason_records_the_number_and_the_threshold(self) -> None:
        """A human triaging the quarantine needs to see how close it was."""
        assert self._render({"vendor": 0.55}).reason == "low_confidence:vendor:0.55<0.70"

    def test_a_low_confidence_optional_field_is_simply_omitted(self) -> None:
        assert self._render({"po_number": 0.1}).rendered == "ACME, INV-1.pdf"

    def test_absent_confidence_is_not_treated_as_zero(self) -> None:
        """An agent that reports no confidence must not have every file quarantined."""
        assert self._render(None).rendered == "ACME, INV-1, PO-1.pdf"
        assert self._render({}).rendered == "ACME, INV-1, PO-1.pdf"

    def test_exactly_at_the_floor_is_accepted(self) -> None:
        assert self._render({"vendor": 0.7}).rendered is not None

    def test_a_run_floor_can_tighten_but_not_loosen_a_field_floor(self) -> None:
        """The stricter of the two wins, so config can never weaken a spec."""
        assert self._render({"vendor": 0.8}, floor=0.9).rendered is None
        assert self._render({"vendor": 0.8}, floor=0.1).rendered is not None

    def test_a_sequence_field_is_never_confidence_checked(self) -> None:
        result = render(
            item_id="1",
            template="{seq:03d}",
            fields=(FieldDecl(name="seq", kind=FieldKind.SEQUENCE, format="03d"),),
            values={},
            policy=NamePolicy(),
            extension=".pdf",
            sequence=7,
            confidences={"seq": 0.0},
            floor=0.99,
        )
        assert result.rendered == "007.pdf"


class TestDateTolerance:
    """A value already in the field's own format is not an error.

    A live model asked for "%Y%m" supplied "202402", and the renderer rejected it —
    a contract arguing with itself, and every invoice was quarantined for it.
    """

    def _render(self, value: str, fmt: str = "%Y%m"):
        return render(
            item_id="1",
            template="{invoice_date}",
            fields=(FieldDecl(name="invoice_date", kind=FieldKind.DATE, format=fmt),),
            values={"invoice_date": value},
            policy=NamePolicy(),
            extension=".pdf",
        )

    def test_an_iso_date_is_formatted(self) -> None:
        assert self._render("2024-02-11").rendered == "202402.pdf"

    def test_a_value_already_in_the_target_format_passes_through(self) -> None:
        assert self._render("202402").rendered == "202402.pdf"

    def test_an_iso_datetime_is_accepted(self) -> None:
        assert self._render("2024-02-11T09:15:00").rendered == "202402.pdf"

    def test_another_format_is_honoured_both_ways(self) -> None:
        assert self._render("2024-02-11", "%Y-%m-%d").rendered == "2024-02-11.pdf"
        assert self._render("11/02/2024", "%d/%m/%Y").rendered == "11-02-2024.pdf"

    def test_genuine_nonsense_is_still_refused(self) -> None:
        result = self._render("not a date at all")
        assert result.rendered is None
        assert result.reason is not None
        assert "expected an ISO date" in result.reason


class TestExtensionTolerance:
    """The extension carries the file type, so a missing dot changes what the file is.

    A live model supplied "pdf" rather than ".pdf" and produced `...po-88120pdf`.
    """

    def _render(self, extension: str):
        return render(
            item_id="1",
            template="{vendor}",
            fields=(FieldDecl(name="vendor"),),
            values={"vendor": "ACME"},
            policy=NamePolicy(),
            extension=extension,
        )

    def test_a_dotted_extension_is_unchanged(self) -> None:
        assert self._render(".pdf").rendered == "ACME.pdf"

    def test_a_bare_extension_gains_its_dot(self) -> None:
        assert self._render("pdf").rendered == "ACME.pdf"

    def test_no_extension_stays_absent(self) -> None:
        assert self._render("").rendered == "ACME"

    def test_case_is_preserved_either_way(self) -> None:
        assert self._render("PDF").rendered == "ACME.PDF"

    def test_a_clamped_name_still_keeps_the_dot(self) -> None:
        result = render(
            item_id="1",
            template="{vendor}",
            fields=(FieldDecl(name="vendor"),),
            values={"vendor": "V" * 300},
            policy=NamePolicy(max_length=32),
            extension="pdf",
        )
        assert result.rendered is not None
        assert result.rendered.endswith(".pdf")
        assert len(result.rendered) <= 32


class TestPerFieldCap:
    """A long vendor name should lose its own tail, not push the invoice number off.

    A live model reached for per-field caps twice, writing them into policy.max_length
    where only a whole-name cap belongs.
    """

    def _render(self, cap: int | None):
        return render(
            item_id="1",
            template="{vendor}, {invoice_number}",
            fields=(
                FieldDecl(name="vendor", max_length=cap),
                FieldDecl(name="invoice_number"),
            ),
            values={"vendor": "Northwind Traders Limited", "invoice_number": "INV-4471"},
            policy=NamePolicy(),
            extension=".pdf",
        )

    def test_a_capped_field_is_truncated(self) -> None:
        assert self._render(8).rendered == "Northwin, INV-4471.pdf"

    def test_the_other_fields_survive_intact(self) -> None:
        assert self._render(8).rendered is not None
        assert "INV-4471" in self._render(8).rendered

    def test_no_cap_leaves_the_value_whole(self) -> None:
        assert self._render(None).rendered == "Northwind Traders Limited, INV-4471.pdf"

    def test_a_cap_longer_than_the_value_changes_nothing(self) -> None:
        assert self._render(100).rendered == "Northwind Traders Limited, INV-4471.pdf"

    def test_truncation_does_not_leave_trailing_punctuation(self) -> None:
        result = render(
            item_id="1",
            template="{vendor}",
            fields=(FieldDecl(name="vendor", max_length=10),),
            values={"vendor": "Acme Ltd - Northern"},
            policy=NamePolicy(),
            extension=".pdf",
        )
        assert result.rendered == "Acme Ltd.pdf"
