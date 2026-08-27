"""Extraction against genuinely-parseable files.

The rest of the suite uses synthetic bytes, which only exercises the failure paths. These
are real PDFs, DOCX, XLSX, email and images, so a backend that no longer matches its
library's API fails here rather than on a user's invoices.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from shakespeare.domain.extraction import Backend, extract
from shakespeare.domain.filesystem import scan

from fixtures.build import build_tree, cleanup


@pytest.fixture
def tree(tmp_path: Path):
    root = tmp_path / "tree"
    made = build_tree(root)
    yield root, made
    cleanup(root)
    shutil.rmtree(root, ignore_errors=True)


def _extract(path: Path, media_type: str, backend: Backend = Backend.AUTO_CHAIN):
    return extract(item_id="1", path=path, media_type=media_type, backend=backend)


class TestRealFormats:
    def test_pdf_text_layer_is_read(self, tree) -> None:
        _, made = tree
        result = _extract(made["digital_pdf"], "application/pdf")
        assert result.usable, result.unavailable_reason
        assert "Northwind Traders" in result.text
        assert "INV-4471" in result.text

    def test_pdf_spans_carry_page_provenance(self, tree) -> None:
        """Field values must be able to cite where they came from."""
        _, made = tree
        result = _extract(made["digital_pdf"], "application/pdf", Backend.PDF_TEXT)
        assert result.spans
        assert result.spans[0].page == 1
        assert result.spans[0].end > result.spans[0].start

    def test_docx_paragraphs_and_tables_are_read(self, tree) -> None:
        _, made = tree
        result = _extract(
            made["docx"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert result.usable, result.unavailable_reason
        assert "INV-4471" in result.text
        assert "1234.50" in result.text, "table cells must be extracted too"

    def test_xlsx_values_are_read(self, tree) -> None:
        _, made = tree
        result = _extract(
            made["xlsx"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        assert result.usable, result.unavailable_reason
        assert "Contoso Supply Co" in result.text
        assert "INV-10233" in result.text

    def test_email_headers_and_body_are_read(self, tree) -> None:
        _, made = tree
        result = _extract(made["eml"], "message/rfc822")
        assert result.usable, result.unavailable_reason
        assert "Invoice INV-4472" in result.text
        assert "PO-88121" in result.text


class TestDegradation:
    def test_a_scan_with_no_text_layer_says_so(self, tree) -> None:
        """This is what routes an item to OCR, or to quarantine when OCR is unavailable."""
        _, made = tree
        result = _extract(made["scanned_pdf"], "application/pdf", Backend.PDF_TEXT)
        assert not result.usable
        assert result.unavailable_reason == "no_text_layer"

    def test_auto_chain_falls_back_and_reports_the_last_reason(self, tree) -> None:
        _, made = tree
        result = _extract(made["scanned_pdf"], "application/pdf")
        assert not result.usable
        # Either OCR ran and found nothing, or it is unavailable. Both are explicit.
        assert result.unavailable_reason is not None
        assert result.unavailable_reason.startswith(
            ("ocr_unavailable", "backend_unavailable", "ocr_produced_no_text", "unreadable")
        )

    def test_a_zero_byte_file_is_a_reason_not_a_crash(self, tree) -> None:
        _, made = tree
        result = _extract(made["zero_byte"], "application/pdf")
        assert not result.usable
        assert result.unavailable_reason

    def test_an_unreadable_file_is_a_reason_not_a_crash(self, tree) -> None:
        _, made = tree
        result = _extract(made["permission_denied"], "application/pdf")
        assert not result.usable
        assert result.unavailable_reason

    def test_char_limit_is_honoured(self, tree) -> None:
        """An unbounded document must not become an unbounded prompt."""
        from shakespeare.domain.extraction import ExtractOptions

        _, made = tree
        unclipped = _extract(
            made["docx"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Backend.DOCX,
        )
        result = extract(
            item_id="1",
            path=made["docx"],
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            backend=Backend.DOCX,
            options=ExtractOptions(page_limit=1, char_limit=100),
        )
        assert len(unclipped.text) > 100, "the fixture must be long enough to clip"
        assert len(result.text) == 100


class TestScanOverTheRealTree:
    def test_unreadable_files_are_reported_not_dropped(self, tree) -> None:
        """Accounting must balance, so a file the scan cannot read still has to appear."""
        root, _ = tree
        items, skipped = scan(root)
        seen = {item.relpath for item in items} | {item["relpath"] for item in skipped}
        assert "locked.pdf" in seen

    def test_media_types_are_detected(self, tree) -> None:
        root, _ = tree
        items, _ = scan(root)
        by_path = {item.relpath: item.media_type for item in items}
        assert by_path["2024/q1/invoice_4471.pdf"] == "application/pdf"
        assert by_path["mail/invoice_4472.eml"] == "message/rfc822"
        assert by_path["scans/IMG_9001.png"] == "image/png"


@pytest.mark.skipif(
    __import__("shutil").which("tesseract") is None, reason="tesseract is not installed"
)
class TestOcr:
    """Runs only where tesseract exists.

    Without it these are skipped rather than silently passing, so the OCR path is never
    mistaken for covered.
    """

    def test_image_ocr_reads_rendered_text(self, tree) -> None:
        _, made = tree
        result = _extract(made["png"], "image/png", Backend.IMAGE_OCR)
        assert result.usable, result.unavailable_reason
        assert "INV" in result.text.upper()

    def test_pdf_ocr_reads_a_scan(self, tree) -> None:
        _, made = tree
        result = _extract(made["scanned_pdf"], "application/pdf", Backend.PDF_OCR)
        assert result.unavailable_reason != "backend_unavailable:pytesseract"
