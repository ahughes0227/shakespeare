"""Content extraction, one operator with config-selected backends.

Every backend degrades explicitly.  A missing OCR binary or an absent optional dependency
returns a reason rather than empty text, so the item is quarantined by the `extract`
obligation instead of silently producing a nameless file.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field

from ...contracts import Contract


class Backend(StrEnum):
    PDF_TEXT = "pdf_text"
    PDF_OCR = "pdf_ocr"
    DOCX = "docx"
    XLSX = "xlsx"
    EMAIL = "email"
    IMAGE_OCR = "image_ocr"
    #: Try backends in order until one yields text.  Fallback is configuration, not agent
    #: reasoning, because an agent never sees its own operator output.
    AUTO_CHAIN = "auto_chain"


class Span(Contract):
    """Where a piece of text came from, so a field value can cite its provenance."""

    page: int | None = None
    start: int
    end: int


class Extraction(Contract):
    item_id: str
    backend: str
    text: str = ""
    spans: tuple[Span, ...] = ()
    unavailable_reason: str | None = None
    char_count: int = 0

    @property
    def usable(self) -> bool:
        return bool(self.text.strip())


class ExtractOptions(Contract):
    page_limit: int = Field(default=20, ge=1, le=500)
    char_limit: int = Field(default=200_000, ge=100)


#: Ordered fallback for `auto_chain`, chosen by media type.
_CHAIN: dict[str, tuple[Backend, ...]] = {
    "application/pdf": (Backend.PDF_TEXT, Backend.PDF_OCR),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (Backend.DOCX,),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": (Backend.XLSX,),
    "message/rfc822": (Backend.EMAIL,),
    "application/vnd.ms-outlook": (Backend.EMAIL,),
    "image/png": (Backend.IMAGE_OCR,),
    "image/jpeg": (Backend.IMAGE_OCR,),
    "image/tiff": (Backend.IMAGE_OCR,),
    "text/plain": (Backend.PDF_TEXT,),
}


def _unavailable(item_id: str, backend: Backend, reason: str) -> Extraction:
    return Extraction(item_id=item_id, backend=str(backend), unavailable_reason=reason)


def _clip(text: str, options: ExtractOptions) -> str:
    return text[: options.char_limit]


def _pdf_text(item_id: str, path: Path, options: ExtractOptions) -> Extraction:
    try:
        import pdfplumber
    except ImportError:
        return _unavailable(item_id, Backend.PDF_TEXT, "backend_unavailable:pdfplumber")

    pages: list[str] = []
    spans: list[Span] = []
    cursor = 0
    try:
        with pdfplumber.open(path) as document:
            for number, page in enumerate(document.pages[: options.page_limit], start=1):
                text = page.extract_text() or ""
                if text:
                    spans.append(Span(page=number, start=cursor, end=cursor + len(text)))
                    cursor += len(text) + 1
                    pages.append(text)
    except Exception as exc:  # noqa: BLE001 - a corrupt document is data, not a crash
        return _unavailable(item_id, Backend.PDF_TEXT, f"unreadable:{type(exc).__name__}")

    text = _clip("\n".join(pages), options)
    if not text.strip():
        # A scanned PDF has no text layer.  Saying so lets auto_chain reach for OCR.
        return _unavailable(item_id, Backend.PDF_TEXT, "no_text_layer")
    return Extraction(
        item_id=item_id,
        backend=str(Backend.PDF_TEXT),
        text=text,
        spans=tuple(spans),
        char_count=len(text),
    )


def _ocr_available() -> str | None:
    try:
        import pytesseract
    except ImportError:
        return "backend_unavailable:pytesseract"
    try:
        pytesseract.get_tesseract_version()
    except Exception:  # noqa: BLE001 - the binary is absent or not on PATH
        return "ocr_unavailable:tesseract_binary_missing"
    return None


def _image_ocr(item_id: str, path: Path, options: ExtractOptions) -> Extraction:
    reason = _ocr_available()
    if reason:
        return _unavailable(item_id, Backend.IMAGE_OCR, reason)
    import pytesseract
    from PIL import Image

    try:
        with Image.open(path) as image:
            text = _clip(pytesseract.image_to_string(image), options)
    except Exception as exc:  # noqa: BLE001
        return _unavailable(item_id, Backend.IMAGE_OCR, f"unreadable:{type(exc).__name__}")
    if not text.strip():
        return _unavailable(item_id, Backend.IMAGE_OCR, "ocr_produced_no_text")
    return Extraction(
        item_id=item_id, backend=str(Backend.IMAGE_OCR), text=text, char_count=len(text)
    )


def _pdf_ocr(item_id: str, path: Path, options: ExtractOptions) -> Extraction:
    reason = _ocr_available()
    if reason:
        return _unavailable(item_id, Backend.PDF_OCR, reason)
    try:
        import pdfplumber
        import pytesseract
    except ImportError:
        return _unavailable(item_id, Backend.PDF_OCR, "backend_unavailable:pdfplumber")

    pages: list[str] = []
    try:
        with pdfplumber.open(path) as document:
            for page in document.pages[: options.page_limit]:
                image = page.to_image(resolution=200).original
                pages.append(pytesseract.image_to_string(image))
    except Exception as exc:  # noqa: BLE001
        return _unavailable(item_id, Backend.PDF_OCR, f"unreadable:{type(exc).__name__}")

    text = _clip("\n".join(pages), options)
    if not text.strip():
        return _unavailable(item_id, Backend.PDF_OCR, "ocr_produced_no_text")
    return Extraction(
        item_id=item_id, backend=str(Backend.PDF_OCR), text=text, char_count=len(text)
    )


def _docx(item_id: str, path: Path, options: ExtractOptions) -> Extraction:
    try:
        import docx
    except ImportError:
        return _unavailable(item_id, Backend.DOCX, "backend_unavailable:python-docx")
    try:
        document = docx.Document(str(path))
        parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append("\t".join(cell.text for cell in row.cells))
    except Exception as exc:  # noqa: BLE001
        return _unavailable(item_id, Backend.DOCX, f"unreadable:{type(exc).__name__}")
    text = _clip("\n".join(parts), options)
    if not text.strip():
        return _unavailable(item_id, Backend.DOCX, "document_has_no_text")
    return Extraction(item_id=item_id, backend=str(Backend.DOCX), text=text, char_count=len(text))


def _xlsx(item_id: str, path: Path, options: ExtractOptions) -> Extraction:
    try:
        import openpyxl
    except ImportError:
        return _unavailable(item_id, Backend.XLSX, "backend_unavailable:openpyxl")
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows: list[str] = []
        for sheet in workbook.worksheets[: options.page_limit]:
            rows.append(f"# {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                rows.append("\t".join("" if cell is None else str(cell) for cell in row))
        workbook.close()
    except Exception as exc:  # noqa: BLE001
        return _unavailable(item_id, Backend.XLSX, f"unreadable:{type(exc).__name__}")
    text = _clip("\n".join(rows), options)
    if not text.strip():
        return _unavailable(item_id, Backend.XLSX, "workbook_has_no_values")
    return Extraction(item_id=item_id, backend=str(Backend.XLSX), text=text, char_count=len(text))


def _email(item_id: str, path: Path, options: ExtractOptions) -> Extraction:
    if path.suffix.lower() == ".msg":
        try:
            import extract_msg
        except ImportError:
            return _unavailable(item_id, Backend.EMAIL, "backend_unavailable:extract-msg")
        try:
            # Read defensively: extract_msg's stubs type the factory as MSGFile, which
            # declares none of these, and the attribute set has moved between releases.
            with extract_msg.Message(str(path)) as outlook:  # type: ignore[no-untyped-call]
                parts = [
                    f"From: {getattr(outlook, 'sender', '')}",
                    f"To: {getattr(outlook, 'to', '')}",
                    f"Subject: {getattr(outlook, 'subject', '')}",
                    f"Date: {getattr(outlook, 'date', '')}",
                    str(getattr(outlook, "body", "") or ""),
                ]
        except Exception as exc:  # noqa: BLE001
            return _unavailable(item_id, Backend.EMAIL, f"unreadable:{type(exc).__name__}")
    else:
        import email
        from email import policy

        try:
            parsed = email.message_from_bytes(path.read_bytes(), policy=policy.default)
            body = parsed.get_body(preferencelist=("plain", "html"))
            parts = [
                f"From: {parsed.get('From', '')}",
                f"To: {parsed.get('To', '')}",
                f"Subject: {parsed.get('Subject', '')}",
                f"Date: {parsed.get('Date', '')}",
                body.get_content() if body else "",
            ]
        except Exception as exc:  # noqa: BLE001
            return _unavailable(item_id, Backend.EMAIL, f"unreadable:{type(exc).__name__}")

    text = _clip("\n".join(str(part) for part in parts), options)
    if not text.strip():
        return _unavailable(item_id, Backend.EMAIL, "message_has_no_text")
    return Extraction(item_id=item_id, backend=str(Backend.EMAIL), text=text, char_count=len(text))


_BACKENDS = {
    Backend.PDF_TEXT: _pdf_text,
    Backend.PDF_OCR: _pdf_ocr,
    Backend.DOCX: _docx,
    Backend.XLSX: _xlsx,
    Backend.EMAIL: _email,
    Backend.IMAGE_OCR: _image_ocr,
}


def extract(
    *,
    item_id: str,
    path: Path,
    media_type: str,
    backend: Backend = Backend.AUTO_CHAIN,
    options: ExtractOptions | None = None,
) -> Extraction:
    options = options or ExtractOptions()

    if backend is not Backend.AUTO_CHAIN:
        return _BACKENDS[backend](item_id, path, options)

    chain = _CHAIN.get(media_type)
    if chain is None:
        return _unavailable(item_id, Backend.AUTO_CHAIN, f"unsupported_media_type:{media_type}")

    last = _unavailable(item_id, Backend.AUTO_CHAIN, "no_backend_attempted")
    for candidate in chain:
        last = _BACKENDS[candidate](item_id, path, options)
        if last.usable:
            return last
    return last
