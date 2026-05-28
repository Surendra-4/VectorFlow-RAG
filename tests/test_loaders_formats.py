# tests/test_loaders_formats.py

"""
Per-format loader tests.

Fixtures are generated at runtime inside the test's tmp_path so we
don't commit binary files. Each format gets a happy-path test, a
provenance test, and a malformed-input test where applicable.

OCR tests are skipped automatically if the ``tesseract`` binary
isn't on PATH.
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from src.loaders.base import LoaderError
from src.loaders.csv_loader import CSVLoader
from src.loaders.docx_loader import DOCXLoader
from src.loaders.image_ocr_loader import ImageOCRLoader
from src.loaders.json_loader import JSONLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.sqlite_loader import SQLiteLoader
from src.loaders.text import MarkdownLoader, TextLoader
from src.loaders.xlsx_loader import XLSXLoader

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


# --------------------------------------------------------------------------- #
# Helpers: fixture generators
# --------------------------------------------------------------------------- #


def _write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _make_pdf(path: Path, pages_text: list[str]) -> Path:
    """Generate a simple multi-page PDF using pypdf's writer."""
    from pypdf import PdfWriter
    from pypdf.generic import RectangleObject
    # The writer alone can't emit text content streams trivially without
    # a font dependency; we use reportlab if available, else fall back
    # to building a minimal PDF by embedding text via the dictionary API.
    # For tests, the simplest path is to use reportlab-style PDF creation
    # via a tiny manual PDF. We'll generate via fpdf2 if installed, else
    # use a fallback that writes blank pages whose text() we control by
    # using the standard 14 fonts directly with reportlab-equivalent ops.
    # To keep the test suite light, build text via the standard fonts
    # using pdfrw/reportlab-replacement: just construct via the writer
    # adding one BlankPage per text and stuffing /Contents with a text stream.
    writer = PdfWriter()
    for text in pages_text:
        # Build a content stream that draws ``text`` near the top-left.
        # Tj operator on standard font Helvetica; pypdf can render and
        # later extract it.
        from pypdf.generic import (
            ArrayObject,
            ContentStream,
            DecodedStreamObject,
            DictionaryObject,
            FloatObject,
            NameObject,
            NumberObject,
            TextStringObject,
        )

        # Add a page with a simple font setup.
        page = writer.add_blank_page(width=612, height=792)  # US Letter
        # Build font resource
        font_dict = DictionaryObject()
        font_dict[NameObject("/Type")] = NameObject("/Font")
        font_dict[NameObject("/Subtype")] = NameObject("/Type1")
        font_dict[NameObject("/Name")] = NameObject("/F1")
        font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")

        resources = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_dict
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources

        # Escape parentheses + backslashes for PDF strings; ASCII subset only
        # for the standard 14 fonts.
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream_text = f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET"
        content = DecodedStreamObject()
        content.set_data(stream_text.encode("latin-1"))
        page[NameObject("/Contents")] = content

    with open(path, "wb") as f:
        writer.write(f)
    return path


def _make_docx(path: Path, paragraphs: list[str]) -> Path:
    import docx
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(str(path))
    return path


def _make_xlsx(path: Path, sheets: dict[str, list[list]]) -> Path:
    from openpyxl import Workbook
    wb = Workbook()
    # Remove the default sheet so naming is predictable.
    default = wb.active
    wb.remove(default)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    wb.save(str(path))
    return path


def _make_sqlite(path: Path, schema: dict[str, list[tuple]]) -> Path:
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()
    for table, rows in schema.items():
        # Infer 2 columns from the first row.
        if not rows:
            continue
        col_count = len(rows[0])
        cols = ", ".join(f"col{i} TEXT" for i in range(col_count))
        cursor.execute(f"CREATE TABLE {table} ({cols})")
        placeholders = ", ".join("?" for _ in range(col_count))
        cursor.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
    conn.commit()
    conn.close()
    return path


def _make_image(path: Path, text: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (640, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 36
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 70), text, fill=(0, 0, 0), font=font)
    img.save(str(path))
    return path


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #


class TestTextLoader:
    def test_loads_plain_text(self, tmp_path):
        p = _write_text(tmp_path / "doc.txt", "hello world")
        doc = TextLoader().load(p)
        assert doc.document_name == "doc.txt"
        assert len(doc.pages) == 1
        assert doc.pages[0].text == "hello world"
        assert doc.pages[0].page_number is None

    def test_provenance_fields(self, tmp_path):
        p = _write_text(tmp_path / "x.txt", "abc")
        doc = TextLoader().load(p)
        meta = doc.pages[0].metadata
        assert meta["document_name"] == "x.txt"
        assert meta["source_path"] == str(p)
        assert meta["loader"] == "TextLoader"
        assert meta["page_number"] is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TextLoader().load(tmp_path / "no.txt")

    def test_unicode_round_trip(self, tmp_path):
        p = _write_text(tmp_path / "u.txt", "café — 世界 — مرحبا")
        doc = TextLoader().load(p)
        assert "世界" in doc.pages[0].text

    def test_can_load_for_log_files(self, tmp_path):
        p = _write_text(tmp_path / "app.log", "log line")
        assert TextLoader().can_load(p) is True


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


class TestMarkdownLoader:
    def test_loads_markdown(self, tmp_path):
        p = _write_text(tmp_path / "readme.md", "# Title\n\nBody paragraph.")
        doc = MarkdownLoader().load(p)
        assert "# Title" in doc.pages[0].text
        assert doc.mime_type == "text/markdown"

    def test_can_load_md_extensions(self, tmp_path):
        for ext in (".md", ".markdown", ".mdown"):
            p = _write_text(tmp_path / f"f{ext}", "x")
            assert MarkdownLoader().can_load(p) is True


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


class TestJSONLoader:
    def test_loads_object(self, tmp_path):
        p = _write_text(tmp_path / "data.json", json.dumps({"b": 2, "a": 1}))
        doc = JSONLoader().load(p)
        # Output is pretty-printed and sorted-keys for stable hashing.
        assert "\"a\": 1" in doc.pages[0].text
        assert doc.pages[0].metadata["json_type"] == "dict"

    def test_loads_array(self, tmp_path):
        p = _write_text(tmp_path / "data.json", json.dumps([1, 2, 3]))
        doc = JSONLoader().load(p)
        assert doc.pages[0].metadata["json_type"] == "list"

    def test_invalid_json_raises(self, tmp_path):
        p = _write_text(tmp_path / "bad.json", "{not valid")
        with pytest.raises(LoaderError):
            JSONLoader().load(p)

    def test_sorted_keys_for_stable_output(self, tmp_path):
        # Two semantically-equal JSONs with reordered keys should
        # produce byte-identical loader output.
        p1 = _write_text(tmp_path / "a.json", json.dumps({"x": 1, "y": 2}))
        p2 = _write_text(tmp_path / "b.json", json.dumps({"y": 2, "x": 1}))
        t1 = JSONLoader().load(p1).pages[0].text
        t2 = JSONLoader().load(p2).pages[0].text
        assert t1 == t2


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


class TestCSVLoader:
    def test_loads_basic_csv(self, tmp_path):
        p = tmp_path / "t.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["col1", "col2"])
            w.writerow(["a", "1"])
            w.writerow(["b", "2"])
        doc = CSVLoader().load(p)
        assert "col1 | col2" in doc.pages[0].text
        assert "a | 1" in doc.pages[0].text

    def test_row_and_column_counts(self, tmp_path):
        p = tmp_path / "t.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "c"])
            w.writerow(["1", "2", "3"])
            w.writerow(["4", "5", "6"])
        doc = CSVLoader().load(p)
        meta = doc.pages[0].metadata
        assert meta["row_count"] == 3
        assert meta["column_count"] == 3

    def test_empty_csv_returns_empty_doc(self, tmp_path):
        p = _write_text(tmp_path / "empty.csv", "")
        doc = CSVLoader().load(p)
        assert doc.pages[0].text == ""

    def test_tsv_extension_accepted(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\t2\n", encoding="utf-8")
        assert CSVLoader().can_load(p) is True


# --------------------------------------------------------------------------- #
# XLSX
# --------------------------------------------------------------------------- #


class TestXLSXLoader:
    def test_loads_single_sheet(self, tmp_path):
        p = _make_xlsx(
            tmp_path / "book.xlsx",
            {"Sheet1": [["a", "b"], [1, 2], [3, 4]]},
        )
        doc = XLSXLoader().load(p)
        assert len(doc.pages) == 1
        assert "a | b" in doc.pages[0].text
        assert "1 | 2" in doc.pages[0].text
        assert doc.pages[0].metadata["sheet_name"] == "Sheet1"
        assert doc.pages[0].page_number == 1

    def test_multi_sheet_produces_multiple_pages(self, tmp_path):
        p = _make_xlsx(
            tmp_path / "multi.xlsx",
            {
                "Alpha": [["x", "y"], [1, 2]],
                "Beta":  [["m"], [9]],
            },
        )
        doc = XLSXLoader().load(p)
        assert len(doc.pages) == 2
        names = {pg.metadata["sheet_name"] for pg in doc.pages}
        assert names == {"Alpha", "Beta"}

    def test_skips_blank_rows(self, tmp_path):
        p = _make_xlsx(
            tmp_path / "blank.xlsx",
            {"S": [["a"], [None], ["b"]]},
        )
        doc = XLSXLoader().load(p)
        # Blank row dropped — row_count counts non-empty rows only.
        assert doc.pages[0].metadata["row_count"] == 2


# --------------------------------------------------------------------------- #
# SQLite
# --------------------------------------------------------------------------- #


class TestSQLiteLoader:
    def test_loads_tables_as_pages(self, tmp_path):
        p = _make_sqlite(
            tmp_path / "data.db",
            {
                "users": [("alice", "1"), ("bob", "2")],
                "books": [("a book", "5"), ("another", "6")],
            },
        )
        doc = SQLiteLoader().load(p)
        assert len(doc.pages) == 2
        names = {pg.metadata["table_name"] for pg in doc.pages}
        assert names == {"users", "books"}

    def test_schema_included_in_page_text(self, tmp_path):
        p = _make_sqlite(tmp_path / "x.db", {"t": [("foo", "bar")]})
        doc = SQLiteLoader().load(p)
        assert "TABLE t" in doc.pages[0].text
        assert "COLUMNS:" in doc.pages[0].text

    def test_empty_database_returns_no_pages(self, tmp_path):
        p = tmp_path / "empty.db"
        sqlite3.connect(str(p)).close()
        doc = SQLiteLoader().load(p)
        assert doc.pages == []

    def test_invalid_file_raises(self, tmp_path):
        bad = _write_text(tmp_path / "fake.db", "not a sqlite database")
        with pytest.raises(LoaderError):
            SQLiteLoader().load(bad)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


class TestPDFLoader:
    def test_loads_multi_page_pdf(self, tmp_path):
        p = _make_pdf(tmp_path / "doc.pdf", ["Page one text", "Page two text"])
        doc = PDFLoader().load(p)
        assert len(doc.pages) == 2
        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2

    def test_pdf_extracts_text(self, tmp_path):
        p = _make_pdf(tmp_path / "doc.pdf", ["Hello PDF World"])
        doc = PDFLoader().load(p)
        # The standard 14 fonts emit recoverable text via pypdf.
        text = doc.pages[0].text
        assert "Hello" in text or "PDF" in text, f"got: {text!r}"

    def test_corrupted_pdf_raises(self, tmp_path):
        bad = _write_text(tmp_path / "not.pdf", "not a real PDF")
        with pytest.raises(LoaderError):
            PDFLoader().load(bad)


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


class TestDOCXLoader:
    def test_loads_paragraphs(self, tmp_path):
        p = _make_docx(tmp_path / "doc.docx", ["First paragraph", "Second paragraph"])
        doc = DOCXLoader().load(p)
        assert "First paragraph" in doc.pages[0].text
        assert "Second paragraph" in doc.pages[0].text

    def test_single_page(self, tmp_path):
        p = _make_docx(tmp_path / "x.docx", ["hi"])
        doc = DOCXLoader().load(p)
        # DOCX has no reliable page boundary → single page, None page_number.
        assert len(doc.pages) == 1
        assert doc.pages[0].page_number is None

    def test_corrupted_docx_raises(self, tmp_path):
        bad = _write_text(tmp_path / "bad.docx", "not a docx")
        with pytest.raises(LoaderError):
            DOCXLoader().load(bad)


# --------------------------------------------------------------------------- #
# Image OCR
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract binary not installed")
class TestImageOCRLoader:
    def test_extracts_text_from_image(self, tmp_path):
        p = _make_image(tmp_path / "img.png", "HELLO OCR")
        doc = ImageOCRLoader().load(p)
        text = doc.pages[0].text.upper()
        assert "HELLO" in text or "OCR" in text, f"got: {text!r}"

    def test_provenance_includes_ocr_engine(self, tmp_path):
        p = _make_image(tmp_path / "img2.png", "ABCDEF")
        doc = ImageOCRLoader().load(p)
        assert doc.pages[0].metadata["ocr_engine"] == "tesseract"

    def test_default_lang_is_eng(self):
        assert ImageOCRLoader().lang == "eng"

    def test_explicit_lang_recorded_in_metadata(self, tmp_path):
        p = _make_image(tmp_path / "img3.png", "HELLO")
        # eng is always installed with tesseract; multi-lang spec falls back
        # gracefully if a pack is missing (logged, not raised).
        doc = ImageOCRLoader(lang="eng").load(p)
        assert doc.pages[0].metadata["ocr_lang"] == "eng"
        assert doc.metadata["ocr_lang"] == "eng"

    def test_missing_lang_pack_degrades_gracefully(self, tmp_path):
        # A bogus language code triggers a TesseractError internally; the
        # loader must NOT raise and must NOT auto-download — it returns
        # empty text with a logged warning.
        p = _make_image(tmp_path / "img4.png", "HELLO")
        doc = ImageOCRLoader(lang="zzz_nonexistent").load(p)
        # No exception; page exists with (likely empty) text.
        assert len(doc.pages) == 1
        assert doc.metadata["ocr_lang"] == "zzz_nonexistent"


class TestImageOCRMissingBinary:
    def test_missing_binary_raises_helpful_error(self, tmp_path, monkeypatch):
        # Force shutil.which to return None, simulating no tesseract installed.
        monkeypatch.setattr(
            "src.loaders.image_ocr_loader.shutil.which",
            lambda name: None,
        )
        # We need an actual image file so the path-resolve step succeeds.
        from PIL import Image
        p = tmp_path / "x.png"
        Image.new("RGB", (10, 10)).save(str(p))

        with pytest.raises(LoaderError) as exc_info:
            ImageOCRLoader().load(p)
        assert "tesseract" in str(exc_info.value).lower()
