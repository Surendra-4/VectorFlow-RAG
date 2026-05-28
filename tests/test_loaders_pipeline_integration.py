# tests/test_loaders_pipeline_integration.py

"""
End-to-end loader → pipeline integration tests (Phase 4).

Covers:

* heterogeneous mixed-format batch ingestion
* per-file failure tolerance (one bad file doesn't kill the batch)
* max-file-size guard
* provenance propagation from loader → retrieval result
* page-number flow for multi-page PDFs / multi-sheet XLSX
* fail_fast behavior
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from src.config import IngestionSettings, Settings, reset_settings_cache
from src.rag_pipeline import RAGPipeline


# Re-use the fixture generators from test_loaders_formats so we don't
# duplicate PDF/DOCX/XLSX builders.
from tests.test_loaders_formats import (  # type: ignore[import-not-found]
    _make_docx,
    _make_pdf,
    _make_sqlite,
    _make_xlsx,
    _write_text,
)


@pytest.fixture
def mixed_corpus(tmp_path):
    """Generate a heterogeneous folder of files for ingestion tests."""
    files = {
        "notes.txt": _write_text(
            tmp_path / "notes.txt",
            "Photosynthesis converts light to chemical energy in plants.",
        ),
        "readme.md": _write_text(
            tmp_path / "readme.md",
            "# Mitochondria\n\nMitochondria are the powerhouse of the cell.",
        ),
        "data.json": _write_text(
            tmp_path / "data.json",
            json.dumps({"organism": "yeast", "process": "fermentation"}),
        ),
        "table.csv": _make_csv(
            tmp_path / "table.csv",
            [["organism", "atp"], ["yeast", "low"], ["mammal", "high"]],
        ),
        "doc.pdf": _make_pdf(
            tmp_path / "doc.pdf",
            ["Page one about ribosomes.", "Page two about ribosomes too."],
        ),
        "report.docx": _make_docx(
            tmp_path / "report.docx",
            ["DOCX content about chloroplasts."],
        ),
        "book.xlsx": _make_xlsx(
            tmp_path / "book.xlsx",
            {"Sheet1": [["term", "definition"], ["lysosome", "recycler"]]},
        ),
        "store.db": _make_sqlite(
            tmp_path / "store.db",
            {"organelles": [("nucleus", "control"), ("vacuole", "storage")]},
        ),
    }
    return files


def _make_csv(path: Path, rows) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)
    return path


# --------------------------------------------------------------------------- #
# Happy-path mixed ingestion
# --------------------------------------------------------------------------- #


class TestMixedFormatIngestion:
    @pytest.fixture
    def pipeline(self, tmp_path, mixed_corpus):
        rag = RAGPipeline(index_dir=str(tmp_path / "mixed_pipeline"))
        report = rag.ingest_files(list(mixed_corpus.values()))
        return rag, report

    def test_all_files_ingested(self, pipeline, mixed_corpus):
        _, report = pipeline
        assert len(report["successes"]) == len(mixed_corpus)
        assert report["failures"] == []
        assert report["chunks"] > 0

    def test_results_carry_full_provenance(self, pipeline):
        rag, _ = pipeline
        results = rag.search("photosynthesis", k=3)
        # The top result should be from notes.txt.
        top = results[0]
        assert top["document_name"] == "notes.txt"
        assert top["source_path"].endswith("notes.txt")
        assert top["chunk_id"] is not None
        assert top["doc_id"] is not None
        assert top["chunk_index"] == 0

    def test_each_format_returns_distinct_doc_ids(self, pipeline):
        rag, _ = pipeline
        # Sample various queries to verify different files contribute
        # to retrieval — each gets its own doc_id.
        queries = [
            "photosynthesis",
            "mitochondria",
            "ribosomes",
            "chloroplasts",
            "lysosome",
        ]
        doc_ids = set()
        for q in queries:
            for r in rag.search(q, k=2):
                if r.get("doc_id"):
                    doc_ids.add(r["doc_id"])
        assert len(doc_ids) >= 3  # at least 3 different source docs surface


# --------------------------------------------------------------------------- #
# Multi-page provenance flow
# --------------------------------------------------------------------------- #


class TestMultiPageProvenance:
    def test_pdf_pages_have_distinct_page_numbers(self, tmp_path):
        pdf = _make_pdf(
            tmp_path / "multi.pdf",
            [
                "Page one talks about alpha topic.",
                "Page two talks about beta topic.",
                "Page three talks about gamma topic.",
            ],
        )
        rag = RAGPipeline(index_dir=str(tmp_path / "rag"))
        rag.ingest_files([pdf])
        results = rag.search("alpha", k=3)
        page_numbers = {r.get("page_number") for r in results}
        # We expect at least one page_number=1 (the alpha page).
        assert 1 in page_numbers

    def test_xlsx_sheets_have_distinct_page_numbers(self, tmp_path):
        xlsx = _make_xlsx(
            tmp_path / "data.xlsx",
            {
                "Cells":     [["cell", "function"], ["mitochondria", "atp"]],
                "Organelles": [["name", "role"], ["nucleus", "control"]],
            },
        )
        rag = RAGPipeline(index_dir=str(tmp_path / "rag_xlsx"))
        rag.ingest_files([xlsx])
        results = rag.search("mitochondria", k=5)
        page_numbers = {r.get("page_number") for r in results if r.get("page_number")}
        assert page_numbers, f"expected at least one page_number, got {page_numbers}"

    def test_pdf_chunk_indices_unique_across_pages(self, tmp_path):
        pdf = _make_pdf(
            tmp_path / "uniq.pdf",
            [
                "Content of the first page goes here for indexing.",
                "Distinct content of the second page goes here as well.",
            ],
        )
        rag = RAGPipeline(index_dir=str(tmp_path / "uniq_rag"))
        rag.ingest_files([pdf])
        # Query broadly enough to retrieve from both pages.
        results = rag.search("content", k=10)
        chunk_indices = [r["chunk_index"] for r in results]
        # Indices should be unique.
        assert len(chunk_indices) == len(set(chunk_indices))


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


class TestFailureHandling:
    def test_corrupted_file_in_batch_does_not_kill_others(self, tmp_path):
        good = _write_text(tmp_path / "good.txt", "valid text content")
        bad = _write_text(tmp_path / "bad.pdf", "not a real PDF")  # PDFLoader will reject
        rag = RAGPipeline(index_dir=str(tmp_path / "fail_tolerance"))
        report = rag.ingest_files([good, bad])
        assert len(report["successes"]) == 1
        assert len(report["failures"]) == 1
        assert "bad.pdf" in report["failures"][0]["path"]

    def test_missing_file_recorded_as_failure(self, tmp_path):
        rag = RAGPipeline(index_dir=str(tmp_path / "missing"))
        report = rag.ingest_files([tmp_path / "does_not_exist.txt"])
        assert report["successes"] == []
        assert len(report["failures"]) == 1

    def test_unknown_extension_recorded_as_failure(self, tmp_path):
        path = _write_text(tmp_path / "weird.zzz", "?")
        rag = RAGPipeline(index_dir=str(tmp_path / "unknown_ext"))
        report = rag.ingest_files([path])
        assert "No loader" in report["failures"][0]["reason"]

    def test_fail_fast_raises_on_first_error(self, tmp_path):
        bad = _write_text(tmp_path / "bad.json", "{not valid json")
        rag = RAGPipeline(index_dir=str(tmp_path / "fail_fast"))
        with pytest.raises(Exception):
            rag.ingest_files([bad], fail_fast=True)


# --------------------------------------------------------------------------- #
# File-size guard
# --------------------------------------------------------------------------- #


class TestFileSizeGuard:
    def test_oversize_file_rejected(self, tmp_path):
        big = _write_text(tmp_path / "huge.txt", "x" * 2048)
        rag = RAGPipeline(index_dir=str(tmp_path / "size_guard"))
        # Cap at 1 KB; the 2 KB file should fail.
        report = rag.ingest_files([big], max_file_size_bytes=1024)
        assert len(report["failures"]) == 1
        assert "max_file_size_bytes" in report["failures"][0]["reason"]

    def test_under_cap_succeeds(self, tmp_path):
        small = _write_text(tmp_path / "small.txt", "tiny")
        rag = RAGPipeline(index_dir=str(tmp_path / "size_under"))
        report = rag.ingest_files([small], max_file_size_bytes=1024)
        assert len(report["successes"]) == 1


# --------------------------------------------------------------------------- #
# Configuration override
# --------------------------------------------------------------------------- #


class TestIngestionSettings:
    def test_settings_default_max_file_size(self):
        s = Settings()
        assert s.ingestion.max_file_size_bytes == 100 * 1024 * 1024

    def test_settings_env_override(self, monkeypatch):
        monkeypatch.setenv("VFR_INGESTION__MAX_FILE_SIZE_BYTES", "1024")
        reset_settings_cache()
        try:
            s = Settings()
            assert s.ingestion.max_file_size_bytes == 1024
        finally:
            reset_settings_cache()
