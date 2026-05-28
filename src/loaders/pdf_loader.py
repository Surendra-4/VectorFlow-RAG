# src/loaders/pdf_loader.py

"""PDF loader — one page per PDF page, via ``pypdf``."""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.loaders.base import BaseLoader, LoadedDocument, LoadedPage, LoaderError
from src.logging_setup import get_logger

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """
    PDF text extraction via pypdf.

    Each PDF page becomes one ``LoadedPage`` with the 1-indexed
    ``page_number``. Pages that yield no extractable text (image-only
    pages) are kept as empty placeholders so chunk indices reflect the
    true page count — but they're filtered out of the chunked output
    by the chunker (which returns zero chunks for empty text).

    Image-only PDFs (scans) are out of scope for this loader; the
    image-OCR loader handles bitmaps directly. A future enhancement
    can detect empty PDFs and route them through OCR per-page.
    """

    name = "pdf"
    extensions = (".pdf",)
    mime_types = ("application/pdf",)

    def load(self, path: Path) -> LoadedDocument:
        path = self._resolve(path)

        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise LoaderError(f"pypdf is required to load {path}") from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise LoaderError(f"Could not open PDF {path}: {exc}") from exc

        pages: List[LoadedPage] = []
        doc_metadata = {}
        try:
            info = reader.metadata
            if info:
                for key in ("/Title", "/Author", "/Subject", "/Creator", "/Producer"):
                    value = info.get(key)
                    if value:
                        doc_metadata[key.lstrip("/").lower()] = str(value)
        except Exception as exc:  # pragma: no cover - some PDFs trip metadata
            logger.debug("Could not read PDF metadata for %s: %s", path, exc)

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Failed to extract page %d of %s: %s", page_idx, path, exc)
                text = ""
            meta = self._base_page_metadata(path, page_number=page_idx)
            pages.append(LoadedPage(text=text, page_number=page_idx, metadata=meta))

        return LoadedDocument(
            source_path=str(path),
            document_name=path.name,
            mime_type=self._mime_for(path) or self.mime_types[0],
            pages=pages,
            metadata={"page_count": len(pages), **doc_metadata},
        )
