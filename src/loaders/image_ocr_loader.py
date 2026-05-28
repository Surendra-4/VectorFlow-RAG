# src/loaders/image_ocr_loader.py

"""
Image OCR loader using Tesseract via ``pytesseract``.

Requires:

* ``pytesseract`` (Python package) — declared in requirements.
* The ``tesseract`` system binary on ``PATH``. If missing, ``load()``
  raises :class:`LoaderError` with install instructions for common
  operating systems.

The image is wrapped as a single-page document. Page numbers are
``None`` (images don't paginate); a multi-image batch ingest treats
each file as its own document. Multi-page TIFFs are handled by
pytesseract internally — each TIFF page becomes its own
``LoadedPage`` so downstream provenance still works.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from src.loaders.base import BaseLoader, LoadedDocument, LoadedPage, LoaderError
from src.logging_setup import get_logger

logger = get_logger(__name__)


_TESSERACT_INSTALL_HINT = (
    "tesseract binary not found on PATH. Install it with:\n"
    "  macOS:  brew install tesseract\n"
    "  Ubuntu: sudo apt-get install tesseract-ocr\n"
    "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
)


class ImageOCRLoader(BaseLoader):
    name = "image_ocr"
    extensions = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")
    mime_types = ("image/png", "image/jpeg", "image/tiff", "image/bmp", "image/webp")

    def __init__(self, lang: Optional[str] = None):
        """
        Args:
            lang: Tesseract language spec, e.g. "eng", "fra", "eng+fra".
                  ``None`` → read ``settings.ingestion.ocr_lang`` (default "eng").
                  Language data files (tessdata) are operator-installed; this
                  loader NEVER auto-downloads them. A missing pack surfaces as
                  a per-page warning and empty text, not a download.
        """
        if lang is not None:
            self.lang = lang
        else:
            from src.config import get_settings

            self.lang = get_settings().ingestion.ocr_lang

    def load(self, path: Path) -> LoadedDocument:
        path = self._resolve(path)

        if shutil.which("tesseract") is None:
            raise LoaderError(_TESSERACT_INSTALL_HINT)

        try:
            import pytesseract
            from PIL import Image, ImageSequence
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise LoaderError(
                f"pytesseract and Pillow are required to OCR {path}"
            ) from exc

        try:
            img = Image.open(str(path))
        except Exception as exc:
            raise LoaderError(f"Could not open image {path}: {exc}") from exc

        pages: List[LoadedPage] = []
        try:
            # Multi-page TIFFs use ImageSequence; single-page formats
            # just iterate once over the only frame.
            for page_idx, frame in enumerate(ImageSequence.Iterator(img), start=1):
                try:
                    text = pytesseract.image_to_string(frame, lang=self.lang) or ""
                except pytesseract.TesseractError as exc:
                    # Most commonly a missing language pack. Log and continue
                    # with empty text — we never auto-install tessdata.
                    logger.warning(
                        "OCR failed on page %d of %s (lang=%s): %s. "
                        "Ensure the tessdata language pack is installed.",
                        page_idx, path, self.lang, exc,
                    )
                    text = ""
                except Exception as exc:
                    logger.warning("OCR failed on page %d of %s: %s", page_idx, path, exc)
                    text = ""

                meta = self._base_page_metadata(path, page_number=page_idx if page_idx > 0 else None)
                meta["ocr_engine"] = "tesseract"
                meta["ocr_lang"] = self.lang
                pages.append(LoadedPage(text=text, page_number=page_idx, metadata=meta))
        finally:
            img.close()

        # If only one page came out, drop the page_number to be consistent
        # with other single-page loaders.
        if len(pages) == 1:
            single = pages[0]
            pages = [LoadedPage(text=single.text, page_number=None,
                                metadata={**single.metadata, "page_number": None})]

        return LoadedDocument(
            source_path=str(path),
            document_name=path.name,
            mime_type=self._mime_for(path) or self.mime_types[0],
            pages=pages,
            metadata={"page_count": len(pages), "ocr_engine": "tesseract", "ocr_lang": self.lang},
        )
