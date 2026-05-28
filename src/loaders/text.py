# src/loaders/text.py

"""Plain-text loaders: ``.txt`` and ``.md``/``.markdown``."""

from __future__ import annotations

from pathlib import Path

from src.loaders.base import BaseLoader, LoadedDocument, LoadedPage, LoaderError


class TextLoader(BaseLoader):
    name = "text"
    extensions = (".txt", ".log", ".text")
    mime_types = ("text/plain",)

    def load(self, path: Path) -> LoadedDocument:
        path = self._resolve(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise LoaderError(f"Could not read {path}: {exc}") from exc
        return LoadedDocument(
            source_path=str(path),
            document_name=path.name,
            mime_type=self._mime_for(path),
            pages=[
                LoadedPage(
                    text=text,
                    page_number=None,
                    metadata=self._base_page_metadata(path),
                )
            ],
        )


class MarkdownLoader(BaseLoader):
    """
    Loads Markdown as plain text.

    Phase 4 treats Markdown the same way as a text file — the chunker is
    sentence-aware and works well on prose-with-markup. A future phase
    can swap in a structure-aware parser (e.g. one that splits by
    heading) without breaking callers, since the loader contract returns
    a generic ``LoadedDocument``.
    """

    name = "markdown"
    extensions = (".md", ".markdown", ".mdown")
    mime_types = ("text/markdown",)

    def load(self, path: Path) -> LoadedDocument:
        path = self._resolve(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise LoaderError(f"Could not read {path}: {exc}") from exc
        return LoadedDocument(
            source_path=str(path),
            document_name=path.name,
            mime_type=self._mime_for(path) or "text/markdown",
            pages=[
                LoadedPage(
                    text=text,
                    page_number=None,
                    metadata=self._base_page_metadata(path),
                )
            ],
        )
