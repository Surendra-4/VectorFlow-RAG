# src/loaders/registry.py

"""
Loader registry — maps file paths to concrete loaders.

Loader resolution is best-effort:

1. Try every registered loader's :meth:`can_load` in registration order.
2. First match wins.
3. If nothing matches, raise :class:`LoaderError`.

Custom loaders register themselves at import time of any module that
wants to add a new format. The :func:`default_registry` builds a
registry populated with the loaders shipped in this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from src.loaders.base import BaseLoader, LoaderError
from src.logging_setup import get_logger

logger = get_logger(__name__)


class LoaderRegistry:
    """Ordered collection of loaders with first-match dispatch."""

    def __init__(self) -> None:
        self._loaders: List[BaseLoader] = []

    def register(self, loader: BaseLoader) -> None:
        if not isinstance(loader, BaseLoader):
            raise TypeError(f"Registered objects must inherit BaseLoader; got {type(loader).__name__}")
        self._loaders.append(loader)
        logger.debug(
            "Registered loader=%s extensions=%s mimes=%s",
            loader.name, loader.extensions, loader.mime_types,
        )

    def loaders(self) -> Iterator[BaseLoader]:
        return iter(self._loaders)

    def find(self, path: Path) -> BaseLoader:
        """Return the first loader claiming to handle ``path``."""
        for loader in self._loaders:
            try:
                if loader.can_load(path):
                    return loader
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Loader %s.can_load raised: %s", loader.name, exc)
        raise LoaderError(f"No loader registered for {path}")

    def __len__(self) -> int:
        return len(self._loaders)


def default_registry() -> LoaderRegistry:
    """Build the canonical registry with every shipped loader."""
    # Local imports avoid circular deps and keep startup cost low for
    # callers that only want a subset of loaders.
    from src.loaders.text import MarkdownLoader, TextLoader
    from src.loaders.json_loader import JSONLoader
    from src.loaders.csv_loader import CSVLoader
    from src.loaders.xlsx_loader import XLSXLoader
    from src.loaders.sqlite_loader import SQLiteLoader
    from src.loaders.pdf_loader import PDFLoader
    from src.loaders.docx_loader import DOCXLoader
    from src.loaders.image_ocr_loader import ImageOCRLoader

    reg = LoaderRegistry()
    # Order matters: more-specific extensions (e.g. .md) before more-generic
    # ones (e.g. .txt) so MarkdownLoader claims .md first.
    reg.register(MarkdownLoader())
    reg.register(TextLoader())
    reg.register(JSONLoader())
    reg.register(CSVLoader())
    reg.register(XLSXLoader())
    reg.register(SQLiteLoader())
    reg.register(PDFLoader())
    reg.register(DOCXLoader())
    reg.register(ImageOCRLoader())
    return reg
