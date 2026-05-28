# src/loaders/base.py

"""
Loader base classes, data model, and the loader protocol.

The data model uses frozen dataclasses for two reasons:

1. Loaders never mutate output post-load — immutability catches
   accidental downstream edits.
2. They're hashable, so caching layers (Phase 6) can use them as
   dict keys safely.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


class LoaderError(IOError):
    """Raised when a loader fails to parse a file.

    Subclass of ``IOError`` so callers using broad ``except IOError``
    catch loader failures alongside missing-file / permission errors,
    while ``isinstance(exc, LoaderError)`` distinguishes the parse case.
    """


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoadedPage:
    """
    A logical page extracted by a loader.

    For single-page formats (TXT, CSV, JSON), there is exactly one
    ``LoadedPage`` and ``page_number`` is ``None`` (no inherent paging).
    For paged formats (PDF, XLSX-sheets, SQLite-tables), each unit
    becomes one ``LoadedPage`` and ``page_number`` is 1-indexed.
    """

    text: str
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedDocument:
    """
    A loaded source document — the unit of work a loader produces.

    ``pages`` is the sequence of content units within the document.
    Per-document metadata (author, created_at, …) lives on ``metadata``;
    per-page extras (sheet_name, table_name, …) live on each page.
    """

    source_path: str
    document_name: str
    mime_type: str
    pages: List[LoadedPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_text(self) -> str:
        """Concatenated text of all pages — used by the pipeline to compute doc_id."""
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def is_empty(self) -> bool:
        return self.total_chars == 0


# --------------------------------------------------------------------------- #
# Protocol + base
# --------------------------------------------------------------------------- #


@runtime_checkable
class LoaderProtocol(Protocol):
    """Structural contract — any duck-typed loader satisfies this."""

    name: str
    extensions: Tuple[str, ...]
    mime_types: Tuple[str, ...]

    def can_load(self, path: Path) -> bool: ...

    def load(self, path: Path) -> LoadedDocument: ...


class BaseLoader:
    """
    Base class for concrete loaders. Subclasses set ``name``,
    ``extensions``, ``mime_types`` and implement :meth:`load`.

    Default :meth:`can_load` matches on extension or MIME type; subclasses
    can override for content-sniffing if needed.
    """

    name: str = "base"
    extensions: Tuple[str, ...] = ()
    mime_types: Tuple[str, ...] = ()

    def can_load(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        if suffix in self.extensions:
            return True
        guessed, _ = mimetypes.guess_type(str(path))
        if guessed and guessed in self.mime_types:
            return True
        return False

    def load(self, path: Path) -> LoadedDocument:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared helpers for subclasses
    # ------------------------------------------------------------------ #

    def _resolve(self, path: Path) -> Path:
        """Normalize to absolute path; raise FileNotFoundError if missing."""
        return Path(path).expanduser().resolve(strict=True)

    def _mime_for(self, path: Path) -> str:
        guessed, _ = mimetypes.guess_type(str(path))
        if guessed:
            return guessed
        # Fall back to the first MIME type the loader claims to handle.
        return self.mime_types[0] if self.mime_types else "application/octet-stream"

    def _base_page_metadata(self, path: Path, *, page_number: Optional[int] = None) -> Dict[str, Any]:
        """Build the canonical per-page metadata dict for this loader/path."""
        return {
            "document_name": path.name,
            "source_path": str(path),
            "page_number": page_number,
            "mime_type": self._mime_for(path),
            "loader": type(self).__name__,
        }
