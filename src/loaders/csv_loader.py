# src/loaders/csv_loader.py

"""CSV loader — flattens rows into one text page."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import List

from src.loaders.base import BaseLoader, LoadedDocument, LoadedPage, LoaderError


def _format_row(row: List[str]) -> str:
    """Render one CSV row as a single ` | `-separated line."""
    return " | ".join(cell.strip() for cell in row)


class CSVLoader(BaseLoader):
    name = "csv"
    extensions = (".csv", ".tsv")
    mime_types = ("text/csv",)

    def load(self, path: Path) -> LoadedDocument:
        path = self._resolve(path)
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise LoaderError(f"Could not read {path}: {exc}") from exc

        # Sniff dialect first; fall back to default Excel dialect on failure.
        try:
            dialect = csv.Sniffer().sniff(raw[:4096])
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(StringIO(raw), dialect)
        lines: List[str] = []
        row_count = 0
        header_count = 0
        try:
            for i, row in enumerate(reader):
                if not row:
                    continue
                lines.append(_format_row(row))
                row_count += 1
                if i == 0:
                    header_count = len(row)
        except csv.Error as exc:
            raise LoaderError(f"Malformed CSV in {path}: {exc}") from exc

        text = "\n".join(lines)

        meta = self._base_page_metadata(path)
        meta["row_count"] = row_count
        meta["column_count"] = header_count

        return LoadedDocument(
            source_path=str(path),
            document_name=path.name,
            mime_type=self._mime_for(path),
            pages=[LoadedPage(text=text, page_number=None, metadata=meta)],
            metadata={"row_count": row_count, "column_count": header_count},
        )
