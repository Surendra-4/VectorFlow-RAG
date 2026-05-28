# src/loaders/sqlite_loader.py

"""SQLite loader — one page per table (schema + rows)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

from src.loaders.base import BaseLoader, LoadedDocument, LoadedPage, LoaderError


def _format_row(values: tuple) -> str:
    return " | ".join("" if v is None else str(v).strip() for v in values)


class SQLiteLoader(BaseLoader):
    """
    Load an SQLite database, producing one page per table.

    Each page contains the table's schema (column names + types) followed
    by all rows formatted as ``col | col | col``.

    Limits: a configurable per-table row cap prevents huge tables from
    blowing up memory. The default (10000 rows) is conservative; users
    with large databases should pre-extract specific tables.
    """

    name = "sqlite"
    extensions = (".db", ".sqlite", ".sqlite3")
    mime_types = ("application/x-sqlite3",)

    MAX_ROWS_PER_TABLE = 10_000

    def load(self, path: Path) -> LoadedDocument:
        path = self._resolve(path)

        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise LoaderError(f"Could not open sqlite {path}: {exc}") from exc

        pages: List[LoadedPage] = []
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            for page_idx, table_name in enumerate(tables, start=1):
                # Pull schema (column names + types) for the header line.
                cursor.execute(f"PRAGMA table_info({table_name})")
                col_info = cursor.fetchall()
                col_names = [c[1] for c in col_info]
                col_types = [c[2] or "ANY" for c in col_info]
                header = (
                    f"TABLE {table_name}\n"
                    f"COLUMNS: " + ", ".join(f"{n} {t}" for n, t in zip(col_names, col_types))
                )

                # Pull rows up to the cap.
                cursor.execute(f"SELECT * FROM {table_name} LIMIT ?", (self.MAX_ROWS_PER_TABLE,))
                rows = cursor.fetchall()
                body = "\n".join(_format_row(row) for row in rows)

                page_text = header + ("\n\n" + body if body else "")

                meta = self._base_page_metadata(path, page_number=page_idx)
                meta["table_name"] = table_name
                meta["row_count"] = len(rows)
                meta["column_count"] = len(col_names)
                pages.append(LoadedPage(text=page_text, page_number=page_idx, metadata=meta))
        except sqlite3.Error as exc:
            raise LoaderError(f"Error reading sqlite {path}: {exc}") from exc
        finally:
            conn.close()

        return LoadedDocument(
            source_path=str(path),
            document_name=path.name,
            mime_type=self._mime_for(path) or self.mime_types[0],
            pages=pages,
            metadata={"table_count": len(pages)},
        )
