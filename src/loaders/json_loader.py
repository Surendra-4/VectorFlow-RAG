# src/loaders/json_loader.py

"""JSON loader — produces one page of pretty-printed canonical JSON."""

from __future__ import annotations

import json
from pathlib import Path

from src.loaders.base import BaseLoader, LoadedDocument, LoadedPage, LoaderError


class JSONLoader(BaseLoader):
    name = "json"
    extensions = (".json",)
    mime_types = ("application/json",)

    def load(self, path: Path) -> LoadedDocument:
        path = self._resolve(path)
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise LoaderError(f"Could not read {path}: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LoaderError(f"Invalid JSON in {path}: {exc}") from exc

        # Canonical pretty-print: sorted keys, indent=2. This gives stable
        # text that hashes consistently across cosmetically-different input.
        pretty = json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)

        meta = self._base_page_metadata(path)
        meta["json_type"] = type(parsed).__name__  # dict/list/str/...

        return LoadedDocument(
            source_path=str(path),
            document_name=path.name,
            mime_type=self._mime_for(path),
            pages=[LoadedPage(text=pretty, page_number=None, metadata=meta)],
        )
