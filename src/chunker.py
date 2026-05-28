# src/chunker.py

"""
Sentence-aware text chunker with overlap and provenance-aware output.

Each emitted chunk carries:

* ``text``         — raw chunk text (display fidelity preserved)
* ``chunk_id``     — stable hash; see :mod:`src.identity`
* ``chunk_index``  — integer position within its source document
* ``doc_id``       — stable document hash
* ``metadata``     — enriched with provenance fields:
                     ``document_id``, ``chunk_id``, ``chunk_index``,
                     ``document_name``, ``source_path``, ``page_number``,
                     plus any caller-supplied keys.

Backward compatibility: tests that only check ``"chunk_id" in chunk``
still pass. The field changes from int to str, but its presence is
preserved.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.config import get_settings
from src.identity import compute_chunk_id, compute_doc_id

# Provenance keys we surface at the top level of metadata. ``None`` is the
# acceptable placeholder value when the caller hasn't supplied something
# yet (e.g. page_number before PDF loaders land).
_PROVENANCE_KEYS = (
    "document_id",
    "chunk_id",
    "chunk_index",
    "document_name",
    "source_path",
    "page_number",
)


class TextChunker:
    """Split text into overlapping sentence-aligned chunks with stable IDs."""

    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        cfg = get_settings().chunker
        self.chunk_size = cfg.chunk_size if chunk_size is None else chunk_size
        self.overlap = cfg.overlap if overlap is None else overlap

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        chunk_index_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Split ``text`` into sentence-aware chunks.

        Args:
            text: source document text
            metadata: caller-supplied metadata; provenance keys
                (``document_name``, ``source_path``, ``page_number``) are
                lifted into the chunk's metadata if present, and any other
                keys are preserved.
            doc_id: explicit document ID. When ``None``, derived from
                ``text``. Passing it lets the caller pre-compute the doc_id
                once per source (useful when the same doc is chunked in
                multiple passes).
            chunk_index_offset: starting value for ``chunk_index`` on the
                first chunk this call emits. Used by callers that chunk a
                document page-by-page (PDF, multi-sheet XLSX) so the
                global ``chunk_index`` within a document stays unique
                across pages. Defaults to ``0`` — single-call usage is
                unaffected.

        Returns:
            A list of chunk dicts. Empty list when ``text`` has no
            non-whitespace content.
        """
        if not text or not text.strip():
            return []
        if chunk_index_offset < 0:
            raise ValueError(f"chunk_index_offset must be >= 0, got {chunk_index_offset}")

        resolved_doc_id = doc_id if doc_id is not None else compute_doc_id(text)
        base_meta = dict(metadata) if metadata else {}

        # Sentence-aware split with overlap accumulation.
        #
        # Two alternatives, by design:
        #   1. (?<=[.!?])\s+   — Latin terminators followed by whitespace.
        #                        Identical to the pre-Phase-11 behavior, so
        #                        English chunk boundaries never shift.
        #   2. (?<=[。！？])\s* — CJK fullwidth terminators, which typically
        #                        have NO trailing whitespace. Only fires after
        #                        a CJK punctuation mark, so Latin text is
        #                        unaffected.
        sentences = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", text)
        # The alternation can yield empty fragments (e.g. trailing terminator);
        # drop them so they don't perturb the accumulation loop.
        sentences = [s for s in sentences if s]

        raw_chunks: List[str] = []
        curr = ""
        for s in sentences:
            if len(curr) + len(s) > self.chunk_size and curr:
                raw_chunks.append(curr.strip())
                curr = curr[-self.overlap :] + " " + s
            else:
                curr += " " + s
        if curr.strip():
            raw_chunks.append(curr.strip())

        chunks: List[Dict[str, Any]] = []
        for local_idx, chunk_text in enumerate(raw_chunks):
            global_idx = chunk_index_offset + local_idx
            cid = compute_chunk_id(resolved_doc_id, global_idx, chunk_text)

            # Build provenance: start with caller's metadata, then write
            # canonical provenance fields. Caller-supplied collisions are
            # overridden — these fields are authoritative.
            chunk_meta = dict(base_meta)
            chunk_meta["document_id"] = resolved_doc_id
            chunk_meta["chunk_id"] = cid
            chunk_meta["chunk_index"] = global_idx
            chunk_meta.setdefault("document_name", base_meta.get("document_name"))
            chunk_meta.setdefault("source_path", base_meta.get("source_path"))
            chunk_meta.setdefault("page_number", base_meta.get("page_number"))

            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_id": cid,
                    "chunk_index": global_idx,
                    "doc_id": resolved_doc_id,
                    "metadata": chunk_meta,
                }
            )

        return chunks


if __name__ == "__main__":
    c = TextChunker(100, 20)
    docs = "This is a test. Second sentence. Third part."
    for chunk in c.chunk_text(docs, metadata={"document_name": "demo.txt"}):
        print(chunk)
