# src/identity.py

"""
Canonical document and chunk identity for VectorFlow-RAG.

Provides deterministic, content-derived IDs so that:

* The same source document always yields the same ``doc_id``.
* The same chunk text (after normalization) at the same position in the
  same document always yields the same ``chunk_id``.
* Identical chunk text in *different* documents yields *different*
  ``chunk_id``s — source attribution is baked into the chunk identity.
* Re-ingestion is a no-op for unchanged content (foundation for dedup
  and incremental indexing, which land in a later phase).

The hashes are content-derived, not random. There is no central
allocator and no need for one — IDs can be computed anywhere given the
same inputs and normalization rules.

Normalization (applied before hashing, never to the stored text):

1. Unicode NFC normalize  — collapses equivalent encodings (e.g. ä).
2. Strip leading/trailing whitespace.
3. Collapse runs of whitespace to a single space.

Case is **preserved** — case can be semantically meaningful, and the
risk of false negatives from casing differences is far smaller than
the risk of false positives from casing collisions.

Hash sizes:

* ``doc_id``    = 16 hex chars = 64 bits.  P(collision) ≈ 1.8 × 10⁻¹⁹ at 1M docs.
* chunk text part = 8 hex chars = 32 bits, namespaced under doc_id.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Final

# Public so callers can pin against the schema in their own hashing.
ID_SCHEMA_VERSION: Final[int] = 1

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Normalize ``text`` for hashing.

    The output is *only* used as hash input. The original text is
    preserved by callers for display.
    """
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    normalized = _WHITESPACE_RUN.sub(" ", normalized).strip()
    return normalized


def _sha256_hex(s: str, *, length: int) -> str:
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return digest[:length]


def compute_doc_id(content: str, *, prefix: str = "doc") -> str:
    """
    Stable, content-derived document identifier.

    Returns:
        ``"doc_<16 hex chars>"`` — e.g. ``"doc_3f1a0e9b6c2d4a8e"``.

    Idempotent: same normalized content → same ID, on any machine.
    """
    normalized = normalize_text(content)
    return f"{prefix}_{_sha256_hex(normalized, length=16)}"


def compute_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    """
    Stable identifier for a chunk within a document.

    Format::

        <doc_id>:<chunk_index>:<8-hex-chunk-text-hash>

    Source attribution is baked in via ``doc_id``, so identical chunk
    text in two different documents produces two different ``chunk_id``s.

    The 8-hex chunk-text hash distinguishes chunks whose ``chunk_index``
    collides (e.g. after re-chunking with different parameters).
    """
    if chunk_index < 0:
        raise ValueError(f"chunk_index must be >= 0, got {chunk_index}")
    text_hash = _sha256_hex(normalize_text(chunk_text), length=8)
    return f"{doc_id}:{chunk_index}:{text_hash}"


def parse_chunk_id(chunk_id: str) -> tuple[str, int, str]:
    """
    Reverse of :func:`compute_chunk_id` — extract the components.

    Returns:
        ``(doc_id, chunk_index, text_hash)``

    Raises:
        ValueError: if ``chunk_id`` is not in the expected shape.
    """
    parts = chunk_id.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Malformed chunk_id: {chunk_id!r}")
    doc_id, idx_str, text_hash = parts
    try:
        idx = int(idx_str)
    except ValueError as exc:
        raise ValueError(f"chunk_index not an int in {chunk_id!r}") from exc
    return doc_id, idx, text_hash
