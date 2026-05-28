# src/query_expansion/base.py

"""Data model + strategy protocol + sanitization utilities."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

# Control characters that can confuse downstream prompt construction
# (newlines preserved — prompts use them intentionally).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Zero-width and bidi-control characters that are invisible but can corrupt
# hashing, display, and prompt construction. Stripped during sanitization.
#   200B ZERO WIDTH SPACE, 200C/200D ZWNJ/ZWJ, FEFF BOM/ZWNBSP,
#   200E/200F LRM/RLM, 202A-202E bidi embeddings/overrides, 2066-2069 isolates.
_ZERO_WIDTH_AND_BIDI = re.compile(
    "[​‌‍﻿‎‏‪-‮⁦-⁩]"
)

# Multiple consecutive whitespace runs (including embedded newlines if any
# survive sanitization) collapsed to single space — keeps prompts compact.
_WS_RUN = re.compile(r"\s+")


def sanitize_query(query: str, *, max_length: int) -> str:
    """
    Defensive cleanup applied before any LLM call.

    * Unicode NFC normalization (composed form) — multilingual-safe and a
      no-op for ASCII, so the English path is byte-identical.
    * Strips ASCII control characters and zero-width / bidi-control marks
      that have no legitimate place in a retrieval query.
    * Collapses whitespace runs.
    * Truncates to ``max_length`` characters (after normalization, so the
      cap counts visible characters consistently).

    This is a defense-in-depth layer — adversarial input via this path
    cannot grow expansion prompts unboundedly, smuggle in control
    sequences, or hide bidi-override attacks.
    """
    if query is None:
        return ""
    cleaned = unicodedata.normalize("NFC", query)
    cleaned = _CONTROL_CHARS.sub("", cleaned)
    cleaned = _ZERO_WIDTH_AND_BIDI.sub("", cleaned)
    cleaned = _WS_RUN.sub(" ", cleaned).strip()
    if max_length is not None and len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """Order-preserving dedup. Empties dropped. Case-sensitive."""
    seen = set()
    out: List[str] = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


@dataclass(frozen=True)
class ExpansionResult:
    """Output of a single expansion strategy."""

    strategy: str
    queries: tuple = ()              # tuple for hashability (Phase 6 cache keys)
    hyde_documents: tuple = ()
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ExpandedQuery:
    """
    The full set of retrieval inputs derived from one user query.

    ``queries`` always begins with the sanitized original query, followed
    by any LLM-generated variants. Empty strings and duplicates are
    removed. ``hyde_documents`` may be empty when HyDE wasn't run.
    """

    original: str
    queries: tuple
    hyde_documents: tuple = ()
    strategies_used: tuple = ()
    errors: tuple = ()
    expansion_latency_ms: float = 0.0

    @property
    def variant_count(self) -> int:
        """Number of generated variants (excluding the original)."""
        return max(0, len(self.queries) - 1)


@runtime_checkable
class ExpansionStrategy(Protocol):
    """Structural contract for an expansion strategy."""

    name: str

    def expand(self, query: str) -> ExpansionResult: ...
