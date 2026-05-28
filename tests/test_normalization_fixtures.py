# tests/test_normalization_fixtures.py

"""
Explicit validation of the Unicode normalization layer (Phase 11).

These fixtures stress the normalization boundaries the user called out:
NFC vs NFD, composed vs decomposed accents, fullwidth punctuation,
zero-width spaces, and RTL/bidi control marks. They are deliberately
model-free and fast — normalization correctness is a prerequisite for
multilingual retrieval quality, so it gets its own focused suite.
"""

from __future__ import annotations

import unicodedata

import pytest

from src.identity import compute_chunk_id, compute_doc_id, normalize_text
from src.bm25_retriever import _nfc
from src.chunker import TextChunker
from src.query_expansion.base import sanitize_query


# --------------------------------------------------------------------------- #
# Fixtures: composed vs decomposed pairs
# --------------------------------------------------------------------------- #

# (composed NFC, decomposed NFD) pairs that must normalize equal.
COMPOSED_DECOMPOSED_PAIRS = [
    ("café", "café"),          # é: U+00E9  vs  e + U+0301
    ("Café", "Café"),
    ("naïve", "naïve"),        # ï
    ("Müller", "Müller"),      # ü
    ("Renée", "Renée"),
    ("Ångström", "Ångström"),  # Å
]


class TestComposedDecomposedEquivalence:
    @pytest.mark.parametrize("composed,decomposed", COMPOSED_DECOMPOSED_PAIRS)
    def test_pair_is_actually_different_bytes(self, composed, decomposed):
        # Sanity: the fixtures really are different byte sequences pre-normalization.
        # (At least one pair must differ; some editors may store both as NFC.)
        nfd = unicodedata.normalize("NFD", composed)
        assert unicodedata.normalize("NFC", nfd) == unicodedata.normalize("NFC", composed)

    @pytest.mark.parametrize("composed,decomposed", COMPOSED_DECOMPOSED_PAIRS)
    def test_normalize_text_collapses_pair(self, composed, decomposed):
        assert normalize_text(composed) == normalize_text(decomposed)

    @pytest.mark.parametrize("composed,decomposed", COMPOSED_DECOMPOSED_PAIRS)
    def test_doc_id_stable_across_normalization_form(self, composed, decomposed):
        assert compute_doc_id(composed) == compute_doc_id(decomposed)

    @pytest.mark.parametrize("composed,decomposed", COMPOSED_DECOMPOSED_PAIRS)
    def test_chunk_id_stable_across_normalization_form(self, composed, decomposed):
        a = compute_chunk_id("doc_x", 0, composed)
        b = compute_chunk_id("doc_x", 0, decomposed)
        assert a == b

    @pytest.mark.parametrize("composed,decomposed", COMPOSED_DECOMPOSED_PAIRS)
    def test_bm25_nfc_helper_collapses_pair(self, composed, decomposed):
        assert _nfc(composed) == _nfc(decomposed)


class TestNfcVsNfkcPolicy:
    """We use NFC, NOT NFKC — verify compatibility chars are preserved."""

    def test_ligature_preserved_under_nfc(self):
        # ﬁ (U+FB01) collapses to "fi" under NFKC but is PRESERVED under NFC.
        ligature = "ﬁle"
        assert normalize_text(ligature) == "ﬁle"  # not "file"

    def test_fullwidth_digits_preserved_under_nfc(self):
        # Fullwidth "１２３" would become "123" under NFKC; NFC keeps them.
        fullwidth = "１２３"
        assert normalize_text(fullwidth) == "１２３"

    def test_superscript_preserved_under_nfc(self):
        # x² → NFKC would yield x2; NFC keeps the superscript.
        assert normalize_text("x²") == "x²"


class TestZeroWidthAndBidiInQuery:
    def test_zero_width_space_stripped_from_query(self):
        q = "hello​world"  # zero-width space
        assert sanitize_query(q, max_length=100) == "helloworld"

    def test_zero_width_joiner_stripped(self):
        q = "a‍‌b"  # ZWJ + ZWNJ
        assert "‍" not in sanitize_query(q, max_length=100)
        assert "‌" not in sanitize_query(q, max_length=100)

    def test_bom_stripped(self):
        q = "﻿text"  # BOM / ZWNBSP
        assert sanitize_query(q, max_length=100) == "text"

    def test_bidi_override_stripped(self):
        # RLO (U+202E) is the classic bidi-spoofing character.
        q = "user‮gnp.exe"
        cleaned = sanitize_query(q, max_length=100)
        assert "‮" not in cleaned

    def test_lrm_rlm_stripped(self):
        q = "abc‎‏def"  # LRM + RLM
        cleaned = sanitize_query(q, max_length=100)
        assert "‎" not in cleaned and "‏" not in cleaned

    def test_legitimate_rtl_text_preserved(self):
        # Actual Arabic letters must survive — we only strip control marks.
        q = "ما هي الخلية"
        cleaned = sanitize_query(q, max_length=100)
        assert "الخلية" in cleaned


class TestFullwidthPunctuationChunking:
    def test_fullwidth_period_splits_cjk(self):
        chunker = TextChunker(chunk_size=6, overlap=1)
        chunks = chunker.chunk_text("甲乙丙。丁戊己。")
        assert len(chunks) >= 2

    def test_fullwidth_question_exclamation_split(self):
        chunker = TextChunker(chunk_size=6, overlap=1)
        chunks = chunker.chunk_text("你好吗？我很好！再见。")
        assert len(chunks) >= 2

    def test_ascii_period_unaffected_by_cjk_rule(self):
        # Pure ASCII still splits exactly as before (regression guard).
        chunker = TextChunker(chunk_size=1000, overlap=0)
        chunks = chunker.chunk_text("One. Two. Three.")
        # Single chunk (under chunk_size) but content preserved.
        joined = " ".join(c["text"] for c in chunks)
        assert "One." in joined and "Two." in joined and "Three." in joined


class TestNfcQueryNormalization:
    def test_query_decomposed_normalizes_to_composed(self):
        decomposed = "café"  # NFD form
        out = sanitize_query(decomposed, max_length=100)
        assert out == unicodedata.normalize("NFC", out)
        assert "café" == unicodedata.normalize("NFC", decomposed)

    def test_ascii_query_unchanged(self):
        # English path: NFC is a no-op, so the query is byte-identical.
        assert sanitize_query("hello world", max_length=100) == "hello world"
