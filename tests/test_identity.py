# tests/test_identity.py

"""Tests for src.identity — pure functions, fast, no heavy deps."""

from __future__ import annotations

import pytest

from src.identity import (
    ID_SCHEMA_VERSION,
    compute_chunk_id,
    compute_doc_id,
    normalize_text,
    parse_chunk_id,
)


# --------------------------------------------------------------------------- #
# normalize_text
# --------------------------------------------------------------------------- #


class TestNormalizeText:
    def test_empty(self):
        assert normalize_text("") == ""

    def test_none_handled(self):
        assert normalize_text(None) == ""  # type: ignore[arg-type]

    def test_strips_outer_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_runs(self):
        assert normalize_text("a  b\t\nc") == "a b c"

    def test_unicode_nfc_normalization(self):
        # Two encodings of ä — composed (NFC) vs decomposed (NFD).
        composed = "ä"          # ä
        decomposed = "ä"        # a + combining diaeresis
        assert composed != decomposed
        assert normalize_text(composed) == normalize_text(decomposed)

    def test_case_preserved(self):
        # We deliberately do not lowercase — case can be semantic.
        assert normalize_text("Foo") != normalize_text("foo")

    def test_idempotent(self):
        once = normalize_text("  Foo  bar  ")
        twice = normalize_text(once)
        assert once == twice


# --------------------------------------------------------------------------- #
# compute_doc_id
# --------------------------------------------------------------------------- #


class TestDocId:
    def test_deterministic(self):
        assert compute_doc_id("hello world") == compute_doc_id("hello world")

    def test_different_content_different_id(self):
        assert compute_doc_id("hello") != compute_doc_id("world")

    def test_format(self):
        doc_id = compute_doc_id("foo")
        assert doc_id.startswith("doc_")
        # 16-hex suffix
        suffix = doc_id.removeprefix("doc_")
        assert len(suffix) == 16
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_whitespace_insensitive(self):
        # Same content; differing whitespace should still hash equal.
        assert compute_doc_id("hello   world") == compute_doc_id("hello world")
        assert compute_doc_id("hello world\n") == compute_doc_id("  hello world  ")

    def test_unicode_form_insensitive(self):
        composed = "Café"          # Café (NFC)
        decomposed = "Café"        # Cafe + combining acute (NFD)
        assert compute_doc_id(composed) == compute_doc_id(decomposed)

    def test_case_sensitive(self):
        assert compute_doc_id("Foo") != compute_doc_id("foo")

    def test_empty_content(self):
        # Empty content gets a well-defined id (no error).
        empty = compute_doc_id("")
        assert empty.startswith("doc_")

    def test_custom_prefix(self):
        assert compute_doc_id("foo", prefix="d").startswith("d_")

    def test_id_schema_version_constant_present(self):
        # Stable surface that callers can pin against.
        assert ID_SCHEMA_VERSION == 1


# --------------------------------------------------------------------------- #
# compute_chunk_id
# --------------------------------------------------------------------------- #


class TestChunkId:
    def test_deterministic(self):
        a = compute_chunk_id("doc_abc", 0, "hello")
        b = compute_chunk_id("doc_abc", 0, "hello")
        assert a == b

    def test_format(self):
        cid = compute_chunk_id("doc_abc", 5, "hello")
        parts = cid.split(":")
        assert len(parts) == 3
        assert parts[0] == "doc_abc"
        assert parts[1] == "5"
        assert len(parts[2]) == 8  # 8-hex chunk text hash

    def test_different_doc_id_different_chunk_id(self):
        # Same chunk content, different doc → different chunk_id.
        # This is the key property for source attribution.
        a = compute_chunk_id("doc_one", 0, "shared boilerplate")
        b = compute_chunk_id("doc_two", 0, "shared boilerplate")
        assert a != b

    def test_different_index_different_chunk_id(self):
        a = compute_chunk_id("doc_x", 0, "same text")
        b = compute_chunk_id("doc_x", 1, "same text")
        assert a != b

    def test_different_text_different_chunk_id(self):
        a = compute_chunk_id("doc_x", 0, "alpha")
        b = compute_chunk_id("doc_x", 0, "beta")
        assert a != b

    def test_whitespace_variation_collapses(self):
        a = compute_chunk_id("doc_x", 0, "hello  world")
        b = compute_chunk_id("doc_x", 0, "hello world")
        assert a == b

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError):
            compute_chunk_id("doc_x", -1, "text")


# --------------------------------------------------------------------------- #
# parse_chunk_id
# --------------------------------------------------------------------------- #


class TestParseChunkId:
    def test_roundtrip(self):
        cid = compute_chunk_id("doc_abc123", 7, "some text")
        doc_id, idx, text_hash = parse_chunk_id(cid)
        assert doc_id == "doc_abc123"
        assert idx == 7
        assert len(text_hash) == 8

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_chunk_id("not-a-chunk-id")

    def test_non_int_index_raises(self):
        with pytest.raises(ValueError):
            parse_chunk_id("doc_abc:notanint:abcd1234")

    def test_doc_id_with_colon_safe(self):
        # We use rsplit(":", 2), so a doc_id that itself contains ":" still parses correctly.
        cid = "weird:doc:id:3:abcd1234"
        doc_id, idx, text_hash = parse_chunk_id(cid)
        assert doc_id == "weird:doc:id"
        assert idx == 3
        assert text_hash == "abcd1234"
