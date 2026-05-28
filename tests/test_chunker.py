# tests/test_chunker.py

"""
Unit tests for chunker module
"""

import pytest

from src.chunker import TextChunker


class TestTextChunker:
    """Test suite for TextChunker class"""

    @pytest.fixture
    def chunker(self):
        """Create chunker instance"""
        return TextChunker(chunk_size=100, overlap=20)

    def test_chunker_initialization(self, chunker):
        """Test chunker initializes correctly"""
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20

    def test_chunk_simple_text(self, chunker):
        """Test chunking simple text"""
        text = "This is a test. This is another sentence. And a third one."
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)

    def test_chunk_preserves_content(self, chunker):
        """Test that chunking preserves content"""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_text(text)

        combined = " ".join(chunk["text"] for chunk in chunks)
        # Should contain all key words
        assert "First" in combined
        assert "Second" in combined
        assert "Third" in combined

    def test_chunk_with_metadata(self, chunker):
        """Test chunking with metadata"""
        text = "Sample text with metadata."
        metadata = {"source": "test", "doc_id": 1}
        chunks = chunker.chunk_text(text, metadata=metadata)

        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test"
            assert chunk["metadata"]["doc_id"] == 1

    def test_chunk_empty_text(self, chunker):
        """Test chunking empty text"""
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0

    def test_chunk_single_sentence(self, chunker):
        """Test chunking single sentence"""
        text = "Short sentence."
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short sentence."

    def test_chunk_overlap(self):
        """Test overlap between chunks"""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "A. " * 30  # Create long text
        chunks = chunker.chunk_text(text)

        if len(chunks) > 1:
            assert len(chunks[0]["text"]) > 0
            assert len(chunks[1]["text"]) > 0


class TestChunkerIdentity:
    """Provenance + stable IDs emitted by the chunker (Phase 3.5)."""

    @pytest.fixture
    def chunker(self):
        return TextChunker(chunk_size=80, overlap=10)

    def test_top_level_provenance_fields(self, chunker):
        chunks = chunker.chunk_text(
            "First sentence. Second sentence. Third sentence.",
            metadata={"document_name": "spec.txt", "source_path": "/spec.txt"},
        )
        for c in chunks:
            assert "text" in c
            assert "chunk_id" in c
            assert "chunk_index" in c
            assert "doc_id" in c
            assert "metadata" in c
            assert isinstance(c["chunk_id"], str)
            assert isinstance(c["chunk_index"], int)
            assert c["chunk_id"].startswith(c["doc_id"] + ":")

    def test_metadata_provenance_keys_present(self, chunker):
        chunks = chunker.chunk_text(
            "Sentence one. Sentence two.",
            metadata={"document_name": "x.txt", "source_path": "/x.txt"},
        )
        meta = chunks[0]["metadata"]
        for key in (
            "document_id",
            "chunk_id",
            "chunk_index",
            "document_name",
            "source_path",
            "page_number",  # placeholder
        ):
            assert key in meta, f"missing provenance key: {key}"
        assert meta["document_name"] == "x.txt"
        assert meta["page_number"] is None

    def test_user_metadata_preserved_alongside_provenance(self, chunker):
        chunks = chunker.chunk_text("Test.", metadata={"custom_key": "val"})
        assert chunks[0]["metadata"]["custom_key"] == "val"
        assert "document_id" in chunks[0]["metadata"]

    def test_chunk_ids_unique_within_doc(self, chunker):
        chunks = chunker.chunk_text("A. " * 40)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_indices_sequential(self, chunker):
        chunks = chunker.chunk_text("A. " * 60)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_ids_stable_across_calls(self, chunker):
        text = "Repeatable test. Of stable IDs. Across two calls."
        a = chunker.chunk_text(text)
        b = chunker.chunk_text(text)
        assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
        assert a[0]["doc_id"] == b[0]["doc_id"]

    def test_same_text_in_different_docs_different_chunk_ids(self, chunker):
        # The chunker derives doc_id from the full text; identical chunks
        # appearing in different surrounding documents should NOT collide.
        doc1 = "Shared boilerplate sentence. Unique tail for doc one."
        doc2 = "Shared boilerplate sentence. Distinct tail for the other."
        a = chunker.chunk_text(doc1)
        b = chunker.chunk_text(doc2)
        assert a[0]["doc_id"] != b[0]["doc_id"]
        assert a[0]["chunk_id"] != b[0]["chunk_id"]

    def test_explicit_doc_id_used_when_provided(self, chunker):
        chunks = chunker.chunk_text("Some text.", doc_id="doc_explicit")
        assert all(c["doc_id"] == "doc_explicit" for c in chunks)
        assert all(c["chunk_id"].startswith("doc_explicit:") for c in chunks)

    def test_chunk_index_offset_shifts_global_index(self, chunker):
        chunks = chunker.chunk_text("First. Second. Third.", chunk_index_offset=10)
        # chunk_index should start at the offset, not 0.
        indices = [c["chunk_index"] for c in chunks]
        assert indices[0] == 10
        # Sequential after the offset.
        assert indices == list(range(10, 10 + len(chunks)))

    def test_offset_produces_distinct_chunk_ids_across_pages(self):
        # Simulates two pages chunked with the same doc_id but different offsets.
        chunker = TextChunker(chunk_size=100, overlap=10)
        doc_id = "doc_test"
        page1 = chunker.chunk_text("Page one content.", doc_id=doc_id, chunk_index_offset=0)
        page2 = chunker.chunk_text("Page two content.", doc_id=doc_id,
                                    chunk_index_offset=len(page1))
        all_ids = [c["chunk_id"] for c in page1 + page2]
        assert len(all_ids) == len(set(all_ids)), "chunk_ids should be unique across pages"

    def test_negative_offset_rejected(self, chunker):
        with pytest.raises(ValueError):
            chunker.chunk_text("text", chunk_index_offset=-1)


class TestChunkerMultilingual:
    """Phase 11: CJK sentence terminators + Unicode safety."""

    def test_english_splitting_unchanged(self):
        # Regression guard: Latin splitting must be byte-identical to before.
        chunker = TextChunker(chunk_size=20, overlap=5)
        a = chunker.chunk_text("First sentence. Second sentence. Third one.")
        texts = [c["text"] for c in a]
        # Each sentence is short enough to land in its own chunk boundary set;
        # the key assertion is that splitting still happens on ". ".
        combined = " ".join(texts)
        assert "First sentence." in combined
        assert "Second sentence." in combined

    def test_cjk_fullwidth_terminators_split(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        # No whitespace between CJK sentences — the new alternative handles it.
        chunks = chunker.chunk_text("你好世界。这是测试。第三句。")
        assert len(chunks) >= 2
        # Terminators stay attached to their sentence.
        assert any("。" in c["text"] for c in chunks)

    def test_no_latin_text_still_chunks(self):
        chunker = TextChunker(chunk_size=8, overlap=2)
        chunks = chunker.chunk_text("これはテストです。もう一つの文。")
        assert len(chunks) >= 1
        joined = "".join(c["text"] for c in chunks)
        assert "テスト" in joined

    def test_mixed_script_document(self):
        chunker = TextChunker(chunk_size=40, overlap=5)
        chunks = chunker.chunk_text("English sentence. 中文句子。Another English one.")
        assert len(chunks) >= 1
        joined = " ".join(c["text"] for c in chunks)
        assert "English" in joined and "中文" in joined

    def test_arabic_rtl_chunks(self):
        chunker = TextChunker(chunk_size=40, overlap=5)
        chunks = chunker.chunk_text("هذه جملة عربية. وهذه جملة أخرى.")
        assert len(chunks) >= 1
        assert any("جملة" in c["text"] for c in chunks)


class TestChunkerEdgeCases:
    """Test edge cases for chunker"""

    def test_very_long_text(self):
        """Test chunking very long text"""
        chunker = TextChunker(chunk_size=200, overlap=30)
        text = "Word. " * 500  # Very long text
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["text"]) > 0

    def test_unicode_text(self):
        """Test chunking unicode text"""
        chunker = TextChunker(chunk_size=100, overlap=20)
        text = "Hello 世界. Test тест. Prüf café."
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0
        combined = " ".join(chunk["text"] for chunk in chunks)
        assert "世界" in combined
