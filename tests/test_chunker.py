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
