"""
Unit tests for embedder module
"""

import sys
import numpy as np
import pytest

from src.embedder import Embedder


class TestEmbedder:
    """Test suite for Embedder class"""

    @pytest.fixture
    def embedder(self):
        """Create embedder instance"""
        return Embedder(model_name="all-MiniLM-L6-v2")

    def test_embedder_initialization(self, embedder):
        """Test embedder initializes correctly"""
        assert embedder is not None
        assert embedder.dimension == 384
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_encode_single_string(self, embedder):
        """Test encoding single string"""
        text = "Hello world"
        embedding = embedder.encode(text, show_progress=False)

        assert embedding.shape == (1, 384)
        assert np.all(np.isfinite(embedding))
        assert np.allclose(np.linalg.norm(embedding[0]), 1.0)  # Should be normalized

    def test_encode_multiple_strings(self, embedder):
        """Test encoding multiple strings"""
        texts = ["Hello", "World", "Test"]
        embeddings = embedder.encode(texts, show_progress=False)

        assert embeddings.shape == (3, 384)
        assert np.all(np.isfinite(embeddings))

    def test_embed_empty_string(self, embedder):
        """Test encoding empty string"""
        # Should not crash, even with empty input
        embedding = embedder.encode("", show_progress=False)
        assert embedding.shape == (1, 384)

    def test_embedding_normalization(self, embedder):
        """Test embeddings are normalized"""
        texts = ["Normalize me", "Check this", "Verify that"]
        embeddings = embedder.encode(texts, show_progress=False)

        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert np.isclose(norm, 1.0), f"Expected norm ~1.0, got {norm}"

    def test_semantic_similarity(self, embedder):
        """Test semantic similarity between related texts"""
        similar_texts = ["machine learning", "deep learning"]
        different_texts = ["machine learning", "cooking recipes"]

        similar_embs = embedder.encode(similar_texts, show_progress=False)
        different_embs = embedder.encode(different_texts, show_progress=False)

        similar_sim = np.dot(similar_embs[0], similar_embs[1])
        different_sim = np.dot(different_embs[0], different_embs[1])

        # Similar texts should have higher similarity
        assert similar_sim > different_sim
        assert similar_sim > 0.5  # ML and deep learning should be quite similar


class TestEmbedderPerformance:
    """Performance tests for embedder"""

    def test_batch_encoding_speed(self):
        """Test batch encoding is reasonably fast"""
        embedder = Embedder(model_name="all-MiniLM-L6-v2")

        # Generate test data
        texts = ["Sample text {}".format(i) for i in range(100)]

        import time

        start = time.time()
        embeddings = embedder.encode(texts, show_progress=False)
        elapsed = time.time() - start

        print(f"\nEncoded 100 texts in {elapsed:.2f}s")
        assert elapsed < 30  # Should be fast on CPU
        assert embeddings.shape == (100, 384)

    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        embedder = Embedder(model_name="all-MiniLM-L6-v2")

        # Should not consume excessive memory
        import sys

        initial_size = sys.getsizeof(embedder)
        assert initial_size < 500_000_000  # Less than 500MB
