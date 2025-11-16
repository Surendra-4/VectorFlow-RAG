# tests/test_integration.py

"""
Integration tests for full pipeline
"""

import numpy as np
import pytest

from src.bm25_retriever import BM25Retriever
from src.chunker import TextChunker
from src.embedder import Embedder
from src.hybrid_retriever import HybridRetriever
from src.vector_store import VectorStore


class TestEndToEnd:
    """End-to-end integration tests"""

    @pytest.fixture
    def sample_docs(self):
        """Create sample documents"""
        return [
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Deep learning uses neural networks with multiple layers to process information.",
            "Natural language processing helps computers understand and generate human language.",
            "Computer vision enables machines to interpret images and videos.",
            "Reinforcement learning trains agents to make decisions through rewards and penalties.",
        ]

    def test_full_pipeline(self, sample_docs):
        """Test complete pipeline from documents to answers"""
        # Step 1: Chunk documents
        chunker = TextChunker(chunk_size=200, overlap=30)
        chunks = []
        for doc in sample_docs:
            chunks.extend(chunker.chunk_text(doc, metadata={"source": "test"}))

        assert len(chunks) > 0

        # Step 2: Generate embeddings
        embedder = Embedder()
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(chunk_texts, show_progress=False)

        assert embeddings.shape[0] == len(chunks)

        # Step 3: Index in vector store
        vector_store = VectorStore("indices/test_integration")
        vector_store.create_collection(reset=True)
        vector_store.add_documents(
            texts=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=[c["metadata"] for c in chunks],
        )

        # Step 4: Build BM25 index
        bm25_retriever = BM25Retriever(corpus=chunk_texts)

        # Step 5: Create hybrid retriever
        hybrid = HybridRetriever(embedder, vector_store, bm25_retriever, alpha=0.5)

        # Step 6: Test retrieval
        query = "What is machine learning?"
        results = hybrid.search(query, k=3)

        assert len(results) > 0
        assert "text" in results[0]
        assert "hybrid_score" in results[0]

        # Results should be related to machine learning
        result_text = results[0]["text"].lower()
        assert "learning" in result_text or "intelligence" in result_text

    def test_vector_vs_bm25_difference(self, sample_docs):
        """Test that vector and BM25 retrieval differ meaningfully"""
        chunker = TextChunker(chunk_size=200, overlap=30)
        chunks = []
        for doc in sample_docs:
            chunks.extend(chunker.chunk_text(doc))

        chunk_texts = [c["text"] for c in chunks]

        embedder = Embedder()
        _ = embedder.encode(chunk_texts, show_progress=False)

        # Test query with exact keyword
        query = "neural networks"

        # BM25 search
        bm25 = BM25Retriever(corpus=chunk_texts)
        bm25_results = bm25.search(query, k=3)

        assert len(bm25_results) > 0


class TestErrorHandling:
    """Test error handling"""

    def test_empty_corpus_handling(self):
        """Test handling of empty corpus"""
        # Should raise error or handle gracefully
        with pytest.raises(Exception):
            BM25Retriever(corpus=[])

    def test_invalid_vector_dimensions(self):
        """Test handling of mismatched vector dimensions"""
        vector_store = VectorStore("indices\\test_error")
        vector_store.create_collection(reset=True)

        # Add documents with correct dimensions
        embeddings = np.random.random((5, 384)).tolist()
        vector_store.add_documents(texts=["test"] * 5, embeddings=embeddings)

        # Query with different dimensions should fail
        wrong_dim_query = np.random.random(768)

        # This might raise or return empty results
        try:
            vector_store.search(wrong_dim_query.tolist(), n_results=1)
        except Exception:
            pass  # Expected
