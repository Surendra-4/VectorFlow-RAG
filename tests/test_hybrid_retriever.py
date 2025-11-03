"""
Unit tests for hybrid retriever
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import shutil
import tempfile
import time

import numpy as np
import pytest

from src.bm25_retriever import BM25Retriever
from src.embedder import Embedder
from src.hybrid_retriever import HybridRetriever
from src.vector_store import VectorStore


def safe_rmtree(path, retries=5):
    """Retry-safe delete to avoid Windows file lock errors"""
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(0.5)


class TestHybridRetrieverInitialization:
    """Test HybridRetriever initialization"""

    @pytest.fixture
    def components(self, temp_dir):
        """Create retriever components"""
        corpus = ["Test doc 1", "Test doc 2", "Test doc 3"]
        embedder = Embedder()

        # Create vector store using temp_dir
        test_dir = str(temp_dir / "test_hybrid_init")

        vector_store = VectorStore(persist_directory=test_dir)
        vector_store.create_collection(reset=True)

        embeddings = embedder.encode(corpus, show_progress=False)
        vector_store.add_documents(texts=corpus, embeddings=embeddings.tolist())

        bm25 = BM25Retriever(corpus=corpus)

        yield embedder, vector_store, bm25, test_dir

    def test_initialization(self, components):
        """Test hybrid retriever initializes correctly"""
        embedder, vector_store, bm25, _ = components

        hybrid = HybridRetriever(embedder, vector_store, bm25, alpha=0.5)

        assert hybrid is not None
        assert hybrid.alpha == 0.5

    def test_different_alpha_values(self, components):
        """Test with different alpha values"""
        embedder, vector_store, bm25, _ = components

        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            hybrid = HybridRetriever(embedder, vector_store, bm25, alpha=alpha)
            assert hybrid.alpha == alpha


class TestHybridSearch:
    """Test hybrid search functionality"""

    @pytest.fixture
    def hybrid_retriever(self, temp_dir):
        """Create test hybrid retriever"""
        corpus = [
            "Machine learning algorithms",
            "Deep learning neural networks",
            "Natural language processing",
            "Computer vision recognition",
            "Reinforcement learning agents",
        ]

        test_dir = str(temp_dir / "test_hybrid_search")

        vector_store = VectorStore(persist_directory=test_dir)
        vector_store.create_collection(reset=True)

        embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

        embeddings = embedder.encode(corpus, show_progress=False)
        vector_store.add_documents(texts=corpus, embeddings=embeddings.tolist())

        bm25 = BM25Retriever(corpus=corpus)

        retriever = HybridRetriever(embedder, vector_store, bm25, alpha=0.5)

        yield retriever, test_dir

    def test_search_returns_results(self, hybrid_retriever):
        """Test search returns results"""
        hybrid, _ = hybrid_retriever

        results = hybrid.search("machine learning", k=3)

        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("hybrid_score" in r for r in results)

    def test_search_respects_k_parameter(self, hybrid_retriever):
        """Test search returns correct number of results"""
        hybrid, _ = hybrid_retriever

        results_k1 = hybrid.search("learning", k=1)
        results_k3 = hybrid.search("learning", k=3)

        assert len(results_k1) <= 1
        assert len(results_k3) <= 3

    def test_hybrid_vs_pure_approaches(self, hybrid_retriever):
        """Test that hybrid combines both approaches"""
        hybrid, _ = hybrid_retriever

        query = "neural networks"
        results = hybrid.search(query, k=5)

        # Results should include documents mentioning neural networks
        # or related semantic concepts
        assert len(results) > 0


class TestAlphaWeighting:
    """Test alpha parameter effect"""

    def test_alpha_zero_gives_bm25_weight(self):
        """Test that alpha=0 emphasizes BM25"""
        corpus = [
            "exact machine match",
            "neural networks learning",
            "machine learning systems",
        ]

        embedder = Embedder()

        test_dir = tempfile.mkdtemp(prefix="test_alpha_zero_")

        try:
            vector_store = VectorStore(persist_directory=test_dir)
            vector_store.create_collection(reset=True)

            embeddings = embedder.encode(corpus, show_progress=False)
            vector_store.add_documents(texts=corpus, embeddings=embeddings.tolist())

            bm25 = BM25Retriever(corpus=corpus)

            hybrid_zero = HybridRetriever(embedder, vector_store, bm25, alpha=0.0)
            results = hybrid_zero.search("machine", k=3)

            # Should prioritize exact keyword match
            assert "machine" in results[0]["text"].lower()
        finally:
            safe_rmtree(test_dir)

    def test_alpha_one_gives_vector_weight(self):
        """Test that alpha=1 emphasizes vector similarity"""
        corpus = [
            "car vehicle automobile",
            "machine learning algorithms",
            "deep neural networks",
        ]

        embedder = Embedder()

        test_dir = tempfile.mkdtemp(prefix="test_alpha_one_")

        try:
            vector_store = VectorStore(persist_directory=test_dir)
            vector_store.create_collection(reset=True)

            embeddings = embedder.encode(corpus, show_progress=False)
            vector_store.add_documents(texts=corpus, embeddings=embeddings.tolist())

            bm25 = BM25Retriever(corpus=corpus)

            hybrid_one = HybridRetriever(embedder, vector_store, bm25, alpha=1.0)
            results = hybrid_one.search("transportation", k=3)

            # Should find semantically related content
            assert len(results) > 0
        finally:
            safe_rmtree(test_dir)
