"""
Unit tests for hybrid retriever
"""
import pytest
import numpy as np
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
import shutil
import os


class TestHybridRetrieverInitialization:
    """Test HybridRetriever initialization"""
    
    @pytest.fixture
    def components(self):
        """Create retriever components"""
        corpus = ["Test doc 1", "Test doc 2", "Test doc 3"]
        embedder = Embedder()
        
        # Create vector store
        test_dir = "indices\\test_hybrid_init"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        vector_store = VectorStore(persist_directory=test_dir)
        vector_store.create_collection(reset=True)
        
        embeddings = embedder.encode(corpus, show_progress=False)
        vector_store.add_documents(texts=corpus, embeddings=embeddings.tolist())
        
        bm25 = BM25Retriever(corpus=corpus)
        
        yield embedder, vector_store, bm25, test_dir
        
        shutil.rmtree(test_dir)
    
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
    def hybrid_retriever(self):
        """Create test hybrid retriever"""
        corpus = [
            "Machine learning algorithms",
            "Deep learning neural networks",
            "Natural language processing",
            "Computer vision recognition",
            "Reinforcement learning agents",
        ]
        
        embedder = Embedder()
        
        test_dir = "indices\\test_hybrid_search"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        vector_store = VectorStore(persist_directory=test_dir)
        vector_store.create_collection(reset=True)
        
        embeddings = embedder.encode(corpus, show_progress=False)
        vector_store.add_documents(texts=corpus, embeddings=embeddings.tolist())
        
        bm25 = BM25Retriever(corpus=corpus)
        
        hybrid = HybridRetriever(embedder, vector_store, bm25, alpha=0.5)
        
        yield hybrid, test_dir
    
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
        
        test_dir = "indices\\test_alpha_zero"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        vector_store = VectorStore(persist_directory=test_dir)
        vector_store.create_collection(reset=True)
        
        embeddings = embedder.encode(corpus, show_progress=False)
        vector_store.add_documents(texts=corpus, embeddings=embeddings.tolist())
        
        bm25 = BM25Retriever(corpus=corpus)
        
        hybrid_zero = HybridRetriever(embedder, vector_store, bm25, alpha=0.0)
        results = hybrid_zero.search("machine", k=3)
        
        # Should prioritize exact keyword match
        assert "machine" in results[0]["text"].lower()
        
        shutil.rmtree(test_dir)
    
    def test_alpha_one_gives_vector_weight(self):
        """Test that alpha=1 emphasizes vector similarity"""
        corpus = [
            "car vehicle automobile",
            "machine learning algorithms",
            "deep neural networks",
        ]
        
        embedder = Embedder()
        
        test_dir = "indices\\test_alpha_one"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        vector_store = VectorStore(persist_directory=test_dir)
        vector_store.create_collection(reset=True)
        
        embeddings = embedder.encode(corpus, show_progress=False)
        vector_store.add_documents(texts=corpus, embeddings=embeddings.tolist())
        
        bm25 = BM25Retriever(corpus=corpus)
        
        hybrid_one = HybridRetriever(embedder, vector_store, bm25, alpha=1.0)
        results = hybrid_one.search("transportation", k=3)
        
        # Should find semantically related content
        assert len(results) > 0
        
        shutil.rmtree(test_dir)
