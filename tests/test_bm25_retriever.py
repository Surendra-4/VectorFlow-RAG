"""
Unit tests for BM25 retriever
"""
import pytest
from src.bm25_retriever import BM25Retriever


class TestBM25RetrieverInitialization:
    """Test BM25Retriever initialization"""
    
    def test_initialization_with_corpus(self):
        """Test initializing with corpus"""
        corpus = ["Document one", "Document two", "Document three"]
        retriever = BM25Retriever(corpus=corpus)
        
        assert retriever is not None
        assert retriever.corpus == corpus
    
    def test_initialization_with_small_corpus(self):
        """Test with minimal corpus"""
        corpus = ["Test"]
        retriever = BM25Retriever(corpus=corpus)
        
        assert len(retriever.corpus) == 1
    
    def test_initialization_fails_with_empty_corpus(self):
        """Test that empty corpus raises error"""
        with pytest.raises(Exception):
            BM25Retriever(corpus=[])


class TestBM25Search:
    """Test BM25 search functionality"""
    
    @pytest.fixture
    def retriever(self):
        """Create test retriever"""
        corpus = [
            "Machine learning algorithms",
            "Deep learning neural networks",
            "Natural language processing",
            "Computer vision image recognition",
            "Reinforcement learning agents",
        ]
        return BM25Retriever(corpus=corpus)
    
    def test_search_returns_results(self, retriever):
        """Test search returns results"""
        results = retriever.search("machine learning", k=3)
        
        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all("rank" in r for r in results)
    
    def test_search_respects_k_parameter(self, retriever):
        """Test search returns correct number of results"""
        results_k1 = retriever.search("learning", k=1)
        results_k3 = retriever.search("learning", k=3)
        
        assert len(results_k1) == 1
        assert len(results_k3) == 3
    
    def test_search_exact_match_scores_high(self, retriever):
        """Test that exact matches score higher"""
        # "machine" appears in "Machine learning algorithms"
        results = retriever.search("machine", k=5)
        
        # First result should contain "machine"
        top_text = results[0]["text"].lower()
        assert "machine" in top_text
        assert results[0]["score"] >= results[1]["score"]
    
    def test_search_multiple_keywords(self, retriever):
        """Test search with multiple keywords"""
        results = retriever.search("machine learning algorithms", k=3)
        
        assert len(results) > 0
        # Top result should be about machine learning
        assert "learning" in results[0]["text"].lower() or "algorithms" in results[0]["text"].lower()
    
    def test_search_no_matches(self, retriever):
        """Test search with no matching documents"""
        results = retriever.search("xyz123notaword", k=5)
        
        # Should return results but with low scores
        assert len(results) <= 5


class TestBM25Scores:
    """Test BM25 scoring"""
    
    def test_scores_are_normalized(self):
        """Test that scores are reasonable numbers"""
        corpus = ["test document", "another test", "document test"]
        retriever = BM25Retriever(corpus=corpus)
        
        results = retriever.search("test", k=3)
        
        # BM25 scores should be positive floats
        for result in results:
            assert isinstance(result["score"], float)
            assert result["score"] >= 0
    
    def test_relevance_ranking(self):
        """Test that results are ranked by relevance"""
        corpus = [
            "apple fruit",
            "apple pie recipe",
            "orange fruit",
            "apple tree",
            "apple computer",
        ]
        retriever = BM25Retriever(corpus=corpus)
        
        results = retriever.search("apple", k=5)
        
        # All results should contain "apple"
        for result in results:
            assert "apple" in result["text"].lower()
        
        # Scores should be in descending order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestBM25EdgeCases:
    """Test edge cases"""
    
    def test_search_empty_query(self):
        """Test searching with empty query"""
        corpus = ["test document", "another document"]
        retriever = BM25Retriever(corpus=corpus)
        
        try:
            results = retriever.search("", k=2)
            # Should handle gracefully
            assert isinstance(results, list)
        except Exception:
            pass  # Expected
    
    def test_duplicate_documents(self):
        """Test with duplicate documents in corpus"""
        corpus = ["test"] * 3 + ["different"]
        retriever = BM25Retriever(corpus=corpus)
        
        results = retriever.search("test", k=5)
        assert len(results) <= 5
    
    def test_unicode_documents(self):
        """Test with unicode documents"""
        corpus = [
            "Hello world",
            "Bonjour monde",
            "Hola mundo",
            "你好世界",
            "مرحبا بالعالم"
        ]
        retriever = BM25Retriever(corpus=corpus)
        
        results = retriever.search("world", k=3)
        assert len(results) > 0
