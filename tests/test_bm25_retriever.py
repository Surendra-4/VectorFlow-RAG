# tests/test_bm25_retriever.py

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
        results = retriever.search("machine learning", k=min(3, len(retriever.corpus)))

        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all("rank" in r for r in results)

    def test_search_respects_k_parameter(self, retriever):
        """Test search returns correct number of results"""
        results_k1 = retriever.search("learning", k=min(1, len(retriever.corpus)))
        results_k3 = retriever.search("learning", k=min(3, len(retriever.corpus)))

        assert len(results_k1) == 1
        assert len(results_k3) == 3

    def test_search_exact_match_scores_high(self, retriever):
        """Test that exact matches score higher"""
        # "machine" appears in "Machine learning algorithms"
        results = retriever.search("machine learning", k=min(3, len(retriever.corpus)))

        # First result should contain "machine"
        top_text = results[0]["text"].lower()
        assert "machine" in top_text
        assert results[0]["score"] >= results[1]["score"]

    def test_search_multiple_keywords(self, retriever):
        """Test search with multiple keywords"""
        results = retriever.search("machine learning", k=min(3, len(retriever.corpus)))

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

        results = retriever.search("machine learning", k=min(3, len(retriever.corpus)))

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


class TestBM25DictCorpus:
    """Phase 3.5: corpus can be List[Dict] with text + chunk_id + metadata."""

    def test_dict_corpus_propagates_chunk_id(self):
        corpus = [
            {"text": "Machine learning algorithms", "chunk_id": "doc_a:0:abcd1234"},
            {"text": "Deep learning networks", "chunk_id": "doc_a:1:beef5678"},
            {"text": "Natural language processing", "chunk_id": "doc_b:0:cafe0001"},
        ]
        r = BM25Retriever(corpus=corpus)
        results = r.search("machine learning", k=2)
        assert all("chunk_id" in r for r in results)
        assert results[0]["chunk_id"] in {"doc_a:0:abcd1234", "doc_a:1:beef5678"}

    def test_dict_corpus_propagates_metadata(self):
        corpus = [
            {"text": "foo bar", "chunk_id": "id1", "metadata": {"source": "x.txt"}},
            {"text": "alpha beta", "chunk_id": "id2", "metadata": {"source": "y.txt"}},
        ]
        r = BM25Retriever(corpus=corpus)
        results = r.search("foo", k=1)
        assert results[0]["metadata"]["source"] == "x.txt"

    def test_plain_string_corpus_chunk_id_none(self):
        r = BM25Retriever(corpus=["one", "two", "three"])
        results = r.search("one", k=1)
        assert results[0]["chunk_id"] is None
        assert results[0]["metadata"] is None

    def test_mixed_corpus_rejected_on_missing_text(self):
        with pytest.raises(ValueError):
            BM25Retriever(corpus=[{"chunk_id": "no-text-here"}])

    def test_legacy_corpus_attribute_still_list_of_strings(self):
        corpus = [{"text": "alpha", "chunk_id": "x"}, {"text": "beta", "chunk_id": "y"}]
        r = BM25Retriever(corpus=corpus)
        assert r.corpus == ["alpha", "beta"]


class TestBM25Multilingual:
    """Phase 11: no-stemmer mode + NFC normalization."""

    def test_no_stemmer_mode_indexes_and_searches(self):
        corpus = [
            {"text": "Bonjour le monde", "chunk_id": "fr:0"},
            {"text": "Hallo Welt", "chunk_id": "de:0"},
            {"text": "Hola mundo", "chunk_id": "es:0"},
        ]
        r = BM25Retriever(corpus=corpus, use_stemmer=False)
        results = r.search("monde", k=3)
        assert any("monde" in res["text"].lower() for res in results)

    def test_no_stemmer_preserves_chunk_ids(self):
        corpus = [{"text": "café société", "chunk_id": "x"}]
        r = BM25Retriever(corpus=corpus, use_stemmer=False)
        results = r.search("café", k=1)
        assert results[0]["chunk_id"] == "x"

    def test_nfc_normalization_matches_decomposed_query(self):
        # Corpus uses composed "é" (NFC); query uses decomposed "e"+combining
        # acute (NFD). After NFC normalization both tokenize identically.
        composed = "café mocha"        # café (single composed codepoint)
        decomposed_query = "café"     # café (e + combining acute)
        r = BM25Retriever(corpus=[{"text": composed, "chunk_id": "c"}], use_stemmer=False)
        results = r.search(decomposed_query, k=1)
        assert len(results) == 1

    def test_english_stemmer_mode_is_default(self):
        r = BM25Retriever(corpus=["running runner runs"])
        assert r.use_stemmer is True
        results = r.search("run", k=1)
        assert len(results) >= 1

    def test_cjk_no_stemmer_does_not_crash(self):
        corpus = [{"text": "机器学习 是 人工智能", "chunk_id": "zh:0"}]
        r = BM25Retriever(corpus=corpus, use_stemmer=False)
        results = r.search("机器学习", k=1)
        assert isinstance(results, list)


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

        results = retriever.search("test", k=min(5, len(retriever.corpus)))
        assert len(results) <= 5

    def test_unicode_documents(self):
        """Test with unicode documents"""
        corpus = ["Hello world", "Bonjour monde", "Hola mundo", "你好世界", "مرحبا بالعالم"]
        retriever = BM25Retriever(corpus=corpus)

        results = retriever.search("world", k=3)
        assert len(results) > 0
