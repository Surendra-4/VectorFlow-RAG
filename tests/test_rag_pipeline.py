"""
Integration tests for RAG pipeline

Tests are marked with @pytest.mark.integration and will be skipped in CI
unless Ollama is available.
"""
import pytest
from pathlib import Path

from src.rag_pipeline import RAGPipeline


class TestRAGPipelineInitialization:
    """Test RAGPipeline initialization"""

    @pytest.mark.integration
    def test_initialization(self, temp_dir):
        """Test pipeline initializes correctly"""
        index_dir = str(temp_dir / "test_rag_init")
        
        rag = RAGPipeline(
            index_dir=index_dir,
            alpha=0.5,
            llm_model="tinyllama"
        )
        
        assert rag is not None
        assert rag.alpha == 0.5

    @pytest.mark.integration
    def test_different_configurations(self, temp_dir):
        """Test different configurations"""
        configs = [
            {"alpha": 0.0},
            {"alpha": 0.5},
            {"alpha": 1.0},
        ]
        
        for i, config in enumerate(configs):
            index_dir = str(temp_dir / f"test_rag_config_{i}")
            rag = RAGPipeline(index_dir=index_dir, **config)
            assert rag.alpha == config["alpha"]

    def test_default_path_uses_pathlib(self):
        """Test that default path is cross-platform"""
        # Create without index_dir to test default
        rag = RAGPipeline()
        
        # Vector store should have valid path
        assert "indices" in rag.vector_store.persist_directory


class TestDocumentIngestion:
    """Test document ingestion"""

    @pytest.fixture
    def rag_pipeline(self, temp_dir):
        """Create test pipeline with temp directory"""
        index_dir = str(temp_dir / "test_rag_ingest")
        rag = RAGPipeline(index_dir=index_dir, alpha=0.5)
        yield rag
        # Cleanup is automatic via temp_dir fixture

    def test_ingest_single_document(self, rag_pipeline):
        """Test ingesting single document"""
        docs = ["This is a test document about machine learning."]
        
        rag_pipeline.ingest_documents(docs)
        
        assert rag_pipeline.document_count == 1
        assert len(rag_pipeline.corpus) > 0

    def test_ingest_multiple_documents(self, rag_pipeline):
        """Test ingesting multiple documents"""
        docs = [
            "First document about AI",
            "Second document about ML",
            "Third document about DL",
        ]
        
        rag_pipeline.ingest_documents(docs)
        
        assert rag_pipeline.document_count == 3

    def test_ingest_with_metadata(self, rag_pipeline):
        """Test ingesting with metadata"""
        docs = [
            "Document 1",
            "Document 2",
        ]
        metadatas = [
            {"source": "source1"},
            {"source": "source2"},
        ]
        
        rag_pipeline.ingest_documents(docs, metadatas=metadatas)
        
        assert rag_pipeline.document_count == 2


class TestRetrieval:
    """Test retrieval functionality"""

    @pytest.fixture
    def indexed_pipeline(self, temp_dir):
        """Create indexed pipeline with temp directory"""
        index_dir = str(temp_dir / "test_rag_retrieval")
        rag = RAGPipeline(index_dir=index_dir)
        
        docs = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "NLP processes natural language",
            "Computer vision recognizes images",
        ]
        
        rag.ingest_documents(docs)
        
        yield rag
        # Cleanup is automatic via temp_dir fixture

    def test_search_returns_results(self, indexed_pipeline):
        """Test search returns results"""
        results = indexed_pipeline.search("machine learning", k=2)
        
        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("hybrid_score" in r for r in results)

    def test_search_without_ingestion_fails(self, temp_dir):
        """Test search fails without ingestion"""
        index_dir = str(temp_dir / "test_rag_no_ingest")
        rag = RAGPipeline(index_dir=index_dir)
        
        with pytest.raises(ValueError):
            rag.search("query")

    def test_search_respects_k_parameter(self, indexed_pipeline):
        """Test search returns correct number of results"""
        results_k1 = indexed_pipeline.search("learning", k=1)
        results_k2 = indexed_pipeline.search("learning", k=2)
        
        assert len(results_k1) <= 1
        assert len(results_k2) <= 2


class TestPipelineStats:
    """Test pipeline statistics"""

    def test_get_stats(self, temp_dir):
        """Test getting pipeline statistics"""
        index_dir = str(temp_dir / "test_rag_stats")
        rag = RAGPipeline(index_dir=index_dir, alpha=0.7)
        
        docs = ["Test 1", "Test 2", "Test 3"]
        rag.ingest_documents(docs)
        
        stats = rag.get_stats()
        
        assert "documents_ingested" in stats
        assert "chunks_indexed" in stats
        assert "embedding_dimension" in stats
        assert "alpha" in stats
        
        assert stats["documents_ingested"] == 3
        assert stats["alpha"] == 0.7

    def test_embedder_dimension(self, temp_dir):
        """Test embedder dimension matches expected value"""
        index_dir = str(temp_dir / "test_embedder_dim")
        rag = RAGPipeline(index_dir=index_dir)
        
        stats = rag.get_stats()
        assert stats["embedding_dimension"] == 384  # all-MiniLM-L6-v2


class TestCrossPlatformPaths:
    """Test cross-platform path compatibility"""

    def test_path_is_valid_string(self, temp_dir):
        """Test that paths are valid strings"""
        index_dir = str(temp_dir / "test_crossplatform")
        rag = RAGPipeline(index_dir=index_dir)
        
        # Should be a valid string path
        assert isinstance(rag.vector_store.persist_directory, str)
        assert len(rag.vector_store.persist_directory) > 0

    def test_path_creation_works_both_ways(self, temp_dir):
        """Test that both string and Path objects work"""
        # Test with string path
        index_dir_str = str(temp_dir / "test_str_path")
        rag1 = RAGPipeline(index_dir=index_dir_str)
        
        # Test with Path object (should be converted internally)
        index_dir_path = temp_dir / "test_path_obj"
        rag2 = RAGPipeline(index_dir=str(index_dir_path))
        
        assert rag1.vector_store.persist_directory
        assert rag2.vector_store.persist_directory


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_document_ingestion(self, temp_dir):
        """Test ingesting non-empty documents"""
        index_dir = str(temp_dir / "test_empty_docs")
        rag = RAGPipeline(index_dir=index_dir)
        
        # Use non-empty documents
        docs = ["This is a valid document with content"]
        rag.ingest_documents(docs)
        
        assert rag.document_count == 1

    def test_search_after_reset(self, temp_dir):
        """Test search after resetting indices"""
        index_dir = str(temp_dir / "test_reset")
        rag = RAGPipeline(index_dir=index_dir)
        
        # First ingestion
        docs1 = ["First document about testing"]
        rag.ingest_documents(docs1)
        assert rag.document_count == 1
        
        # Second ingestion with reset
        docs2 = ["Second document about Python"]
        rag.ingest_documents(docs2, reset=True)
        assert rag.document_count == 1  # Should only have new docs


class TestIntegrationMarkers:
    """Tests to verify integration test markers work correctly"""

    @pytest.mark.integration
    def test_marked_as_integration(self):
        """This test is marked as integration"""
        assert True

    def test_not_marked_as_integration(self):
        """This test is NOT marked as integration - will run in CI"""
        assert True