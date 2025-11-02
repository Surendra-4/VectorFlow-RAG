"""
Integration tests for RAG pipeline
"""
import pytest
import os
import shutil
from src.rag_pipeline import RAGPipeline
import time


def safe_rmtree(path, retries=5):
    """Retry-safe delete to avoid Windows file lock errors"""
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(0.5)



class TestRAGPipelineInitialization:
    """Test RAGPipeline initialization"""
    
    def test_initialization(self):
        """Test pipeline initializes correctly"""
        rag = RAGPipeline(
            index_dir="indices\\test_rag_init",
            alpha=0.5,
            llm_model="tinyllama"
        )
        
        assert rag is not None
        assert rag.alpha == 0.5
    
    def test_different_configurations(self):
        """Test different configurations"""
        configs = [
            {"alpha": 0.0},
            {"alpha": 0.5},
            {"alpha": 1.0},
        ]
        
        for config in configs:
            rag = RAGPipeline(**config)
            assert rag.alpha == config["alpha"]


class TestDocumentIngestion:
    """Test document ingestion"""
    
    @pytest.fixture
    def rag_pipeline(self):
        """Create test pipeline"""
        test_dir = "indices\\test_rag_ingest"
        if os.path.exists(test_dir):
            safe_rmtree(test_dir)
        
        rag = RAGPipeline(index_dir=test_dir, alpha=0.5)
        yield rag
        
        safe_rmtree(test_dir)
    
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
    def indexed_pipeline(self):
        """Create indexed pipeline"""
        test_dir = "indices\\test_rag_retrieval"
        if os.path.exists(test_dir):
            safe_rmtree(test_dir)
        
        rag = RAGPipeline(index_dir=test_dir)
        
        docs = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "NLP processes natural language",
            "Computer vision recognizes images",
        ]
        
        rag.ingest_documents(docs)
        
        yield rag
        
        safe_rmtree(test_dir)
    
    def test_search_returns_results(self, indexed_pipeline):
        """Test search returns results"""
        results = indexed_pipeline.search("machine learning", k=2)
        
        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("hybrid_score" in r for r in results)
    
    def test_search_without_ingestion_fails(self):
        """Test search fails without ingestion"""
        test_dir = "indices\\test_rag_no_ingest"
        if os.path.exists(test_dir):
            safe_rmtree(test_dir)
        
        rag = RAGPipeline(index_dir=test_dir)
        
        with pytest.raises(ValueError):
            rag.search("query")
        
        safe_rmtree(test_dir)


class TestPipelineStats:
    """Test pipeline statistics"""
    
    def test_get_stats(self):
        """Test getting pipeline statistics"""
        test_dir = "indices\\test_rag_stats"
        if os.path.exists(test_dir):
            safe_rmtree(test_dir)
        
        rag = RAGPipeline(index_dir=test_dir, alpha=0.7)
        
        docs = ["Test 1", "Test 2", "Test 3"]
        rag.ingest_documents(docs)
        
        stats = rag.get_stats()
        
        assert "documents_ingested" in stats
        assert "chunks_indexed" in stats
        assert "embedding_dimension" in stats
        assert "alpha" in stats
        
        assert stats["documents_ingested"] == 3
        assert stats["alpha"] == 0.7
        
        safe_rmtree(test_dir)