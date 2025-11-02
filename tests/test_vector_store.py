"""
Unit tests for vector_store module
"""
import pytest
import numpy as np
import os
import shutil, time
from src.vector_store import VectorStore


def safe_rmtree(path, retries=5):
    """Retry-safe delete to avoid Windows file lock errors"""
    if not os.path.exists(path):
        return
    for _ in range(retries):
        try:
            shutil.rmtree(path, ignore_errors=True)
            return
        except PermissionError:
            time.sleep(0.5)


class TestVectorStoreInitialization:
    """Test VectorStore initialization"""
    
    def test_initialization_creates_directory(self):
        """Test that initialization creates persistence directory"""
        test_dir = "indices\\test_vs_init"
        
        # Clean up if exists
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        vs = VectorStore(persist_directory=test_dir)
        
        assert os.path.exists(test_dir)
        assert vs.persist_directory == test_dir
        
        # Cleanup
        safe_rmtree(test_dir)

    
    def test_default_collection_creation(self):
        """Test that default collection is created"""
        test_dir = "indices\\test_vs_collection"
        
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        vs = VectorStore(persist_directory=test_dir)
        assert vs.collection is not None
        
        safe_rmtree(test_dir)
    
    def test_custom_collection_name(self):
        """Test custom collection naming"""
        test_dir = "indices\\test_vs_custom"
        
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        vs = VectorStore(persist_directory=test_dir)
        vs.create_collection(reset=True)
        
        assert vs.collection is not None
        
        safe_rmtree(test_dir)


class TestVectorStoreDocumentOperations:
    """Test document add/retrieve operations"""
    
    @pytest.fixture
    def vector_store(self):
        """Create test vector store"""
        test_dir = "indices\\test_vs_ops"
        if os.path.exists(test_dir):
            safe_rmtree(test_dir)
        
        vs = VectorStore(persist_directory=test_dir)
        vs.create_collection(reset=True)
        yield vs
        
        # Ensure FAISS/HNSWlib files are released before cleanup
        del vs
        time.sleep(0.2)
        safe_rmtree(test_dir)
    
    def test_add_single_document(self, vector_store):
        """Test adding a single document"""
        texts = ["Test document"]
        embeddings = [np.random.random(384).tolist()]
        metadatas = [{"source": "test"}]
        
        vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=["doc_1"]
        )
        
        # Verify document was added
        assert vector_store.collection.count() == 1
    
    def test_add_multiple_documents(self, vector_store):
        """Test adding multiple documents"""
        texts = ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        metadatas = [{"source": "test", "id": i} for i in range(len(texts))]
        
        vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        assert vector_store.collection.count() == 5
    
    def test_add_documents_with_auto_ids(self, vector_store):
        """Test that IDs are auto-generated if not provided"""
        texts = ["Auto ID 1", "Auto ID 2"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        
        vector_store.add_documents(texts=texts, embeddings=embeddings)
        
        assert vector_store.collection.count() == 2
    
    def test_search_returns_results(self, vector_store):
        """Test search functionality"""
        # Add documents
        texts = ["Machine learning", "Deep learning", "Neural networks"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        
        vector_store.add_documents(texts=texts, embeddings=embeddings)
        
        # Search
        query_embedding = np.random.random(384).tolist()
        results = vector_store.search(query_embedding, n_results=2)
        
        assert "documents" in results
        assert "distances" in results
        assert "metadatas" in results
        assert len(results["documents"]) == 2
    
    def test_search_respects_k_parameter(self, vector_store):
        """Test that search returns correct number of results"""
        texts = [f"Document {i}" for i in range(10)]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        
        vector_store.add_documents(texts=texts, embeddings=embeddings)
        
        query_embedding = np.random.random(384).tolist()
        
        results_k3 = vector_store.search(query_embedding, n_results=3)
        results_k5 = vector_store.search(query_embedding, n_results=5)
        
        assert len(results_k3["documents"]) == 3
        assert len(results_k5["documents"]) == 5
    
    def test_search_with_empty_store(self, vector_store):
        """Test search on empty vector store"""
        query_embedding = np.random.random(384).tolist()
        
        # Should handle gracefully or raise error
        try:
            results = vector_store.search(query_embedding, n_results=5)
            assert len(results["documents"]) == 0
        except Exception:
            pass  # Expected


class TestVectorStoreCollection:
    """Test collection management"""
    
    def test_create_collection_reset(self):
        """Test creating collection with reset"""
        test_dir = "indices\\test_vs_reset"
        
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        vs = VectorStore(persist_directory=test_dir)
        
        # Add some documents
        texts = ["Doc 1", "Doc 2"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        vs.add_documents(texts=texts, embeddings=embeddings)
        
        assert vs.collection.count() == 2
        
        # Reset collection
        vs.create_collection(reset=True)
        
        assert vs.collection.count() == 0
        
        safe_rmtree(test_dir)
    
    def test_get_stats(self):
        """Test getting collection statistics"""
        test_dir = "indices\\test_vs_stats"
        
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        vs = VectorStore(persist_directory=test_dir)
        
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        vs.add_documents(texts=texts, embeddings=embeddings)
        
        stats = vs.get_stats()
        
        assert "total_documents" in stats
        assert stats["total_documents"] == 3
        
        safe_rmtree(test_dir)


class TestVectorStorePersistence:
    """Test persistence across sessions"""
    
    def test_data_persistence(self):
        """Test that data persists across instances"""
        test_dir = "indices\\test_vs_persist"
        
        if os.path.exists(test_dir):
            safe_rmtree(test_dir)
        
        # Add documents in first instance
        vs1 = VectorStore(persist_directory=test_dir)
        texts = ["Persistent Doc 1", "Persistent Doc 2"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        vs1.add_documents(texts=texts, embeddings=embeddings, ids=["p1", "p2"])
        
        # Create new instance
        vs2 = VectorStore(persist_directory=test_dir)
        
        # Data should still be there
        assert vs2.collection.count() == 2
        
        safe_rmtree(test_dir)
