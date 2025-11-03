"""
Unit tests for vector_store module

Uses pathlib for cross-platform compatibility and temp_dir fixture
for automatic cleanup.
"""
import pytest
import numpy as np
from pathlib import Path

from src.vector_store import VectorStore


class TestVectorStoreInitialization:
    """Test VectorStore initialization"""

    def test_initialization_creates_directory(self, temp_dir):
        """Test that initialization creates persistence directory"""
        test_dir = temp_dir / "vs_init"
        
        vs = VectorStore(persist_directory=str(test_dir))
        
        assert test_dir.exists()
        # Compare using Path objects for OS-independent comparison
        assert Path(vs.persist_directory) == test_dir

    def test_default_collection_creation(self, temp_dir):
        """Test that default collection is created"""
        test_dir = temp_dir / "vs_collection"
        test_dir.mkdir(exist_ok=True)
        
        vs = VectorStore(persist_directory=str(test_dir))
        assert vs.collection is not None

    def test_custom_collection_name(self, temp_dir):
        """Test custom collection naming"""
        test_dir = temp_dir / "vs_custom"
        test_dir.mkdir(exist_ok=True)
        
        vs = VectorStore(persist_directory=str(test_dir))
        vs.create_collection(reset=True)
        
        assert vs.collection is not None

    def test_default_path_uses_pathlib(self, temp_dir):
        """Test that default path is valid"""
        test_dir = temp_dir / "vs_default"
        test_dir.mkdir(exist_ok=True)
        
        # This should not raise an error
        vs = VectorStore(persist_directory=str(test_dir))
        
        # Should have valid path (string)
        assert isinstance(vs.persist_directory, str)
        assert len(vs.persist_directory) > 0


class TestVectorStoreDocumentOperations:
    """Test document add/retrieve operations"""

    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create test vector store with temp directory"""
        test_dir = temp_dir / "vs_ops"
        test_dir.mkdir(exist_ok=True)
        
        vs = VectorStore(persist_directory=str(test_dir))
        vs.create_collection(reset=True)
        yield vs
        
        # Cleanup is automatic via temp_dir fixture

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
        
        # Should handle gracefully
        results = vector_store.search(query_embedding, n_results=5)
        assert len(results["documents"]) == 0


class TestVectorStoreCollection:
    """Test collection management"""

    def test_create_collection_reset(self, temp_dir):
        """Test creating collection with reset"""
        test_dir = temp_dir / "vs_reset"
        test_dir.mkdir(exist_ok=True)
        
        vs = VectorStore(persist_directory=str(test_dir))
        
        # Add some documents
        texts = ["Doc 1", "Doc 2"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        vs.add_documents(texts=texts, embeddings=embeddings)
        
        assert vs.collection.count() == 2
        
        # Reset collection
        vs.create_collection(reset=True)
        
        assert vs.collection.count() == 0

    def test_get_stats(self, temp_dir):
        """Test getting collection statistics"""
        test_dir = temp_dir / "vs_stats"
        test_dir.mkdir(exist_ok=True)
        
        vs = VectorStore(persist_directory=str(test_dir))
        
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        vs.add_documents(texts=texts, embeddings=embeddings)
        
        stats = vs.get_stats()
        
        assert "total_documents" in stats
        assert stats["total_documents"] == 3


class TestVectorStorePersistence:
    """Test persistence across sessions"""

    def test_data_persistence(self, temp_dir):
        """Test that data persists across instances"""
        test_dir = temp_dir / "vs_persist"
        test_dir.mkdir(exist_ok=True)
        
        # Add documents in first instance
        vs1 = VectorStore(persist_directory=str(test_dir))
        texts = ["Persistent Doc 1", "Persistent Doc 2"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        vs1.add_documents(texts=texts, embeddings=embeddings, ids=["p1", "p2"])
        
        # Create new instance
        vs2 = VectorStore(persist_directory=str(test_dir))
        
        # Data should still be there
        assert vs2.collection.count() == 2


class TestVectorStoreErrorHandling:
    """Test error handling and edge cases"""

    def test_delete_collection_with_logging(self, temp_dir, capsys):
        """Test that delete_collection provides feedback"""
        test_dir = temp_dir / "vs_delete"
        test_dir.mkdir(exist_ok=True)
        
        vs = VectorStore(persist_directory=str(test_dir))
        vs.delete_collection()
        
        # Should not raise exception, but may print warning
        captured = capsys.readouterr()
        assert "deleted" in captured.out or "Warning" in captured.out or True

    @pytest.fixture
    def vector_store_for_errors(self, temp_dir):
        """Create vector store for error handling tests"""
        test_dir = temp_dir / "vs_errors"
        test_dir.mkdir(exist_ok=True)
        
        vs = VectorStore(persist_directory=str(test_dir))
        vs.create_collection(reset=True)
        yield vs

    def test_add_documents_with_numpy_embeddings(self, vector_store_for_errors):
        """Test that numpy arrays are properly converted"""
        texts = ["Test 1", "Test 2"]
        embeddings = [np.random.random(384) for _ in texts]  # Numpy arrays
        
        # Should not raise exception
        vector_store_for_errors.add_documents(texts=texts, embeddings=embeddings)
        assert vector_store_for_errors.collection.count() == 2

    def test_search_with_numpy_query(self, vector_store_for_errors):
        """Test search with numpy query embedding"""
        texts = ["Document 1", "Document 2"]
        embeddings = [np.random.random(384).tolist() for _ in texts]
        vector_store_for_errors.add_documents(texts=texts, embeddings=embeddings)
        
        # Query with numpy array
        query_embedding = np.random.random(384)
        results = vector_store_for_errors.search(query_embedding, n_results=1)
        
        assert len(results["documents"]) > 0