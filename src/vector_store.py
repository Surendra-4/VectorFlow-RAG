"""
Vector Store module using ChromaDB for efficient vector storage and retrieval
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from chromadb import PersistentClient


class VectorStore:
    """
    Vector database for storing and retrieving document embeddings.

    Uses ChromaDB with PersistentClient (Chroma >=0.5.3 API).
    Provides cross-platform support using pathlib.
    """

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = "vectorflow_docs",
    ):
        """
        Initialize VectorStore with persistence directory.

        Args:
            persist_directory: Directory to persist ChromaDB data.
                              If None, uses "indices/chroma_db"
            collection_name: Name of the collection to create/retrieve
        """
        # Convert default path using pathlib for cross-platform compatibility
        if persist_directory is None:
            persist_directory = str(Path("indices") / "chroma_db")
        else:
            # Convert to Path and back to string to normalize separators
            persist_directory = str(Path(persist_directory))

        # Ensure writable directory
        # Just use the path as-is, don't try to test it
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.persist_directory = persist_directory

        # Initialize ChromaDB client (using modern PersistentClient API)
        # CRITICAL: Pass string path, not Path object! ChromaDB concatenates with strings
        try:
            self.client = PersistentClient(path=self.persist_directory)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize chromadb client: {exc}")

        self.collection_name = collection_name
        self.collection = self._get_or_create_collection(collection_name)
        self.documents = []
        self.embeddings = []
        self.embedding_dim = 0

    def _get_or_create_collection(self, name: str):
        """
        Get or create a ChromaDB collection.

        Args:
            name: Name of the collection

        Returns:
            ChromaDB collection object
        """
        try:
            return self.client.get_or_create_collection(name=name, metadata={"desc": "docs"})
        except Exception:
            # Fallback for older ChromaDB versions
            try:
                return self.client.create_collection(name=name, metadata={"desc": "docs"})
            except Exception as e:
                raise RuntimeError(f"Unable to create or retrieve collection '{name}': {e}")

    def create_collection(self, reset: bool = False):
        """
        Create or retrieve a collection.

        Args:
            reset: If True, delete existing collection first

        Returns:
            Collection object
        """
        if reset:
            self.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection(self.collection_name)
        return self.collection

    def delete_collection(self, name: Optional[str] = None):
        """
        Delete a collection from ChromaDB.

        Args:
            name: Collection name. If None, uses self.collection_name
        """
        name = name or self.collection_name
        try:
            if hasattr(self.client, "delete_collection"):
                self.client.delete_collection(name)
                print(f"✓ Collection '{name}' deleted successfully")
            elif hasattr(self.client, "get_collection"):
                col = self.client.get_collection(name)
                if col and hasattr(col, "delete"):
                    col.delete()
                    print(f"✓ Collection '{name}' deleted successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not delete collection '{name}': {e}")

    def add_documents(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ):
        """
        Add documents with embeddings to the vector store.

        Args:
            texts: List of document texts
            embeddings: List of embedding vectors (numpy arrays or lists)
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents. Auto-generated if not provided.
        """
        # Convert embeddings to lists if they're numpy arrays
        try:
            embeddings_list = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
        except Exception:
            embeddings_list = list(embeddings)

        # Generate IDs if not provided
        ids = list(ids) if ids else [f"id_{i}" for i in range(len(texts))]

        # Generate metadata if not provided
        metadatas = list(metadatas) if metadatas is not None else [{"source": "manual"} for _ in texts]

        # Add to ChromaDB
        self.collection.add(
            documents=list(texts),
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids,
        )

        # Track in memory
        self.documents.extend(texts)
        self.embeddings.extend(embeddings_list)
        self.embedding_dim = len(embeddings_list[0]) if embeddings_list else 0

    def search(self, query_embedding, n_results=5):
        """
        Search for similar documents using embedding.

        Args:
            query_embedding: Query embedding vector (numpy array or list)
            n_results: Number of results to return

        Returns:
            Dict with "documents", "distances", "metadatas" keys
        """
        # Handle empty store
        if not self.documents:
            return {"documents": [], "distances": [], "metadatas": []}

        # Convert query embedding to list if needed
        q_emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        n_results = max(1, int(n_results))

        # Query ChromaDB
        res = self.collection.query(query_embeddings=[q_emb], n_results=n_results)
        documents = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]

        return {"documents": documents, "distances": distances, "metadatas": metadatas}

    def get_stats(self):
        """
        Get statistics about the vector store.

        Returns:
            Dict with collection stats
        """
        return {
            "collection_name": self.collection_name,
            "num_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "total_documents": len(self.documents),
        }
