import os
import tempfile
from typing import Any, Dict, Optional, Sequence
from chromadb import PersistentClient

class VectorStore:
    def __init__(
        self,
        persist_directory: str = "indices/chroma_db",
        collection_name: str = "vectorflow_docs",
    ):
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory

        if os.environ.get("GITHUB_ACTIONS") or not os.access(self.persist_directory, os.W_OK):
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_")

        try:
            self.client = PersistentClient(path=self.persist_directory)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize chromadb client: {exc}")

        self.collection_name = collection_name
        self.collection = self._get_or_create_collection(collection_name)
        self.documents = []
        self.embeddings = []

    def _get_or_create_collection(self, name: str):
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name=name, metadata={"desc": "docs"})

    def create_collection(self, reset: bool = False):
        if reset:
            self.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection(self.collection_name)
        return self.collection

    def delete_collection(self, name: Optional[str] = None):
        name = name or self.collection_name
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

    def add_documents(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ):
        embeddings_list = [
            e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings
        ]
        ids = list(ids) if ids else [f"id_{i}" for i in range(len(texts))]
        metadatas = metadatas or [{"source": "manual"} for _ in texts]

        self.collection.add(
            documents=list(texts),
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids,
        )

        self.documents.extend(texts)
        self.embeddings.extend(embeddings_list)
        self.embedding_dim = len(embeddings_list[0]) if embeddings_list else 0

    def search(self, query_embedding, n_results=5):
        if not self.documents:
            return {"documents": [], "distances": [], "metadatas": []}

        q_emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        res = self.collection.query(query_embeddings=[q_emb], n_results=n_results)
        return {
            "documents": res.get("documents", [[]])[0],
            "distances": res.get("distances", [[]])[0],
            "metadatas": res.get("metadatas", [[]])[0],
        }

    def get_stats(self):
        return {
            "collection_name": self.collection_name,
            "num_documents": len(self.documents),
            "embedding_dim": getattr(self, "embedding_dim", 0),
            "total_documents": len(self.documents),
        }
