import os
import tempfile
from typing import Any, Dict, Optional, Sequence

# Handle both new (>=0.5.x) and legacy (<0.5) chromadb APIs
try:
    from chromadb import PersistentClient  # new API
    CHROMA_MODE = "new"
except ImportError:
    try:
        from chromadb import Client, Settings
        from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE
        CHROMA_MODE = "legacy"
    except ImportError:
        CHROMA_MODE = "none"


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "indices/chroma_db",
        collection_name: str = "vectorflow_docs",
    ):
        # Ensure writable directory
        try:
            os.makedirs(persist_directory, exist_ok=True)
            test_file = os.path.join(persist_directory, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            self.persist_directory = persist_directory
        except Exception:
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_")

        # Enforce tmp dir for CI
        if os.environ.get("GITHUB_ACTIONS") or not os.access(self.persist_directory, os.W_OK):
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_")

        # Initialize client according to API
        try:
            if CHROMA_MODE == "new":
                # Modern Chroma ≥0.5.3 API
                self.client = PersistentClient(path=self.persist_directory)
            elif CHROMA_MODE == "legacy":
                settings = Settings(
                    allow_reset=True,
                    is_persistent=True,
                    persist_directory=self.persist_directory,
                )
                self.client = Client(
                    tenant=DEFAULT_TENANT,
                    database=DEFAULT_DATABASE,
                    settings=settings,
                )
            else:
                raise RuntimeError("chromadb not installed or importable")
        except Exception as exc:
            raise RuntimeError(f"Could not initialize chromadb client: {exc}")

        self.collection_name = collection_name
        self.collection = self._get_or_create_collection(collection_name)
        self.documents = []
        self.embeddings = []

    def _get_or_create_collection(self, name: str):
        if hasattr(self.client, "get_or_create_collection"):
            return self.client.get_or_create_collection(name=name, metadata={"desc": "docs"})
        if hasattr(self.client, "create_collection"):
            return self.client.create_collection(name=name, metadata={"desc": "docs"})
        raise RuntimeError("Unable to create or retrieve Chroma collection")

    def create_collection(self, reset: bool = False):
        if reset:
            self.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection(self.collection_name)
        return self.collection

    def delete_collection(self, name: Optional[str] = None):
        name = name or self.collection_name
        try:
            if hasattr(self.client, "delete_collection"):
                self.client.delete_collection(name)
            elif hasattr(self.client, "get_collection"):
                col = self.client.get_collection(name)
                if col and hasattr(col, "delete"):
                    col.delete()
        except Exception:
            pass

    def add_documents(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ):
        try:
            embeddings_list = [
                e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings
            ]
        except Exception:
            embeddings_list = list(embeddings)

        ids = list(ids) if ids else [f"id_{i}" for i in range(len(texts))]
        metadatas = list(metadatas) if metadatas is not None else [{"source": "manual"} for _ in texts]

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
        n_results = max(1, int(n_results))

        res = self.collection.query(query_embeddings=[q_emb], n_results=n_results)
        documents = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        return {"documents": documents, "distances": distances, "metadatas": metadatas}

    def get_stats(self):
        return {
            "collection_name": self.collection_name,
            "num_documents": len(self.documents),
            "embedding_dim": getattr(self, "embedding_dim", 0),
            "total_documents": len(self.documents),
        }
