import os
import tempfile
from typing import Any, Dict, Optional, Sequence

# Try to support both new and older chromadb releases.
try:
    # Chroma >= 0.5.x (new API)
    from chromadb import Client  # PersistentClient is deprecated
    from chromadb.config import Settings as ChromaSettings  # type: ignore
    CHROMA_CLIENT_TYPE = "persistent"
except Exception:
    try:
        # Older chromadb (<0.5) fallback
        from chromadb import Client, Settings  # type: ignore
        from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE  # type: ignore
        CHROMA_CLIENT_TYPE = "legacy"
    except Exception:
        CHROMA_CLIENT_TYPE = "none"


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "indices/chroma_db",
        collection_name: str = "vectorflow_docs",
    ):
        # make path writable; fall back to temp dir if needed
        try:
            os.makedirs(persist_directory, exist_ok=True)
            test_file = os.path.join(persist_directory, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            self.persist_directory = persist_directory
        except Exception:
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_")

        # In CI or if path not writable force temp dir and persistent backend env
        if os.environ.get("GITHUB_ACTIONS") or not os.access(self.persist_directory, os.W_OK):
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_")
            os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")
            os.environ.setdefault("CHROMA_DB_DIR", self.persist_directory)

        # Initialize client with whichever API is available
        try:
            if CHROMA_CLIENT_TYPE == "persistent":
                # New API (Chroma ≥0.5.x)
                settings = ChromaSettings(
                    allow_reset=True,
                    is_persistent=True,
                    persist_directory=self.persist_directory,
                )
                self.client = Client(settings)
            elif CHROMA_CLIENT_TYPE == "legacy":
                # Older API fallback
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
        self.documents = []   # in-memory tracking for tests
        self.embeddings = []

    def _get_or_create_collection(self, name: str):
        """Use get_or_create_collection when available, fallback to create/get."""
        if hasattr(self.client, "get_or_create_collection"):
            try:
                return self.client.get_or_create_collection(name=name, metadata={"desc": "docs"})
            except Exception:
                pass

        try:
            if hasattr(self.client, "get_collection"):
                return self.client.get_collection(name)
        except Exception:
            pass

        if hasattr(self.client, "create_collection"):
            return self.client.create_collection(name=name, metadata={"desc": "docs"})
        raise RuntimeError("Unable to create or retrieve collection from chromadb client")

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
                return
            col = None
            if hasattr(self.client, "get_collection"):
                try:
                    col = self.client.get_collection(name)
                except Exception:
                    col = None
            if col is not None and hasattr(col, "delete"):
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
