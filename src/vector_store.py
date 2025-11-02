import os
import tempfile
from typing import Any, Dict, Optional, Sequence

class VectorStore:
    """
    Lightweight wrapper around Chroma that tries the modern API first
    and falls back to legacy API if necessary. Designed for CI usage
    where disk permissions and existing DB files can cause issues.
    """

    def __init__(
        self,
        persist_directory: str = "indices/chroma_db",
        collection_name: str = "vectorflow_docs",
    ):
        # prepare persistence dir (use tmp if not writable or in CI)
        try:
            os.makedirs(persist_directory, exist_ok=True)
            test_file = os.path.join(persist_directory, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            self.persist_directory = persist_directory
        except Exception:
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_")

        if os.environ.get("GITHUB_ACTIONS") or not os.access(self.persist_directory, os.W_OK):
            # force a clean temp dir in CI or when not writable
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_")
            # recommended backend for local/CI durable storage
            os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")
            os.environ.setdefault("CHROMA_DB_DIR", self.persist_directory)

        # Initialize chroma client. Try new API then legacy.
        self.client = None
        try:
            import chromadb
            # preferred path: PersistentClient available in newer releases
            if hasattr(chromadb, "PersistentClient"):
                try:
                    from chromadb import PersistentClient  # type: ignore
                    self.client = PersistentClient(path=self.persist_directory)
                except Exception as e_new:
                    # If PersistentClient fails, attempt fallback to Client API
                    # (this can happen on some patched releases). We'll try legacy next.
                    last_exc = e_new
                    self.client = None
            else:
                last_exc = None
        except Exception as ie:
            # chromadb not importable at all
            raise RuntimeError(f"chromadb import failed: {ie}") from ie

        # If we don't have a client yet, attempt legacy Client path
        if self.client is None:
            try:
                # legacy API fallback
                from chromadb import Client, Settings  # type: ignore
                # Some older versions require DEFAULT_TENANT/DEFAULT_DATABASE
                try:
                    from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE  # type: ignore
                    tenant = DEFAULT_TENANT
                    database = DEFAULT_DATABASE
                except Exception:
                    tenant = None
                    database = None

                settings = Settings(
                    allow_reset=True,
                    is_persistent=True,
                    persist_directory=self.persist_directory,
                )

                if tenant is not None and database is not None:
                    self.client = Client(tenant=tenant, database=database, settings=settings)
                else:
                    # Some older Client signatures accept only settings
                    self.client = Client(settings=settings)
            except Exception as legacy_exc:
                # Provide clear error with both exceptions if available
                combined = f"PersistentClient error: {locals().get('last_exc', None)}; Legacy Client error: {legacy_exc}"
                raise RuntimeError(f"Could not initialize chromadb client: {combined}") from legacy_exc

        # Confirm client created
        if self.client is None:
            raise RuntimeError("Could not initialize chromadb client: unknown error")

        # Setup collection and in-memory trackers
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection(collection_name)
        self.documents = []
        self.embeddings = []

    def _get_or_create_collection(self, name: str):
        # Prefer get_or_create_collection if present
        if hasattr(self.client, "get_or_create_collection"):
            try:
                return self.client.get_or_create_collection(name=name, metadata={"desc": "docs"})
            except Exception:
                pass

        # Try get_collection
        if hasattr(self.client, "get_collection"):
            try:
                return self.client.get_collection(name)
            except Exception:
                pass

        # Try create_collection
        if hasattr(self.client, "create_collection"):
            try:
                return self.client.create_collection(name=name, metadata={"desc": "docs"})
            except Exception:
                pass

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
                # newer clients
                self.client.delete_collection(name)
                return
            # else try to get collection and call its delete
            if hasattr(self.client, "get_collection"):
                try:
                    col = self.client.get_collection(name)
                except Exception:
                    col = None
                if col is not None and hasattr(col, "delete"):
                    col.delete()
        except Exception:
            # swallow deletion errors (tests expect resilience)
            pass

    def add_documents(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ):
        # normalize embeddings
        try:
            embeddings_list = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
        except Exception:
            embeddings_list = list(embeddings)

        ids = list(ids) if ids else [f"id_{i}" for i in range(len(texts))]
        metadatas = list(metadatas) if metadatas is not None else [{"source": "manual"} for _ in texts]

        # collection.add is stable across versions
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
