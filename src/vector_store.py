# src/vector_store.py
import os
from typing import Any, Dict, Optional, Sequence
import tempfile
import chromadb
from chromadb.config import Settings


# Keep Embedder import usage local to callers. VectorStore is agnostic of embedder.
class VectorStore:
    def __init__(
        self,
        persist_directory: str = "indices/chroma_db",
        collection_name: str = "vectorflow_docs",
    ):
        
        # Use Settings-backed Client to be compatible across chroma versions
        try:
            os.makedirs(persist_directory, exist_ok=True)
            test_file = os.path.join(persist_directory, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)

        except Exception:
            persist_directory = tempfile.mkdtemp(prefix="chroma_")

        self.persist_directory = persist_directory

        # Ensure Chroma always writes to a writable temp dir if needed
        if os.environ.get("GITHUB_ACTIONS") or not os.access(self.persist_directory, os.W_OK):
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_")
            os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
            os.environ["CHROMA_DB_DIR"] = self.persist_directory

        try:
            self.client=chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_directory,
                )
            )

        except Exception:
            # fallback to older API if available
            try:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
            except Exception as exc:
                raise RuntimeError(f"Could not initialize chromadb client: {exc}")
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection(collection_name)
        self.documents = []   # to track added documents
        self.embeddings = []  # to track stored embeddings

    def _get_or_create_collection(self, name: str):
        try:
            return self.client.get_collection(name)
        except Exception:
            # chroma has get_or_create_collection or create_collection depending on version
            try:
                return self.client.get_or_create_collection(
                    name=name, metadata={"desc": "docs"}
                )
            except Exception:
                return self.client.create_collection(
                    name=name, metadata={"desc": "docs"}
                )

    def create_collection(self, reset: bool = False):
        if reset:
            self.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection(self.collection_name)
        return self.collection

    def delete_collection(self, name: Optional[str] = None):
        name = name or self.collection_name
        try:
            # try client-level delete
            if hasattr(self.client, "delete_collection"):
                self.client.delete_collection(name)
            else:
                # some chroma APIs remove collection via client.get_collection(name).delete()
                col = self.client.get_collection(name)
                if hasattr(col, "delete"):
                    col.delete()
        except Exception:
            # ignore if collection doesn't exist
            pass

    def add_documents(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ):
        # Normalize inputs
        # Convert numpy arrays to lists if needed
        try:
            embeddings_list = [
                e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings
            ]
        except Exception:
            embeddings_list = list(embeddings)

        ids = list(ids) if ids is not None else [f"id_{i}" for i in range(len(texts))]
        metadatas = (
            list(metadatas)
            if metadatas is not None
            else [{"source": "manual"} for _ in texts]
        )

        # add expects lists in many chroma versions
        self.collection.add(
            documents=list(texts),
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids,
        )

        self.documents.extend(texts)
        self.embeddings.extend(embeddings_list)
        self.embedding_dim = len(embeddings_list[0]) if embeddings_list else 0

    def search(self, query_embedding, n_results = 5):

        if not self.documents:
            return {"documents": [], "distances": [], "metadatas": []}
    
        # Accept query_embedding as list or numpy vector
        q_emb = (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else list(query_embedding)
        )

        # guard: ensure k >=1
        n_results = max(1, int(n_results))
        res = self.collection.query(query_embeddings=[q_emb], n_results=n_results)
        # query returns lists inside; return safe defaults if missing
        documents = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        return {"documents": documents, "distances": distances, "metadatas": metadatas}
    
    def get_stats(self):
        stats = {
            "collection_name": self.collection_name,
            "num_documents": len(self.documents),
            "embedding_dim": getattr(self, "embedding_dim", 0),
            "total_documents": len(self.documents)
            }
        return stats



if __name__ == "__main__":
    # quick smoke test (requires chroma and Embedder)
    from src.embedder import Embedder

    e = Embedder()
    vs = VectorStore("indices/test_chroma")
    texts = ["Hello", "World"]
    embs = e.encode(texts)
    vs.create_collection(reset=True)
    vs.add_documents(texts=texts, embeddings=embs, ids=["t1", "t2"])
    print(vs.search(embs[0], n_results=2))
