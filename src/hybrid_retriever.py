# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\src\hybrid_retriever.py

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.bm25_retriever import BM25Retriever

class HybridRetriever:
    def __init__(self, embedder:Embedder, vector_store:VectorStore, bm25_retriever:BM25Retriever, alpha=0.5):
        self.e, self.vs, self.bm, self.alpha = embedder, vector_store, bm25_retriever, alpha

    def search(self, query:str, k=5):
        vec_emb = self.e.encode(query)[0]

        # Ensure k does not exceed the corpus size for BM25
        bm25_k = min(k*3, len(self.bm.corpus))
        vres = self.vs.search(vec_emb, n_results=k*3)
        bres = self.bm.search(query, k=bm25_k)

        combined = {}
        for d, dist in zip(vres["documents"], vres["distances"]):
            sim = 1 / (1 + dist)
            combined.setdefault(d, {"vector_score": 0, "bm25_score": 0})["vector_score"] = sim

        max_b = max(r["score"] for r in bres) or 1
        for r in bres:
            combined.setdefault(r["text"], {"vector_score": 0, "bm25_score": 0})["bm25_score"] = r["score"] / max_b

        final = sorted(
            [{"text": str(d), "hybrid_score": self.alpha*s["vector_score"] + (1-self.alpha)*s["bm25_score"]} 
             for d, s in combined.items()],
            key=lambda x: -x["hybrid_score"]
        )[:k]

        return final

if __name__ == "__main__":
    # Use a larger corpus to avoid small-corpus issues
    docs = [
        "Apple is a fruit",
        "Banana is yellow",
        "Cherry is red",
        "Dates are sweet",
        "Elderberry is healthy",
        "Fig is delicious"
    ]

    e = Embedder()
    vs = VectorStore("indices\\hybrid_test")
    vs.add_documents(docs, e.encode(docs).tolist())

    bm = BM25Retriever(docs)
    h = HybridRetriever(e, vs, bm, alpha=0.5)

    print(h.search("Apple", k=3))
