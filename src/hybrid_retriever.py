# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\src\hybrid_retriever.py

from src.bm25_retriever import BM25Retriever
from src.embedder import Embedder
from src.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, embedder, vector_store, bm25_retriever, alpha=0.5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.alpha = alpha

    def search(self, query: str, k=None, n_results=None):
        k = n_results or k or 5
        # Encode query
        vec_emb = self.embedder.encode(query)[0]

        # Retrieve vector-based results
        vector_results = self.vector_store.search(vec_emb, n_results=k)
        docs = vector_results.get("documents", [])
        dists = vector_results.get("distances", [])

        # Get BM25 results (limit to corpus size)
        bm25_k = min(k * 3, len(self.bm25.corpus))
        bm25_results = self.bm25.search(query, k=bm25_k)

        # Merge by normalized hybrid scoring
        vector_scores = {}
        for doc, dist in zip(docs, dists):
            score = 1 / (1 + dist) if dist is not None else 0
            vector_scores[doc] = score
        bm25_scores = {r["text"]: r["score"] for r in bm25_results}

        all_texts = set(vector_scores.keys()) | set(bm25_scores.keys())
        results = []
        for text in all_texts:
            v = vector_scores.get(text, 0)
            b = bm25_scores.get(text, 0)
            hybrid_score = self.alpha * v + (1 - self.alpha) * b
            results.append({"text": text, "hybrid_score": hybrid_score})

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:k]


if __name__ == "__main__":
    docs = [
        "Apple is a fruit",
        "Banana is yellow",
        "Cherry is red",
        "Dates are sweet",
        "Elderberry is healthy",
        "Fig is delicious",
    ]

    embedder = Embedder()
    vector_store = VectorStore("indices\\hybrid_test")
    vector_store.add_documents(docs, embedder.encode(docs).tolist())

    bm25 = BM25Retriever(docs)
    hybrid = HybridRetriever(embedder, vector_store, bm25, alpha=0.5)

    print(hybrid.search("Apple", k=3))
