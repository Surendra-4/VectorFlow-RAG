# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\src\bm25_retriever.py

from typing import List

import bm25s
import Stemmer


class BM25Retriever:
    def __init__(self, corpus: List[str]):
        self.stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize(corpus, stemmer=self.stemmer, show_progress=False)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)
        self.corpus = corpus

    def search(self, query: str, k: int = 5):
        qt = bm25s.tokenize(query, stemmer=self.stemmer)
        docs, scores = self.retriever.retrieve(qt, k=k)
        results = []
        for i in range(docs.shape[1]):
            idx = int(docs[0, i])
            results.append(
                {"text": self.corpus[idx], "score": float(scores[0, i]), "rank": i + 1}
            )
        return results


if __name__ == "__main__":
    c = ["Python language", "Machine learning", "Semantic search"]
    r = BM25Retriever(c)
    print(r.search("machine", 3))
