# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\src\bm25_retriever.py

import bm25s, Stemmer
from typing import List, Dict

class BM25Retriever:
    def __init__(self, corpus:List[str]):
        self.stemmer=Stemmer.Stemmer("english")
        tokens=bm25s.tokenize(corpus, stemmer=self.stemmer, show_progress=False)
        self.retriever=bm25s.BM25(); self.retriever.index(tokens); self.corpus=corpus
    def search(self, query:str, k=5):
        qt=bm25s.tokenize(query, stemmer=self.stemmer)
        docs,scores=self.retriever.retrieve(qt,k=k)
        return [{"text":docs[0,i],"score":float(scores[0,i]),"rank":i+1} for i in range(docs.shape[1])]

if __name__=="__main__":
    c=["Python language","Machine learning","Semantic search"]
    r=BM25Retriever(c)
    print(r.search("machine",3))