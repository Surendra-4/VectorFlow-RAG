# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\src\vector_store.py

import os, chromadb
from src.embedder import Embedder
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_directory="indices\\chroma_db"):
        os.makedirs(persist_directory,exist_ok=True)
        self.persist_directory = persist_directory
        self.client=chromadb.PersistentClient(path=persist_directory)
        self.collection=self.client.get_or_create_collection(name="vectorflow_docs", metadata={"desc":"docs"})
    def add_documents(self, texts:List[str], embeddings:List[List[float]], metadatas:List[Dict]=None, ids:List[str]=None):
        ids = ids or [f"id_{i}" for i in range(len(texts))]
        self.collection.add(documents=texts, embeddings=embeddings, metadatas = metadatas or [{"source": "manual"} for _ in texts], ids=ids)
    def search(self, query_embedding:List[float], n_results=5):
        res=self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return {"documents":res["documents"][0],"distances":res["distances"][0],"metadatas":res["metadatas"][0]}

if __name__=="__main__":
    # from src.embedder import Embedder
    e=Embedder(); vs=VectorStore("indices\\test_chroma"); vs.add_documents(["Hello","World"], e.encode(["Hello","World"]).tolist())
    print(vs.search(e.encode("Hello")[0],2))