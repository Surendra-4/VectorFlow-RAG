# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\src\embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
    def encode(self, texts, batch_size=32, show_progress=True):
        if isinstance(texts, str): texts = [texts]
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True, normalize_embeddings=True)
        
if __name__=="__main__":
    e=Embedder(); emb=e.encode(["Hello","World"]); print(emb.shape)