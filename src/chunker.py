# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\src\chunker.py

import re


class TextChunker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size, self.overlap = chunk_size, overlap

    def chunk_text(self, text, metadata=None):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, curr, cid = [], "", 0
        for s in sentences:
            if len(curr) + len(s) > self.chunk_size and curr:
                chunks.append({"text": curr.strip(), "metadata": metadata or {}, "chunk_id": cid})
                curr = curr[-self.overlap :] + " " + s
                cid += 1
            else:
                curr += " " + s
        if curr.strip():
            chunks.append({"text": curr.strip(), "metadata": metadata or {}, "chunk_id": cid})
        return chunks


if __name__ == "__main__":
    c = TextChunker(100, 20)
    docs = "This is a test. Second sentence. Third part."
    print(c.chunk_text(docs))
