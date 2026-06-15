# tests/test_interfaces.py

"""Tests that verify the Protocol contracts in src.interfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.interfaces import (
    BM25RetrieverProtocol,
    CacheProtocol,
    EmbedderProtocol,
    LLMClientProtocol,
    RerankerProtocol,
    VectorStoreProtocol,
)


class _MockEmbedder:
    """Minimal class duck-typing the EmbedderProtocol."""

    model_name = "mock"
    dimension = 8

    def encode(self, texts, batch_size: int = 32, show_progress: bool = True, input_type=None):
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(list(texts)), self.dimension), dtype=np.float32)


class _MockVectorStore:
    persist_directory = "/tmp/mock"
    collection_name = "mock"

    def __init__(self):
        self._docs: List[str] = []

    def add_documents(self, texts, embeddings, metadatas=None, ids=None):
        self._docs.extend(texts)

    def search(self, query_embedding, n_results: int = 5):
        return {"documents": self._docs[:n_results], "distances": [], "metadatas": []}

    def get_embeddings(self, ids):
        return None  # this mock doesn't retain vectors; None ⇒ "recompute"

    def delete_collection(self, name: Optional[str] = None) -> None:
        self._docs.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {"total_documents": len(self._docs)}


class _MockBM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def search(self, query: str, k: int = 5):
        return [{"text": doc, "score": 1.0, "rank": i} for i, doc in enumerate(self.corpus[:k])]


class _MockCache:
    def __init__(self):
        self._d: Dict[str, Any] = {}

    def get(self, key): return self._d.get(key)
    def set(self, key, value, ttl=None): self._d[key] = value
    def delete(self, key): self._d.pop(key, None)
    def clear(self): self._d.clear()


class _MockReranker:
    def rerank(self, query, candidates, top_n=None):
        return candidates[: top_n or len(candidates)]


class _MockLLM:
    model = "mock-llm"

    def generate(self, prompt, context=None, max_tokens=512, temperature=0.7, stream=True):
        return f"answered: {prompt}"


class TestProtocolRuntimeChecks:
    """Each Protocol is runtime_checkable; mocks should satisfy isinstance()."""

    def test_embedder_protocol(self):
        assert isinstance(_MockEmbedder(), EmbedderProtocol)

    def test_vector_store_protocol(self):
        assert isinstance(_MockVectorStore(), VectorStoreProtocol)

    def test_bm25_protocol(self):
        assert isinstance(_MockBM25(["a", "b"]), BM25RetrieverProtocol)

    def test_cache_protocol(self):
        assert isinstance(_MockCache(), CacheProtocol)

    def test_reranker_protocol(self):
        assert isinstance(_MockReranker(), RerankerProtocol)

    def test_llm_protocol(self):
        assert isinstance(_MockLLM(), LLMClientProtocol)


class TestProtocolNegativeChecks:
    """Wrong-shaped objects should NOT satisfy the Protocol."""

    def test_missing_method_fails(self):
        class _Bad:
            persist_directory = "x"
            collection_name = "y"
            # missing add_documents, search, delete_collection, get_stats

        assert not isinstance(_Bad(), VectorStoreProtocol)

    def test_arbitrary_object_fails(self):
        assert not isinstance(object(), CacheProtocol)


class TestMockBehavior:
    """Smoke-test the mocks themselves so they're useful in later tests."""

    def test_vector_store_round_trip(self):
        vs = _MockVectorStore()
        vs.add_documents(["a", "b", "c"], [[0.0] * 8] * 3)
        assert vs.get_stats()["total_documents"] == 3
        assert vs.search([0.0] * 8, n_results=2)["documents"] == ["a", "b"]

    def test_cache_round_trip(self):
        c = _MockCache()
        c.set("k", 1)
        assert c.get("k") == 1
        c.delete("k")
        assert c.get("k") is None
