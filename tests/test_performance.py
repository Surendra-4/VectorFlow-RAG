"""
Performance benchmarking tests
"""

import time

import numpy as np
import pytest

from src.bm25_retriever import BM25Retriever
from src.chunker import TextChunker
from src.embedder import Embedder


class TestEmbedderPerformance:
    """Test embedder performance"""

    @pytest.mark.slow
    def test_embedding_throughput(self):
        """Test embedding throughput"""
        embedder = Embedder()

        # Generate test data
        texts = ["Sample text {}".format(i) for i in range(100)]

        start = time.time()
        embeddings = embedder.encode(texts, show_progress=False)
        elapsed = time.time() - start

        throughput = len(texts) / elapsed

        print(f"\nEmbedding throughput: {throughput:.1f} docs/sec")
        print(f"Total time: {elapsed:.2f}s for {len(texts)} documents")

        # Should handle 100 docs in reasonable time
        assert elapsed < 60  # Less than 1 minute

    @pytest.mark.slow
    def test_embedding_memory_efficiency(self):
        """Test memory usage"""
        embedder = Embedder()

        texts = ["Memory test {}".format(i) for i in range(1000)]

        embeddings = embedder.encode(texts, show_progress=False)

        # Should fit in memory
        assert embeddings.shape[0] == 1000
        assert embeddings.shape[1] == 384


class TestBM25Performance:
    """Test BM25 performance"""

    @pytest.mark.slow
    def test_bm25_indexing_speed(self):
        """Test BM25 indexing performance"""
        corpus = [f"Document {i} with content" for i in range(1000)]

        start = time.time()
        retriever = BM25Retriever(corpus=corpus)
        elapsed = time.time() - start

        print(f"\nBM25 indexing: {len(corpus)} docs in {elapsed:.2f}s")

        assert elapsed < 30  # Should index quickly

    @pytest.mark.slow
    def test_bm25_search_latency(self):
        """Test BM25 search latency"""
        corpus = [f"Document {i}" for i in range(500)]
        retriever = BM25Retriever(corpus=corpus)

        queries = [f"query {i}" for i in range(50)]

        start = time.time()
        for query in queries:
            retriever.search(query, k=10)
        elapsed = time.time() - start

        avg_latency = (elapsed / len(queries)) * 1000

        print(f"\nBM25 search latency: {avg_latency:.2f}ms average")

        assert avg_latency < 100  # Should be fast


@pytest.mark.slow
class TestEndToEndPerformance:
    """Test end-to-end performance"""

    def test_full_pipeline_latency(self):
        """Test complete pipeline latency"""
        import os
        import shutil

        from src.rag_pipeline import RAGPipeline

        test_dir = "indices\\test_perf_e2e"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        rag = RAGPipeline(index_dir=test_dir)

        # Ingest documents
        docs = [f"Document {i} about testing" for i in range(10)]
        rag.ingest_documents(docs)

        # Measure search latency
        queries = [f"search query {i}" for i in range(10)]
        latencies = []

        for query in queries:
            start = time.time()
            rag.search(query, k=3)
            latencies.append((time.time() - start) * 1000)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"\nSearch latency (average): {avg_latency:.2f}ms")
        print(f"Search latency (p95): {p95_latency:.2f}ms")

        # Should be reasonably fast
        assert avg_latency < 200  # Average < 200ms

        shutil.rmtree(test_dir, ignore_errors=True)
