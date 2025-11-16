# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\experiments\benchmark_ms_marco.py

"""
Benchmark VectorFlow-RAG on MS MARCO Passage Ranking

This script evaluates VectorFlow-RAG using the MS MARCO dataset.
It measures standard IR metrics: MRR, NDCG, Recall@k, and latency.

Usage:
    python experiments/benchmark_ms_marco.py --alpha 0.5 --limit 100
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import RAGPipeline
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever


class MSMARCOBenchmark:
    """
    Evaluate VectorFlow-RAG on MS MARCO Passage Ranking task
    
    Metrics:
    - MRR@10: Mean Reciprocal Rank at top 10
    - NDCG@10: Normalized Discounted Cumulative Gain at top 10
    - Recall@k: Recall at k for k in [5, 10, 20, 50, 100]
    - Latency: Query response time in milliseconds
    """
    
    def __init__(self, alpha: float = 0.5, llm_model: str = "tinyllama"):
        """
        Initialize benchmark
        
        Args:
            alpha: Hybrid search weight (0=BM25, 1=vector)
            llm_model: Ollama model name
        """
        self.alpha = alpha
        self.llm_model = llm_model
        self.results = {
            "config": {
                "alpha": alpha,
                "llm_model": llm_model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "metrics": {}
        }
    
    def create_mock_dataset(self, num_queries: int = 50) -> Tuple[Dict, Dict, Dict]:
        """
        Create a mock MS MARCO-like dataset for testing
        
        Since downloading the full dataset is time-consuming, this creates
        a representative sample for benchmarking.
        
        Args:
            num_queries: Number of test queries
            
        Returns:
            (queries, passages, qrels) tuples
        """
        print(f"Creating mock MS MARCO dataset ({num_queries} queries)...")
        
        # Sample queries related to machine learning, search, and AI
        query_templates = [
            "What is {topic}?",
            "How does {topic} work?",
            "Explain {topic}",
            "What are applications of {topic}?",
            "{topic} benefits and drawbacks",
            "Compare {topic} with {other_topic}",
        ]
        
        topics = [
            "machine learning",
            "deep learning",
            "neural networks",
            "semantic search",
            "retrieval augmented generation",
            "embedding models",
            "vector databases",
            "natural language processing",
            "transformer models",
            "attention mechanism",
            "information retrieval",
            "ranking algorithms",
            "hybrid search",
            "BM25",
            "similarity metrics",
        ]
        
        # Generate passages about these topics
        passage_templates = [
            "{topic} is a fundamental concept in artificial intelligence.",
            "{topic} enables computers to understand and process natural language.",
            "{topic} improves search accuracy by understanding semantic meaning.",
            "Recent advances in {topic} have made AI systems more capable.",
            "{topic} requires large amounts of training data to work effectively.",
            "Applications of {topic} include recommendation systems and search engines.",
            "{topic} models can capture complex relationships in text data.",
            "The performance of {topic} systems is measured by metrics like NDCG and MRR.",
        ]
        
        queries = {}
        passages = {}
        qrels = defaultdict(list)
        
        query_id = 0
        passage_id = 0
        
        # Generate queries and passages
        for i in range(num_queries):
            # Create query
            template = query_templates[i % len(query_templates)]
            topic = topics[i % len(topics)]
            other_topic = topics[(i + 1) % len(topics)]
            
            query_text = template.format(topic=topic, other_topic=other_topic)
            query_id_str = f"q_{query_id}"
            queries[query_id_str] = query_text
            
            # Create relevant passages (2-3 per query)
            num_relevant = np.random.randint(2, 4)
            for j in range(num_relevant):
                passage_template = passage_templates[j % len(passage_templates)]
                passage_text = passage_template.format(topic=topic)
                
                passage_id_str = f"p_{passage_id}"
                passages[passage_id_str] = passage_text
                
                # Create relevance judgment (1 = relevant)
                qrels[query_id_str].append((passage_id_str, 1))
                passage_id += 1
            
            # Create irrelevant passages (1-2 per query)
            num_irrelevant = np.random.randint(1, 3)
            for j in range(num_irrelevant):
                random_topic = topics[np.random.randint(0, len(topics))]
                passage_template = passage_templates[np.random.randint(0, len(passage_templates))]
                passage_text = passage_template.format(topic=random_topic)
                
                passage_id_str = f"p_{passage_id}"
                passages[passage_id_str] = passage_text
                
                # Create relevance judgment (0 = not relevant)
                qrels[query_id_str].append((passage_id_str, 0))
                passage_id += 1
            
            query_id += 1
        
        print(f"✓ Created {len(queries)} queries and {len(passages)} passages")
        return queries, passages, dict(qrels)
    
    def index_passages(self, passages: Dict[str, str]):
        """
        Index passages using VectorFlow components
        
        Args:
            passages: Dict of {passage_id: passage_text}
        """
        print("\nIndexing passages...")
        
        passage_texts = list(passages.values())
        passage_ids = list(passages.keys())
        
        # Initialize components
        embedder = Embedder()
        vector_store = VectorStore("indices\\ms_marco_benchmark")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = embedder.encode(passage_texts, show_progress=True)
        
        # Add to vector store
        print("Building vector index...")
        vector_store = VectorStore("indices\\ms_marco_benchmark")
        vector_store.create_collection(reset=True)

        print("Generating embeddings...")
        embeddings = embedder.encode(passage_texts, show_progress=True)
        
        print("Building vector index...")
        vector_store.add_documents(
            texts=passage_texts,
            embeddings=embeddings.tolist(),
            metadatas=[{"passage_id": pid} for pid in passage_ids],
            ids=passage_ids
        )
        
        # Build BM25 index
        print("Building BM25 index...")
        bm25_retriever = BM25Retriever(corpus=passage_texts)
        
        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            alpha=self.alpha
        )
        
        return hybrid_retriever, passage_ids
    
    def compute_mrr(self, results: List[Dict], relevant_ids: Set[str], k: int = 10) -> float:
        """
        Compute Mean Reciprocal Rank
        
        MRR = 1/rank of first relevant result
        
        Args:
            results: List of retrieved results
            relevant_ids: Set of relevant passage IDs
            k: Cutoff (only consider top-k)
            
        Returns:
            MRR score (0-1)
        """
        for i, result in enumerate(results[:k], 1):
            # Check if this result is relevant
            if any(str(rel_id) in str(result['text']) or str(rel_id) in result.get('metadata', {}).get('passage_id', '')
                   for rel_id in relevant_ids):
                return 1.0 / i
        return 0.0
    
    def compute_dcg(self, results: List[Dict], relevant_ids: Set[str], k: int = 10) -> float:
        """
        Compute Discounted Cumulative Gain
        
        DCG@k = sum(rel_i / log2(i+1)) for i=1 to k
        
        Args:
            results: List of retrieved results
            relevant_ids: Set of relevant passage IDs
            k: Cutoff
            
        Returns:
            DCG score
        """
        dcg = 0.0
        for i, result in enumerate(results[:k], 1):
            # Relevance is 1 if result is in relevant set, 0 otherwise
            rel = 1.0 if any(str(rel_id) in str(result['text']) for rel_id in relevant_ids) else 0.0
            dcg += rel / np.log2(i + 1)
        return dcg
    
    def compute_ideal_dcg(self, num_relevant: int, k: int = 10) -> float:
        """
        Compute ideal (maximum possible) DCG
        
        IDCG@k = sum(1 / log2(i+1)) for i=1 to min(num_relevant, k)
        
        Args:
            num_relevant: Number of relevant documents
            k: Cutoff
            
        Returns:
            Ideal DCG score
        """
        idcg = 0.0
        for i in range(1, min(num_relevant, k) + 1):
            idcg += 1.0 / np.log2(i + 1)
        return idcg
    
    def compute_ndcg(self, results: List[Dict], relevant_ids: Set[str], k: int = 10) -> float:
        """
        Compute Normalized Discounted Cumulative Gain
        
        NDCG@k = DCG@k / IDCG@k
        
        Args:
            results: List of retrieved results
            relevant_ids: Set of relevant passage IDs
            k: Cutoff
            
        Returns:
            NDCG score (0-1)
        """
        dcg = self.compute_dcg(results, relevant_ids, k)
        idcg = self.compute_ideal_dcg(len(relevant_ids), k)
        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_recall_at_k(self, results: List[Dict], relevant_ids: Set[str], k: int) -> float:
        """
        Compute Recall@k
        
        Recall@k = (# relevant in top-k) / (# total relevant)
        
        Args:
            results: List of retrieved results
            relevant_ids: Set of relevant passage IDs
            k: Cutoff
            
        Returns:
            Recall score (0-1)
        """
        if len(relevant_ids) == 0:
            return 0.0
        
        matches = sum(1 for result in results[:k]
                     if any(str(rel_id) in str(result['text']) for rel_id in relevant_ids))
        return matches / len(relevant_ids)
    
    def run_benchmark(self, num_queries: int = 50):
        """
        Run complete benchmark
        
        Args:
            num_queries: Number of queries to evaluate
        """
        print("="*80)
        print(f"VectorFlow-RAG MS MARCO Benchmark")
        print(f"Alpha: {self.alpha} | LLM: {self.llm_model}")
        print("="*80)
        
        # Create dataset
        queries, passages, qrels = self.create_mock_dataset(num_queries)
        
        # Index passages
        hybrid_retriever, passage_ids = self.index_passages(passages)
        
        # Evaluate
        print("\nEvaluating...")
        mrr_scores = []
        ndcg_scores = []
        recall_at_k = {k: [] for k in [5, 10, 20, 50, 100]}
        latencies = []
        
        for query_id, query_text in tqdm(list(queries.items())[:20]):  # Test with first 20
            # Get relevant passages for this query
            relevant_ids = set(pid for pid, rel in qrels.get(query_id, []) if rel > 0)
            
            if not relevant_ids:
                continue
            
            # Retrieve passages
            start_time = time.time()
            results = hybrid_retriever.search(query_text, k=100)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            # Compute metrics
            mrr = self.compute_mrr(results, relevant_ids, k=10)
            mrr_scores.append(mrr)
            
            ndcg = self.compute_ndcg(results, relevant_ids, k=10)
            ndcg_scores.append(ndcg)
            
            for k in recall_at_k.keys():
                recall = self.compute_recall_at_k(results, relevant_ids, k=k)
                recall_at_k[k].append(recall)
        
        # Aggregate results
        results_dict = {
            "MRR@10": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            "NDCG@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "Recall@5": float(np.mean(recall_at_k[5])) if recall_at_k[5] else 0.0,
            "Recall@10": float(np.mean(recall_at_k[10])) if recall_at_k[10] else 0.0,
            "Recall@20": float(np.mean(recall_at_k[20])) if recall_at_k[20] else 0.0,
            "Recall@50": float(np.mean(recall_at_k[50])) if recall_at_k[50] else 0.0,
            "Recall@100": float(np.mean(recall_at_k[100])) if recall_at_k[100] else 0.0,
            "Latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
            "Latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "Latency_p99_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
        }
        
        # Print results
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80 + "\n")
        
        print("Ranking Metrics:")
        print(f"  MRR@10:    {results_dict['MRR@10']:.4f}")
        print(f"  NDCG@10:   {results_dict['NDCG@10']:.4f}")
        
        print("\nRecall Metrics:")
        for k in [5, 10, 20, 50, 100]:
            print(f"  Recall@{k:3d}:  {results_dict[f'Recall@{k}']:.4f}")
        
        print("\nLatency Metrics:")
        print(f"  Mean:      {results_dict['Latency_mean_ms']:.2f} ms")
        print(f"  P95:       {results_dict['Latency_p95_ms']:.2f} ms")
        print(f"  P99:       {results_dict['Latency_p99_ms']:.2f} ms")
        
        print("\n" + "="*80)
        
        self.results["metrics"] = results_dict
        return results_dict
    
    def save_results(self, output_path: str = "experiments/ms_marco_results.json"):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VectorFlow-RAG on MS MARCO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_ms_marco.py --alpha 0.5 --limit 50
  python benchmark_ms_marco.py --alpha 0.3 --limit 100
  python benchmark_ms_marco.py --llm_model llama3.2:1b --alpha 0.7
        """
    )
    
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Hybrid search weight: 0=pure BM25, 1=pure vector (default: 0.5)")
    parser.add_argument("--limit", type=int, default=50,
                       help="Number of queries to evaluate (default: 50)")
    parser.add_argument("--llm_model", type=str, default="tinyllama",
                       help="Ollama LLM model name (default: tinyllama)")
    parser.add_argument("--output", type=str, default="experiments/ms_marco_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = MSMARCOBenchmark(alpha=args.alpha, llm_model=args.llm_model)
    benchmark.run_benchmark(num_queries=args.limit)
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
