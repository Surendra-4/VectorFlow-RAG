"""
A/B Test Different Embedding Models

Compare embedding models on:
- Quality: Semantic similarity and retrieval performance
- Speed: Inference latency
- Resource Usage: Model size and memory
- Cost: Inference cost per query

This demonstrates understanding of ML system trade-offs.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever


class EmbeddingAblation:
    """
    Compare different embedding models systematically
    
    Models to test:
    1. all-MiniLM-L6-v2 (80MB, 384-dim) - Fast, lightweight
    2. all-mpnet-base-v2 (420MB, 768-dim) - Balanced
    3. BAAI/bge-large-en-v1.5 (1.3GB, 1024-dim) - SOTA quality
    """
    
    # Embedding models to compare
    MODELS = {
        "all-MiniLM-L6-v2": {
            "name": "MiniLM-L6 (80MB)",
            "size_mb": 80,
            "dimension": 384,
            "speed_class": "Fast",
            "quality_class": "Good",
        },
        "all-mpnet-base-v2": {
            "name": "MPNet-Base (420MB)",
            "size_mb": 420,
            "dimension": 768,
            "speed_class": "Moderate",
            "quality_class": "Very Good",
        },
        "BAAI/bge-large-en-v1.5": {
            "name": "BGE-Large (1.3GB)",
            "size_mb": 1340,
            "dimension": 1024,
            "speed_class": "Slow",
            "quality_class": "Excellent",
        },
    }
    
    def __init__(self, test_queries_file: str = None):
        """Initialize ablation study"""
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {},
        }
        self.test_queries_file = test_queries_file
    
    def create_test_corpus(self, num_docs: int = 100) -> Tuple[List[str], List[Dict]]:
        """
        Create test corpus of documents
        
        Args:
            num_docs: Number of test documents
            
        Returns:
            (documents, metadatas) tuples
        """
        print(f"Creating test corpus ({num_docs} documents)...")
        
        topics = [
            "machine learning algorithms",
            "deep learning neural networks",
            "natural language processing",
            "computer vision image classification",
            "semantic search embeddings",
            "retrieval augmented generation",
            "transformer attention mechanism",
            "BERT language model",
            "vector databases indexing",
            "information retrieval ranking",
        ]
        
        templates = [
            "{topic} is a fundamental concept in AI and ML systems.",
            "Applications of {topic} include search engines and recommendation systems.",
            "Recent advances in {topic} have improved model accuracy significantly.",
            "{topic} requires understanding both theory and practical implementation.",
            "The {topic} field has evolved rapidly over the past 5 years.",
            "Researchers are actively working on improving {topic} efficiency.",
            "{topic} combines mathematical foundations with engineering practices.",
            "Understanding {topic} is essential for modern machine learning.",
        ]
        
        documents = []
        metadatas = []
        
        for i in range(num_docs):
            topic = topics[i % len(topics)]
            template = templates[i % len(templates)]
            doc_text = template.format(topic=topic)
            
            documents.append(doc_text)
            metadatas.append({
                "doc_id": i,
                "topic": topic,
                "source": "test_corpus"
            })
        
        return documents, metadatas
    
    def create_test_queries(self) -> List[Dict]:
        """
        Create test queries for evaluation
        
        Returns:
            List of query dicts with text and expected_topics
        """
        print("Creating test queries...")
        
        queries = [
            {
                "text": "How do machine learning algorithms work?",
                "expected_topics": ["machine learning algorithms"],
                "type": "conceptual"
            },
            {
                "text": "What is semantic search and embeddings?",
                "expected_topics": ["semantic search embeddings"],
                "type": "conceptual"
            },
            {
                "text": "Explain deep learning neural networks",
                "expected_topics": ["deep learning neural networks"],
                "type": "educational"
            },
            {
                "text": "How is natural language processing used?",
                "expected_topics": ["natural language processing"],
                "type": "application"
            },
            {
                "text": "What are vector databases?",
                "expected_topics": ["vector databases indexing"],
                "type": "technical"
            },
            {
                "text": "Retrieval augmented generation explained",
                "expected_topics": ["retrieval augmented generation"],
                "type": "educational"
            },
            {
                "text": "How do transformers use attention?",
                "expected_topics": ["transformer attention mechanism"],
                "type": "technical"
            },
            {
                "text": "BERT and language models",
                "expected_topics": ["BERT language model"],
                "type": "technical"
            },
            {
                "text": "How to rank documents in information retrieval?",
                "expected_topics": ["information retrieval ranking"],
                "type": "technical"
            },
            {
                "text": "Computer vision and image classification",
                "expected_topics": ["computer vision image classification"],
                "type": "application"
            },
        ]
        
        return queries
    
    def benchmark_model(self, 
                       model_name: str, 
                       documents: List[str],
                       queries: List[Dict]) -> Dict:
        """
        Benchmark single embedding model
        
        Args:
            model_name: Model identifier
            documents: List of test documents
            queries: List of test queries
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*80}")
        print(f"Testing: {self.MODELS[model_name]['name']}")
        print(f"{'='*80}")
        
        try:
            # Load model
            print("Loading model...")
            start_load = time.time()
            embedder = Embedder(model_name=model_name, device="cpu")
            load_time = time.time() - start_load
            print(f"✓ Loaded in {load_time:.2f}s")
            
            # Embed documents
            print("Embedding documents...")
            start_doc = time.time()
            doc_embeddings = embedder.encode(documents, show_progress=True)
            doc_time = time.time() - start_doc
            doc_latency_per = doc_time / len(documents) * 1000
            
            # Create vector store
            vector_store = VectorStore(f"indices\\ablation_{model_name.replace('/', '_')}")
            vector_store.create_collection(reset=True)
            vector_store.add_documents(
                texts=documents,
                embeddings=doc_embeddings.tolist(),
                ids=[f"doc_{i}" for i in range(len(documents))]
            )
            
            # Create hybrid retriever
            bm25_retriever = BM25Retriever(corpus=documents)
            hybrid_retriever = HybridRetriever(
                embedder=embedder,
                vector_store=vector_store,
                bm25_retriever=bm25_retriever,
                alpha=0.5
            )
            
            # Test query latencies
            print("\nTesting query latencies...")
            query_latencies = []
            retrieval_quality = []
            
            for query_dict in tqdm(queries, desc="Processing queries"):
                query_text = query_dict["text"]
                expected_topics = query_dict["expected_topics"]
                
                # Measure retrieval time
                start_query = time.time()
                results = hybrid_retriever.search(query_text, k=10)
                query_time = (time.time() - start_query) * 1000
                query_latencies.append(query_time)
                
                # Check if top results contain expected topics
                top_texts = [r['text'] for r in results[:3]]
                topic_found = any(
                    any(topic.lower() in text.lower() for topic in expected_topics)
                    for text in top_texts
                )
                retrieval_quality.append(1.0 if topic_found else 0.0)
            
            # Calculate similarity metrics
            print("\nCalculating similarity metrics...")
            similarity_scores = []
            
            # Compare semantically related queries
            query_pairs = [
                ("machine learning algorithms", "deep learning"),
                ("semantic search", "vector embeddings"),
                ("neural networks", "deep learning"),
            ]
            
            for q1_text, q2_text in query_pairs:
                q1_emb = embedder.encode(q1_text)[0]
                q2_emb = embedder.encode(q2_text)[0]
                
                # Cosine similarity
                similarity = np.dot(q1_emb, q2_emb)
                similarity_scores.append(similarity)
            
            # Compile results
            results = {
                "model_name": model_name,
                "model_label": self.MODELS[model_name]['name'],
                "embedding_dim": embedder.dimension,
                "load_time_s": load_time,
                "doc_embedding_time_total_s": doc_time,
                "doc_embedding_latency_ms": doc_latency_per,
                "avg_query_latency_ms": np.mean(query_latencies),
                "p95_query_latency_ms": np.percentile(query_latencies, 95),
                "p99_query_latency_ms": np.percentile(query_latencies, 99),
                "retrieval_quality_acc": np.mean(retrieval_quality),
                "semantic_similarity_score": np.mean(similarity_scores),
                "model_size_mb": self.MODELS[model_name]['size_mb'],
                "speed_class": self.MODELS[model_name]['speed_class'],
                "quality_class": self.MODELS[model_name]['quality_class'],
            }
            
            print("\n✓ Benchmark complete for " + self.MODELS[model_name]['name'])
            return results
            
        except Exception as e:
            print(f"❌ Error benchmarking {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_full_ablation(self, num_docs: int = 100):
        """
        Run complete ablation study
        
        Args:
            num_docs: Number of test documents
        """
        print("\n" + "="*80)
        print("EMBEDDING MODEL ABLATION STUDY")
        print("="*80)
        
        # Create test data
        documents, metadatas = self.create_test_corpus(num_docs)
        queries = self.create_test_queries()
        
        print(f"\nTest Setup:")
        print(f"  Documents: {len(documents)}")
        print(f"  Queries: {len(queries)}")
        print(f"  Models: {len(self.MODELS)}")
        
        # Benchmark each model
        for model_name in self.MODELS.keys():
            result = self.benchmark_model(model_name, documents, queries)
            if result:
                self.results["models"][model_name] = result
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary comparison table"""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80 + "\n")
        
        models_data = self.results["models"]
        if not models_data:
            print("No results to display")
            return
        
        # Create comparison table
        print("PERFORMANCE COMPARISON")
        print("-" * 140)
        print(f"{'Model':<30} {'Dim':<6} {'Load':<8} {'Doc Lat':<10} {'Query Lat':<12} {'Quality':<10} {'Similarity':<12} {'Size':<10}")
        print("-" * 140)
        
        for model_name, data in models_data.items():
            if data is None:
                continue
            
            print(f"{data['model_label']:<30} "
                  f"{data['embedding_dim']:<6} "
                  f"{data['load_time_s']:<8.2f}s "
                  f"{data['doc_embedding_latency_ms']:<10.2f}ms "
                  f"{data['avg_query_latency_ms']:<12.2f}ms "
                  f"{data['retrieval_quality_acc']:<10.3f} "
                  f"{data['semantic_similarity_score']:<12.3f} "
                  f"{data['model_size_mb']:<10}MB")
        
        print("-" * 140)
        
        # Detailed metrics
        print("\n\nDETAILED METRICS")
        print("-" * 140)
        
        for model_name, data in models_data.items():
            if data is None:
                continue
            
            print(f"\n{data['model_label']}:")
            print(f"  Embedding Dimension:       {data['embedding_dim']} dims")
            print(f"  Model Size:                {data['model_size_mb']} MB")
            print(f"  Load Time:                 {data['load_time_s']:.2f} seconds")
            print(f"  Document Embedding:        {data['doc_embedding_latency_ms']:.2f} ms per doc")
            print(f"  Query Latency (mean):      {data['avg_query_latency_ms']:.2f} ms")
            print(f"  Query Latency (p95):       {data['p95_query_latency_ms']:.2f} ms")
            print(f"  Query Latency (p99):       {data['p99_query_latency_ms']:.2f} ms")
            print(f"  Retrieval Quality:         {data['retrieval_quality_acc']:.1%}")
            print(f"  Semantic Similarity:       {data['semantic_similarity_score']:.3f}")
        
        print("\n" + "="*80)
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        print("-" * 140)
        
        models_list = list(models_data.values())
        models_list = [m for m in models_list if m is not None]
        
        if models_list:
            fastest = min(models_list, key=lambda x: x['avg_query_latency_ms'])
            best_quality = max(models_list, key=lambda x: x['retrieval_quality_acc'])
            most_balanced = min(models_list, 
                              key=lambda x: abs((x['avg_query_latency_ms']/100 - x['retrieval_quality_acc'])))
            
            print(f"\n✅ Fastest Model:      {fastest['model_label']}")
            print(f"   - {fastest['avg_query_latency_ms']:.2f}ms per query")
            
            print(f"\n✅ Best Quality:       {best_quality['model_label']}")
            print(f"   - {best_quality['retrieval_quality_acc']:.1%} quality score")
            
            print(f"\n✅ Best Balance:       {most_balanced['model_label']}")
            print(f"   - {most_balanced['avg_query_latency_ms']:.2f}ms latency, {most_balanced['retrieval_quality_acc']:.1%} quality")
        
        print("\n" + "="*80)
    
    def plot_results(self, output_file: str = "experiments/embedding_comparison.png"):
        """Create visualization of results"""
        print("\nGenerating plots...")
        
        models_data = self.results["models"]
        if not models_data:
            print("No data to plot")
            return
        
        models_data = {k: v for k, v in models_data.items() if v is not None}
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Embedding Model Ablation Study', fontsize=16, fontweight='bold')
        
        model_labels = [v['model_label'] for v in models_data.values()]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models_data)]
        
        # 1. Query Latency
        ax = axes[0, 0]
        latencies = [v['avg_query_latency_ms'] for v in models_data.values()]
        ax.bar(range(len(model_labels)), latencies, color=colors, alpha=0.7)
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=15, ha='right')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Query Latency (Lower is Better)')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(latencies):
            ax.text(i, v + 1, f'{v:.1f}ms', ha='center', va='bottom')
        
        # 2. Retrieval Quality
        ax = axes[0, 1]
        quality = [v['retrieval_quality_acc'] for v in models_data.values()]
        ax.bar(range(len(model_labels)), quality, color=colors, alpha=0.7)
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=15, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Retrieval Quality (Higher is Better)')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(quality):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
        
        # 3. Model Size
        ax = axes[0, 2]
        sizes = [v['model_size_mb'] for v in models_data.values()]
        ax.bar(range(len(model_labels)), sizes, color=colors, alpha=0.7)
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=15, ha='right')
        ax.set_ylabel('Size (MB)')
        ax.set_title('Model Size')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(sizes):
            ax.text(i, v + 20, f'{v}MB', ha='center', va='bottom')
        
        # 4. Embedding Dimension
        ax = axes[1, 0]
        dims = [v['embedding_dim'] for v in models_data.values()]
        ax.bar(range(len(model_labels)), dims, color=colors, alpha=0.7)
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=15, ha='right')
        ax.set_ylabel('Dimensions')
        ax.set_title('Embedding Dimension')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(dims):
            ax.text(i, v + 20, f'{v}D', ha='center', va='bottom')
        
        # 5. Semantic Similarity
        ax = axes[1, 1]
        sim = [v['semantic_similarity_score'] for v in models_data.values()]
        ax.bar(range(len(model_labels)), sim, color=colors, alpha=0.7)
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=15, ha='right')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Semantic Similarity (Higher is Better)')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(sim):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 6. Trade-off Matrix
        ax = axes[1, 2]
        # Speed vs Quality trade-off
        latencies_norm = np.array(latencies) / max(latencies)
        quality_norm = np.array(quality)
        
        scatter = ax.scatter(latencies_norm, quality_norm, s=300, c=range(len(models_data)), 
                            cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
        
        for i, label in enumerate(model_labels):
            ax.annotate(label, (latencies_norm[i], quality_norm[i]), 
                       fontsize=9, ha='center', va='center')
        
        ax.set_xlabel('Latency (Normalized)')
        ax.set_ylabel('Quality Score')
        ax.set_title('Speed-Quality Trade-off')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {output_file}")
    
    def save_results(self, output_file: str = "experiments/embedding_ablation_results.json"):
        """Save results to JSON"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
        print(f"✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="A/B test embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python embedding_ablation.py
  python embedding_ablation.py --num_docs 200
        """
    )
    
    parser.add_argument("--num_docs", type=int, default=100,
                       help="Number of test documents (default: 100)")
    parser.add_argument("--output", type=str, default="experiments/embedding_ablation_results.json",
                       help="Output JSON file")
    parser.add_argument("--plot", type=str, default="experiments/embedding_comparison.png",
                       help="Output plot file")
    
    args = parser.parse_args()
    
    # Run ablation
    ablation = EmbeddingAblation()
    ablation.run_full_ablation(num_docs=args.num_docs)
    ablation.plot_results(output_file=args.plot)
    ablation.save_results(output_file=args.output)


if __name__ == "__main__":
    main()
