# C:\Users\use\OneDrive\Desktop\VectorFlow-RAG\experiments\visualize_results.py

"""
Visualize MS MARCO benchmark results
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_metrics(results: dict):
    """Create plots for benchmark results"""
    metrics = results['metrics']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"VectorFlow-RAG Benchmark Results (α={results['config']['alpha']})", fontsize=16)
    
    # 1. Ranking Metrics
    ax = axes[0, 0]
    ranking_metrics = ['MRR@10', 'NDCG@10']
    ranking_values = [metrics[m] for m in ranking_metrics]
    ax.bar(ranking_metrics, ranking_values, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Score')
    ax.set_title('Ranking Metrics (higher is better)')
    ax.set_ylim([0, 1])
    for i, v in enumerate(ranking_values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # 2. Recall@k
    ax = axes[0, 1]
    recall_ks = [5, 10, 20, 50, 100]
    recall_values = [metrics[f'Recall@{k}'] for k in recall_ks]
    ax.plot(recall_ks, recall_values, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('k')
    ax.set_ylabel('Recall')
    ax.set_title('Recall@k Curve')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. Latency Distribution
    ax = axes[1, 0]
    latency_metrics = ['Mean', 'P95', 'P99']
    latency_values = [
        metrics['Latency_mean_ms'],
        metrics['Latency_p95_ms'],
        metrics['Latency_p99_ms']
    ]
    ax.bar(latency_metrics, latency_values, color=['#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Query Latency Distribution')
    for i, v in enumerate(latency_values):
        ax.text(i, v + 1, f'{v:.1f}ms', ha='center')
    
    # 4. Config Info
    ax = axes[1, 1]
    ax.axis('off')
    config_text = f"""
Configuration:
  Alpha (hybrid weight): {results['config']['alpha']}
  LLM Model: {results['config']['llm_model']}
  Timestamp: {results['config']['timestamp']}

Key Results:
  MRR@10: {metrics['MRR@10']:.4f}
  NDCG@10: {metrics['NDCG@10']:.4f}
  Recall@10: {metrics['Recall@10']:.4f}
  
Latency:
  Mean: {metrics['Latency_mean_ms']:.2f}ms
  P95: {metrics['Latency_p95_ms']:.2f}ms
    """
    ax.text(0.1, 0.5, config_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load results
    results = load_results("experiments/ms_marco_results.json")
    
    # Plot
    fig = plot_metrics(results)
    
    # Save
    plt.savefig("experiments/benchmark_results.png", dpi=150, bbox_inches='tight')
    print("✓ Saved plot to experiments/benchmark_results.png")
    
    # Show
    plt.show()
