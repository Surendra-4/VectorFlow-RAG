"""
Compare VectorFlow-RAG performance across different alpha values
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_all_results(results_dir: str = "experiments") -> dict:
    """Load all alpha comparison results"""
    all_results = {}
    
    for filepath in Path(results_dir).glob("results_alpha_*.json"):
        alpha_str = filepath.stem.replace("results_alpha_", "")
        alpha = float(alpha_str)
        
        with open(filepath, 'r') as f:
            all_results[alpha] = json.load(f)
    
    return dict(sorted(all_results.items()))

def plot_comparison(all_results: dict):
    """Plot comparison across alphas"""
    alphas = list(all_results.keys())
    
    # Extract metrics
    mrr = [all_results[a]['metrics']['MRR@10'] for a in alphas]
    ndcg = [all_results[a]['metrics']['NDCG@10'] for a in alphas]
    recall10 = [all_results[a]['metrics']['Recall@10'] for a in alphas]
    latency = [all_results[a]['metrics']['Latency_mean_ms'] for a in alphas]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VectorFlow-RAG: Hybrid Search Alpha Comparison', fontsize=16)
    
    # MRR
    ax = axes[0, 0]
    ax.plot(alphas, mrr, marker='o', linewidth=2, markersize=8, label='MRR@10')
    ax.set_xlabel('Alpha (0=BM25, 1=Vector)')
    ax.set_ylabel('MRR@10')
    ax.set_title('Mean Reciprocal Rank')
    ax.grid(True, alpha=0.3)
    
    # NDCG
    ax = axes[0, 1]
    ax.plot(alphas, ndcg, marker='s', linewidth=2, markersize=8, label='NDCG@10', color='orange')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('NDCG@10')
    ax.set_title('Normalized Discounted Cumulative Gain')
    ax.grid(True, alpha=0.3)
    
    # Recall@10
    ax = axes[1, 0]
    ax.plot(alphas, recall10, marker='^', linewidth=2, markersize=8, label='Recall@10', color='green')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Recall@10')
    ax.set_title('Recall at Top 10')
    ax.grid(True, alpha=0.3)
    
    # Latency
    ax = axes[1, 1]
    ax.plot(alphas, latency, marker='d', linewidth=2, markersize=8, label='Latency (ms)', color='red')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Query Latency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load all results
    all_results = load_all_results()
    
    if not all_results:
        print("No results found. Run benchmark_ms_marco.py with different alphas first.")
        exit(1)
    
    print(f"Loaded results for alphas: {list(all_results.keys())}")
    
    # Plot comparison
    fig = plot_comparison(all_results)
    
    # Save
    plt.savefig("experiments/alpha_comparison.png", dpi=150, bbox_inches='tight')
    print("✓ Saved plot to experiments/alpha_comparison.png")
    
    plt.show()
