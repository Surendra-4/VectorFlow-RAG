"""
Plot and compare embedding model results across multiple experiments
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_embedding_comparison_detailed():
    """Create comprehensive comparison plots"""
    
    # Load results
    with open("experiments/embedding_ablation_results.json") as f:
        results = json.load(f)
    
    models_data = results["models"]
    if not models_data:
        print("No results found")
        return
    
    # Extract data
    model_names = [v['model_label'] for v in models_data.values()]
    latencies = [v['avg_query_latency_ms'] for v in models_data.values()]
    quality = [v['retrieval_quality_acc'] for v in models_data.values()]
    sizes = [v['model_size_mb'] for v in models_data.values()]
    dims = [v['embedding_dim'] for v in models_data.values()]
    similarity = [v['semantic_similarity_score'] for v in models_data.values()]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Embedding Model Ablation Study - Comprehensive Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Latency Comparison
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.barh(model_names, latencies, color=colors, alpha=0.7)
    ax.set_xlabel('Latency (ms)')
    ax.set_title('Query Latency')
    ax.invert_yaxis()
    for i, v in enumerate(latencies):
        ax.text(v + 1, i, f'{v:.1f}ms', va='center')
    
    # 2. Quality Comparison
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.barh(model_names, quality, color=colors, alpha=0.7)
    ax.set_xlabel('Quality Score')
    ax.set_title('Retrieval Quality')
    ax.set_xlim([0, 1])
    ax.invert_yaxis()
    for i, v in enumerate(quality):
        ax.text(v + 0.02, i, f'{v:.1%}', va='center')
    
    # 3. Model Size
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.barh(model_names, sizes, color=colors, alpha=0.7)
    ax.set_xlabel('Size (MB)')
    ax.set_title('Model Size')
    ax.invert_yaxis()
    for i, v in enumerate(sizes):
        ax.text(v + 50, i, f'{v}MB', va='center')
    
    # 4. Embedding Dimension
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.barh(model_names, dims, color=colors, alpha=0.7)
    ax.set_xlabel('Dimensions')
    ax.set_title('Embedding Dimension')
    ax.invert_yaxis()
    for i, v in enumerate(dims):
        ax.text(v + 30, i, f'{v}D', va='center')
    
    # 5. Semantic Similarity
    ax = fig.add_subplot(gs[1, 1])
    bars = ax.barh(model_names, similarity, color=colors, alpha=0.7)
    ax.set_xlabel('Similarity Score')
    ax.set_title('Semantic Similarity')
    ax.invert_yaxis()
    for i, v in enumerate(similarity):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # 6. Speed vs Quality Scatter
    ax = fig.add_subplot(gs[1, 2])
    scatter = ax.scatter(latencies, quality, s=500, c=range(len(models_data)), 
                        cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
    for i, name in enumerate(model_names):
        ax.annotate(name.split('(')[0].strip(), 
                   (latencies[i], quality[i]), 
                   fontsize=9, ha='center', va='center')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Quality')
    ax.set_title('Speed-Quality Trade-off')
    ax.grid(True, alpha=0.3)
    
    # 7. Latency vs Size
    ax = fig.add_subplot(gs[2, 0])
    scatter = ax.scatter(sizes, latencies, s=500, c=range(len(models_data)),
                        cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
    for i, name in enumerate(model_names):
        ax.annotate(name.split('(')[0].strip(),
                   (sizes[i], latencies[i]),
                   fontsize=9, ha='center', va='center')
    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Size-Latency Trade-off')
    ax.grid(True, alpha=0.3)
    
    # 8. Quality vs Dimensions
    ax = fig.add_subplot(gs[2, 1])
    scatter = ax.scatter(dims, quality, s=500, c=range(len(models_data)),
                        cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
    for i, name in enumerate(model_names):
        ax.annotate(name.split('(')[0].strip(),
                   (dims[i], quality[i]),
                   fontsize=9, ha='center', va='center')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Quality')
    ax.set_title('Dimension-Quality Correlation')
    ax.grid(True, alpha=0.3)
    
    # 9. Summary Table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    # Find best in each category
    best_latency_idx = np.argmin(latencies)
    best_quality_idx = np.argmax(quality)
    best_size_idx = np.argmin(sizes)
    
    summary_text = f"""BEST IN CATEGORY:

⭐ Fastest:
   {model_names[best_latency_idx]}

⭐ Best Quality:
   {model_names[best_quality_idx]}

⭐ Smallest:
   {model_names[best_size_idx]}

RECOMMENDATION:
Use MPNet-Base for
balanced production
deployments
"""
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig("experiments/embedding_detailed_comparison.png", dpi=150, bbox_inches='tight')
    print("✓ Detailed comparison plot saved to experiments/embedding_detailed_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_embedding_comparison_detailed()
