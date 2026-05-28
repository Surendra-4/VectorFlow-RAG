# src/retrieval_trace.py

"""
Structured retrieval trace — substrate for future observability/debugging.

Off by default, populated only when ``RAGPipeline.search(..., return_trace=True)``
is called or when ``settings.observability.collect_trace`` (future) is set.

Captures every step in the retrieval lifecycle:

* original query
* expanded queries + HyDE documents
* per-strategy raw candidate lists
* RRF-fused intermediate ranking
* reranker decisions (when enabled)
* final selected chunks
* per-stage timings

This is intentionally a passive data carrier — no logic. Phase 9
(monitoring dashboard) will consume these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StrategyCandidates:
    """Raw retrieval output from one strategy (e.g. one query variant on one modality)."""

    label: str                                 # e.g. "vector:original", "bm25:variant_2", "vector:hyde_0"
    modality: str                              # "vector" | "bm25"
    source_query: str                          # the query or HyDE doc used
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class RetrievalTrace:
    """All structured signals from a single retrieval call."""

    original_query: str = ""
    sanitized_query: str = ""

    # Expansion stage
    expanded_queries: List[str] = field(default_factory=list)
    hyde_documents: List[str] = field(default_factory=list)
    strategies_used: List[str] = field(default_factory=list)
    expansion_errors: List[str] = field(default_factory=list)
    expansion_latency_ms: float = 0.0

    # Retrieval stage
    per_strategy: List[StrategyCandidates] = field(default_factory=list)
    fused_pool: List[Dict[str, Any]] = field(default_factory=list)
    fusion_latency_ms: float = 0.0

    # Rerank stage (only populated if reranker active)
    reranker_used: bool = False
    rerank_input_size: int = 0
    rerank_output_size: int = 0
    rerank_latency_ms: float = 0.0

    # Final
    final_results: List[Dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0

    # Cache observability (Phase 6)
    from_cache: bool = False
    cache_stats: Dict[str, Any] = field(default_factory=dict)

    # Free-form bag for caller-supplied annotations (backend, etc.)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON logging / Phase 9 dashboard consumption."""
        return {
            "original_query": self.original_query,
            "sanitized_query": self.sanitized_query,
            "expanded_queries": list(self.expanded_queries),
            "hyde_documents_count": len(self.hyde_documents),
            "strategies_used": list(self.strategies_used),
            "expansion_errors": list(self.expansion_errors),
            "expansion_latency_ms": self.expansion_latency_ms,
            "per_strategy": [
                {
                    "label": s.label,
                    "modality": s.modality,
                    "source_query": s.source_query,
                    "candidate_count": len(s.candidates),
                    "latency_ms": s.latency_ms,
                }
                for s in self.per_strategy
            ],
            "fused_pool_size": len(self.fused_pool),
            "fusion_latency_ms": self.fusion_latency_ms,
            "reranker_used": self.reranker_used,
            "rerank_input_size": self.rerank_input_size,
            "rerank_output_size": self.rerank_output_size,
            "rerank_latency_ms": self.rerank_latency_ms,
            "final_result_count": len(self.final_results),
            "total_latency_ms": self.total_latency_ms,
            "from_cache": self.from_cache,
            "cache_stats": dict(self.cache_stats),
            "extras": dict(self.extras),
        }
