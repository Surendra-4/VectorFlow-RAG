# src/rrf.py

"""
Reciprocal Rank Fusion (Cormack et al., CIKM 2009).

Given multiple ranked lists of items, RRF combines them into a single
ranking using a rank-based score:

    RRF(d) = Σ_i  1 / (k_rrf + rank_i(d))

where ``rank_i(d)`` is the 1-indexed position of item ``d`` in modality
``i``. Items missing from a modality contribute nothing for that modality.

Why rank-based fusion (vs. linear score combination):

* Scale-invariant: works across heterogeneous retrievers (BM25 / cosine /
  cross-encoder) without per-modality score normalization.
* Hyperparameter-light: only ``k_rrf`` (default 60 from the original paper);
  results are robust over a wide range.
* Empirically strong: TREC studies show RRF beats hand-tuned linear
  fusion on most datasets without per-dataset tuning.

This module is deliberately decoupled from any retriever — it operates on
plain item identifiers so it can be unit-tested in isolation and reused
for arbitrary fusion problems.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = 60,
) -> List[Tuple[Hashable, float]]:
    """
    Fuse multiple ranked lists into a single ranking via RRF.

    Args:
        ranked_lists: A sequence of ranked lists. Each inner list is in
            descending relevance order — the first item is rank 1.
        k: RRF dampening constant. The CIKM 2009 default is 60. Larger
            values smooth differences between top ranks; smaller values
            sharpen them. Values <= 0 are rejected.

    Returns:
        A list of ``(item, fused_score)`` tuples sorted by ``fused_score``
        descending. Items appearing in any input list are included exactly
        once. Stable order across ties is determined by first-appearance
        order across the input lists (deterministic, reproducible).

    Raises:
        ValueError: if ``k <= 0``.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be > 0, got {k}")

    scores: Dict[Hashable, float] = {}
    first_seen: Dict[Hashable, int] = {}

    counter = 0
    for ranked in ranked_lists:
        for rank_zero_based, item in enumerate(ranked):
            rank = rank_zero_based + 1  # 1-indexed
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in first_seen:
                first_seen[item] = counter
                counter += 1

    # Sort by score desc, then by first-appearance order for deterministic ties.
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def rrf_with_ranks(
    ranked_lists: Mapping[str, Sequence[Hashable]],
    k: int = 60,
) -> List[Dict[str, object]]:
    """
    RRF variant that also reports each item's per-modality rank.

    Useful for retrieval transparency — the caller can see whether a
    candidate was a vector hit, a BM25 hit, or both.

    Args:
        ranked_lists: A mapping of modality name → ranked list. Each ranked
            list is in descending relevance order (first item is rank 1).
        k: RRF dampening constant.

    Returns:
        A list of dicts ``{"item", "score", "ranks"}`` where ``ranks`` is a
        ``{modality_name: rank}`` mapping with rank 1-indexed (only modalities
        in which the item appeared are included).

        Sorted by score descending; deterministic on ties.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be > 0, got {k}")

    scores: Dict[Hashable, float] = {}
    ranks: Dict[Hashable, Dict[str, int]] = {}
    first_seen: Dict[Hashable, int] = {}

    counter = 0
    for modality, ranked in ranked_lists.items():
        for rank_zero_based, item in enumerate(ranked):
            rank = rank_zero_based + 1
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(item, {})[modality] = rank
            if item not in first_seen:
                first_seen[item] = counter
                counter += 1

    fused = [
        {"item": item, "score": scores[item], "ranks": ranks[item]}
        for item in scores
    ]
    fused.sort(key=lambda r: (-r["score"], first_seen[r["item"]]))
    return fused
