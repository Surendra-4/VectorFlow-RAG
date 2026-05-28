# tests/test_rrf.py

"""Tests for src.rrf."""

from __future__ import annotations

import math

import pytest

from src.rrf import reciprocal_rank_fusion, rrf_with_ranks


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self):
        result = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
        items = [item for item, _ in result]
        assert items == ["a", "b", "c"]

    def test_two_lists_intersection_boosted(self):
        # 'a' appears at rank 1 in both lists, 'b' only in list1, 'c' only in list2.
        # 'a' should beat both even though k=60.
        result = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=60)
        items = [item for item, _ in result]
        assert items[0] == "a"
        # 'b' and 'c' tie on score (each rank 1 in one list, missing in the other);
        # deterministic tiebreaker preserves first-appearance: 'b' before 'c'.
        assert items[1:] == ["b", "c"]

    def test_score_formula(self):
        # 'a' rank 1 in list1 only → 1/(60+1)
        # 'b' rank 2 in list1 only → 1/(60+2)
        result = reciprocal_rank_fusion([["a", "b"]], k=60)
        scores = {item: score for item, score in result}
        assert math.isclose(scores["a"], 1.0 / 61, rel_tol=1e-9)
        assert math.isclose(scores["b"], 1.0 / 62, rel_tol=1e-9)

    def test_score_summation(self):
        # 'x' appears at rank 2 in list1 and rank 1 in list2
        # → 1/62 + 1/61
        result = reciprocal_rank_fusion([["a", "x"], ["x", "b"]], k=60)
        scores = {item: score for item, score in result}
        expected = 1.0 / 62 + 1.0 / 61
        assert math.isclose(scores["x"], expected, rel_tol=1e-9)

    def test_intersection_beats_pure_top_one(self):
        # 'shared' is rank 5 in both lists; 'top' is rank 1 in only one.
        # rrf(shared) = 2/65 ≈ 0.0308
        # rrf(top)    = 1/61 ≈ 0.0164
        result = reciprocal_rank_fusion(
            [
                ["top", "x1", "x2", "x3", "shared"],
                ["y1", "y2", "y3", "y4", "shared"],
            ],
            k=60,
        )
        items = [item for item, _ in result]
        assert items[0] == "shared"

    def test_unique_items_only_appear_once(self):
        result = reciprocal_rank_fusion(
            [["a", "b", "a"], ["a", "b"]],
            k=60,
        )
        items = [item for item, _ in result]
        # 'a' appears twice in the first list — RRF should sum both contributions
        # but 'a' must appear only once in the output.
        assert items.count("a") == 1
        assert items.count("b") == 1

    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([], k=60) == []
        assert reciprocal_rank_fusion([[], []], k=60) == []

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], k=0)
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], k=-1)

    def test_smaller_k_sharpens_top_difference(self):
        # With smaller k, the gap between rank 1 and rank 10 grows.
        small = reciprocal_rank_fusion([["a"] + [f"d{i}" for i in range(9)]], k=1)
        big = reciprocal_rank_fusion([["a"] + [f"d{i}" for i in range(9)]], k=1000)
        gap_small = small[0][1] - small[-1][1]
        gap_big = big[0][1] - big[-1][1]
        assert gap_small > gap_big

    def test_results_sorted_descending(self):
        result = reciprocal_rank_fusion(
            [["a", "b", "c"], ["c", "b", "d"]], k=60
        )
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)


class TestRrfWithRanks:
    def test_reports_per_modality_ranks(self):
        result = rrf_with_ranks(
            {"vec": ["a", "b", "c"], "bm25": ["c", "a"]},
            k=60,
        )
        ranks_by_item = {r["item"]: r["ranks"] for r in result}
        assert ranks_by_item["a"] == {"vec": 1, "bm25": 2}
        assert ranks_by_item["b"] == {"vec": 2}
        assert ranks_by_item["c"] == {"vec": 3, "bm25": 1}

    def test_score_matches_basic_rrf(self):
        ranks = {"vec": ["a", "b"], "bm25": ["b", "a"]}
        with_ranks = rrf_with_ranks(ranks, k=60)
        without = reciprocal_rank_fusion(list(ranks.values()), k=60)
        scores_a = {r["item"]: r["score"] for r in with_ranks}
        scores_b = dict(without)
        for item in scores_a:
            assert math.isclose(scores_a[item], scores_b[item], rel_tol=1e-9)

    def test_only_appears_in_modalities_where_ranked(self):
        result = rrf_with_ranks({"a": ["x"], "b": ["y"]}, k=60)
        ranks_by_item = {r["item"]: r["ranks"] for r in result}
        assert ranks_by_item == {"x": {"a": 1}, "y": {"b": 1}}

    def test_deterministic_order_on_ties(self):
        # Two items with identical scores should sort by first-appearance order.
        result = rrf_with_ranks({"a": ["foo"], "b": ["bar"]}, k=60)
        items = [r["item"] for r in result]
        assert items == ["foo", "bar"]
