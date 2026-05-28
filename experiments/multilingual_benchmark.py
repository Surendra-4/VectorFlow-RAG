# experiments/multilingual_benchmark.py

"""
Multilingual retrieval benchmark with versioned artifacts (Phase 11).

Runs the golden corpus through one or more profiles and reports per-language
Recall@k and MRR for monolingual, cross-lingual, and code-switching query
sets. Writes timestamped JSON + CSV artifacts under experiments/artifacts/
so model-to-model and over-time comparisons are reproducible.

Usage:
    python -m experiments.multilingual_benchmark --profiles english multilingual
    python -m experiments.multilingual_benchmark --k 5

The English regression gate: the script prints English-subset Recall@1 for
every profile so a reviewer can confirm the multilingual profile doesn't
degrade English retrieval below the english-profile baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def _build_pipeline(profile: str, index_dir: str):
    """Construct a pipeline under a given profile with caching/expansion off."""
    from src.config import Settings
    from src.rag_pipeline import RAGPipeline

    settings = Settings(profile=profile)
    return RAGPipeline(index_dir=index_dir, settings=settings, enable_cache=False)


def _recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    top = retrieved_ids[:k]
    hit = any(r in top for r in relevant_ids)
    return 1.0 if hit else 0.0


def _reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def run_profile(profile: str, k: int) -> Dict:
    from experiments.multilingual_golden import CORPUS, all_query_sets, english_subset_queries

    with tempfile.TemporaryDirectory(prefix=f"mlbench_{profile}_") as td:
        rag = _build_pipeline(profile, td)

        # Ingest the corpus. We pass explicit doc ids via metadata so we can
        # map retrieval hits back to golden ids by document_name.
        texts = [text for (_id, _lang, text) in CORPUS]
        metadatas = [
            {"document_name": _id, "source_path": f"golden://{_id}", "gold_lang": _lang}
            for (_id, _lang, _text) in CORPUS
        ]
        t0 = time.perf_counter()
        rag.ingest_documents(texts, metadatas=metadatas, reset=True)
        ingest_ms = (time.perf_counter() - t0) * 1000

        def retrieved_ids_for(query: str) -> List[str]:
            results = rag.search(query, k=k)
            # document_name carries the golden id we set above.
            return [r.get("document_name") for r in results]

        per_set: Dict[str, Dict] = {}
        latencies: List[float] = []
        for set_name, queries in all_query_sets().items():
            recalls, rrs = [], []
            per_lang: Dict[str, List[float]] = {}
            for query, lang, relevant in queries:
                qt = time.perf_counter()
                got = retrieved_ids_for(query)
                latencies.append((time.perf_counter() - qt) * 1000)
                r = _recall_at_k(got, relevant, k)
                rr = _reciprocal_rank(got, relevant)
                recalls.append(r)
                rrs.append(rr)
                per_lang.setdefault(lang, []).append(r)
            per_set[set_name] = {
                "recall_at_k": round(statistics.mean(recalls), 4) if recalls else 0.0,
                "mrr": round(statistics.mean(rrs), 4) if rrs else 0.0,
                "n_queries": len(queries),
                "per_language_recall": {
                    lng: round(statistics.mean(vals), 4) for lng, vals in per_lang.items()
                },
            }

        # English regression gate metric.
        eng_recalls = []
        for query, _lang, relevant in english_subset_queries():
            eng_recalls.append(_recall_at_k(retrieved_ids_for(query), relevant, k))
        english_recall_at_k = round(statistics.mean(eng_recalls), 4) if eng_recalls else 0.0

        return {
            "profile": profile,
            "embedder_model": rag.embedder.model_name,
            "k": k,
            "ingest_ms": round(ingest_ms, 1),
            "query_latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
            "english_subset_recall_at_k": english_recall_at_k,
            "query_sets": per_set,
        }


def main():
    parser = argparse.ArgumentParser(description="Multilingual retrieval benchmark")
    parser.add_argument(
        "--profiles", nargs="+", default=["english", "multilingual"],
        help="Profiles to benchmark (english, multilingual, multilingual_quality)",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    results = []
    for profile in args.profiles:
        print(f"\n=== Benchmarking profile: {profile} ===")
        try:
            res = run_profile(profile, args.k)
            results.append(res)
            print(f"  embedder: {res['embedder_model']}")
            print(f"  English subset Recall@{args.k}: {res['english_subset_recall_at_k']}")
            for set_name, stats in res["query_sets"].items():
                print(f"  {set_name}: Recall@{args.k}={stats['recall_at_k']} MRR={stats['mrr']}")
        except Exception as exc:  # pragma: no cover - benchmark resilience
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results.append({"profile": profile, "error": f"{type(exc).__name__}: {exc}"})

    artifact = {
        "schema_version": 1,
        "timestamp": timestamp,
        "k": args.k,
        "profiles": results,
    }

    json_path = ARTIFACT_DIR / f"multilingual_benchmark_{timestamp}.json"
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = ARTIFACT_DIR / f"multilingual_benchmark_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["profile", "embedder_model", "k", "query_set", "recall_at_k", "mrr", "english_recall_at_k"])
        for res in results:
            if "error" in res:
                w.writerow([res["profile"], "ERROR", args.k, "-", "-", "-", res["error"]])
                continue
            for set_name, stats in res["query_sets"].items():
                w.writerow([
                    res["profile"], res["embedder_model"], res["k"], set_name,
                    stats["recall_at_k"], stats["mrr"], res["english_subset_recall_at_k"],
                ])

    print(f"\nArtifacts written:\n  {json_path}\n  {csv_path}")
    return artifact


if __name__ == "__main__":
    main()
