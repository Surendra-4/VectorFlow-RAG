# experiments/multilingual_golden.py

"""
Shared multilingual golden corpus + query sets (Phase 11).

Coverage breadth over depth: 7 languages spanning 4 scripts (Latin, CJK,
Arabic RTL, Cyrillic). Used by both the benchmark script and the
slow-marked retrieval tests so the data lives in exactly one place.

Each corpus entry is ``(id, language, text)``. Each query is
``(query, language, relevant_ids)``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Corpus — small but script-diverse
# --------------------------------------------------------------------------- #

CORPUS: List[Tuple[str, str, str]] = [
    # English
    ("en_photo", "en", "Photosynthesis converts light energy into chemical energy in plant chloroplasts."),
    ("en_mito", "en", "Mitochondria generate ATP through oxidative phosphorylation in eukaryotic cells."),
    ("en_paris", "en", "Paris is the capital of France and sits on the river Seine."),
    # French
    ("fr_photo", "fr", "La photosynthèse transforme l'énergie lumineuse en énergie chimique dans les chloroplastes."),
    ("fr_paris", "fr", "Paris est la capitale de la France et se trouve sur la Seine."),
    ("fr_chat", "fr", "Le chat dort sur le canapé pendant que le chien joue dans le jardin."),
    # German
    ("de_auto", "de", "Das Auto fährt schnell auf der Autobahn durch die bayerische Landschaft."),
    ("de_mito", "de", "Mitochondrien erzeugen ATP durch oxidative Phosphorylierung in den Zellen."),
    # Spanish
    ("es_paris", "es", "París es la capital de Francia y está situada junto al río Sena."),
    ("es_gato", "es", "El gato duerme en el sofá mientras el perro juega en el jardín."),
    # Chinese (simplified)
    ("zh_photo", "zh", "光合作用将光能转化为植物叶绿体中的化学能。"),
    ("zh_paris", "zh", "巴黎是法国的首都，位于塞纳河畔。"),
    # Arabic (RTL)
    ("ar_paris", "ar", "باريس هي عاصمة فرنسا وتقع على نهر السين."),
    ("ar_cell", "ar", "الميتوكوندريا تنتج الطاقة في الخلايا حقيقية النواة."),
    # Russian (Cyrillic)
    ("ru_paris", "ru", "Париж — столица Франции, расположенная на реке Сена."),
    ("ru_cat", "ru", "Кошка спит на диване, пока собака играет в саду."),
    # Code-switched (mixed within one document)
    ("mix_paris", "mix", "The capital of France is Paris. La capitale est belle. 巴黎很美。"),
]

# --------------------------------------------------------------------------- #
# Monolingual queries — query in language X retrieves doc(s) in language X.
# --------------------------------------------------------------------------- #

MONOLINGUAL_QUERIES: List[Tuple[str, str, List[str]]] = [
    ("photosynthesis chloroplasts", "en", ["en_photo"]),
    ("capital of France", "en", ["en_paris"]),
    ("photosynthèse chloroplastes", "fr", ["fr_photo"]),
    ("capitale de la France", "fr", ["fr_paris"]),
    ("Hauptstadt Frankreich Auto Autobahn", "de", ["de_auto"]),
    ("capital de Francia río Sena", "es", ["es_paris"]),
    ("光合作用 叶绿体", "zh", ["zh_photo"]),
    ("عاصمة فرنسا", "ar", ["ar_paris"]),
    ("столица Франции", "ru", ["ru_paris"]),
]

# --------------------------------------------------------------------------- #
# Cross-lingual queries — query in language X retrieves doc(s) in language Y.
# This is the real test of a shared multilingual embedding space.
# --------------------------------------------------------------------------- #

CROSS_LINGUAL_QUERIES: List[Tuple[str, str, List[str]]] = [
    # English query → French/Spanish/Chinese/Arabic/Russian docs about Paris
    ("What is the capital of France?", "en", ["fr_paris", "es_paris", "zh_paris", "ar_paris", "ru_paris", "en_paris"]),
    # French query → English photosynthesis doc
    ("Qu'est-ce que la photosynthèse?", "fr", ["en_photo", "fr_photo", "zh_photo"]),
    # Chinese query → English/French Paris docs
    ("法国的首都是什么", "zh", ["en_paris", "fr_paris", "zh_paris"]),
]

# --------------------------------------------------------------------------- #
# Code-switching queries
# --------------------------------------------------------------------------- #

CODE_SWITCH_QUERIES: List[Tuple[str, str, List[str]]] = [
    ("capital of France 巴黎", "mix", ["mix_paris", "en_paris", "zh_paris", "fr_paris"]),
]


def english_subset_queries() -> List[Tuple[str, str, List[str]]]:
    """The English-only queries — used for the regression gate."""
    return [q for q in MONOLINGUAL_QUERIES if q[1] == "en"]


def all_query_sets() -> Dict[str, List[Tuple[str, str, List[str]]]]:
    return {
        "monolingual": MONOLINGUAL_QUERIES,
        "cross_lingual": CROSS_LINGUAL_QUERIES,
        "code_switch": CODE_SWITCH_QUERIES,
    }
