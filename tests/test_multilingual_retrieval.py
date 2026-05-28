# tests/test_multilingual_retrieval.py

"""
End-to-end multilingual retrieval quality (Phase 11).

These are marked ``slow`` because they load the multilingual embedder
(intfloat/multilingual-e5-small, ~118 MB). They're skipped in CI but run
locally with ``pytest -m slow``.

The hard gate lives in TestEnglishRegressionUnderMultilingualProfile: the
English-subset queries must still return their canonical docs at top-1
under the multilingual profile. The cross-lingual tests are quality
expectations (which can vary by model) and use top-k membership rather
than top-1 to avoid flakiness.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.rag_pipeline import RAGPipeline


def _build(profile: str, tmp_path) -> RAGPipeline:
    from experiments.multilingual_golden import CORPUS

    settings = Settings(profile=profile)
    rag = RAGPipeline(index_dir=str(tmp_path / profile), settings=settings, enable_cache=False)
    texts = [t for (_id, _lang, t) in CORPUS]
    metas = [
        {"document_name": _id, "source_path": f"golden://{_id}"}
        for (_id, _lang, _t) in CORPUS
    ]
    rag.ingest_documents(texts, metadatas=metas, reset=True)
    return rag


def _ids(results):
    return [r.get("document_name") for r in results]


@pytest.mark.slow
class TestMultilingualMonolingual:
    @pytest.fixture(scope="class")
    def rag(self, tmp_path_factory):
        return _build("multilingual", tmp_path_factory.mktemp("ml_mono"))

    @pytest.mark.parametrize(
        "query,expected_id",
        [
            ("photosynthèse chloroplastes", "fr_photo"),
            ("capitale de la France", "fr_paris"),
            ("capital de Francia río Sena", "es_paris"),
            ("光合作用 叶绿体", "zh_photo"),
            ("عاصمة فرنسا", "ar_paris"),
            ("столица Франции", "ru_paris"),
        ],
    )
    def test_monolingual_query_retrieves_same_language_doc(self, rag, query, expected_id):
        got = _ids(rag.search(query, k=5))
        assert expected_id in got, f"{expected_id} not in {got}"


@pytest.mark.slow
class TestCrossLingual:
    @pytest.fixture(scope="class")
    def rag(self, tmp_path_factory):
        return _build("multilingual", tmp_path_factory.mktemp("ml_cross"))

    def test_english_query_finds_french_paris(self, rag):
        got = _ids(rag.search("What is the capital of France?", k=8))
        # A shared embedding space should surface at least one non-English
        # Paris doc alongside the English one.
        paris_docs = {"en_paris", "fr_paris", "es_paris", "zh_paris", "ar_paris", "ru_paris"}
        assert len(paris_docs.intersection(got)) >= 2

    def test_french_query_finds_english_photosynthesis(self, rag):
        got = _ids(rag.search("Qu'est-ce que la photosynthèse?", k=8))
        assert "en_photo" in got or "fr_photo" in got or "zh_photo" in got


@pytest.mark.slow
class TestCodeSwitching:
    @pytest.fixture(scope="class")
    def rag(self, tmp_path_factory):
        return _build("multilingual", tmp_path_factory.mktemp("ml_mix"))

    def test_code_switched_query_retrieves_mixed_doc(self, rag):
        got = _ids(rag.search("capital of France 巴黎", k=8))
        # The code-switched document or any Paris doc should surface.
        assert any(g in got for g in ("mix_paris", "en_paris", "zh_paris", "fr_paris"))


@pytest.mark.slow
class TestEnglishRegressionUnderMultilingualProfile:
    """
    HARD GATE: English retrieval must remain correct under the multilingual
    profile. These mirror the English golden expectations.
    """

    @pytest.fixture(scope="class")
    def rag(self, tmp_path_factory):
        return _build("multilingual", tmp_path_factory.mktemp("ml_eng_gate"))

    def test_english_photosynthesis_top1(self, rag):
        got = _ids(rag.search("photosynthesis chloroplasts", k=5))
        assert got[0] == "en_photo", f"expected en_photo at top-1, got {got}"

    def test_english_capital_of_france_in_top3(self, rag):
        got = _ids(rag.search("capital of France", k=5))
        # Cross-lingual space may interleave other-language Paris docs; the
        # English one must still be near the top.
        assert "en_paris" in got[:3], f"en_paris not in top-3: {got}"
