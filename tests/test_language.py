# tests/test_language.py

"""
Tests for advisory language detection (Phase 11).

Hard rule under test: detection is advisory only. These tests verify the
detector's behavior and graceful degradation — there is intentionally NO
test asserting that language drives retrieval, because it must never do so.
"""

from __future__ import annotations

import pytest

from src.language import LanguageDetector, get_language_detector, language_name


class TestLanguageDetector:
    @pytest.fixture
    def detector(self):
        return LanguageDetector()

    def test_detects_english(self, detector):
        lang = detector.detect("This is a reasonably long English sentence about cells.")
        assert lang == "en"

    def test_detects_french(self, detector):
        lang = detector.detect("Ceci est une phrase française assez longue pour être détectée.")
        assert lang == "fr"

    def test_short_text_returns_none(self, detector):
        # Below the min_chars guard — too noisy to classify.
        assert detector.detect("hi") is None
        assert detector.detect("RRF?") is None

    def test_empty_returns_none(self, detector):
        assert detector.detect("") is None
        assert detector.detect("   ") is None

    def test_custom_min_chars(self, detector):
        # With a low threshold a short string can classify; with the default
        # it returns None.
        assert detector.detect("Bonjour", min_chars=3) is not None
        assert detector.detect("Bonjour") is None  # default min_chars=20


class TestLanguageName:
    def test_known_codes(self):
        assert language_name("en") == "English"
        assert language_name("fr") == "French"
        assert language_name("zh") == "Chinese"
        assert language_name("ar") == "Arabic"

    def test_unknown_code_falls_back_to_code(self):
        assert language_name("xx") == "xx"

    def test_none_returns_none(self):
        assert language_name(None) is None


class TestSingleton:
    def test_get_language_detector_is_singleton(self):
        assert get_language_detector() is get_language_detector()


class TestMultiQueryLanguageHint:
    """The expansion language hint is opt-in and appends to the prompt only."""

    def _mock_llm(self, response: str):
        class _M:
            def __init__(self):
                self.last_prompt = None

            def generate(self, prompt, context=None, max_tokens=512, temperature=0.7, stream=True):
                self.last_prompt = prompt
                return response

        return _M()

    def test_hint_off_by_default_no_language_clause(self):
        from src.query_expansion.multi_query import MultiQueryExpander

        llm = self._mock_llm("variant one\nvariant two")
        ex = MultiQueryExpander(llm=llm, n_variants=2, language_hint=False)
        ex.expand("This is a sufficiently long French-free English query about cells.")
        assert "Write every phrasing in" not in llm.last_prompt

    def test_hint_on_adds_language_clause(self):
        from src.query_expansion.multi_query import MultiQueryExpander

        llm = self._mock_llm("phrase une\nphrase deux")
        ex = MultiQueryExpander(llm=llm, n_variants=2, language_hint=True)
        ex.expand("Ceci est une requête française assez longue pour la détection de langue.")
        assert "Write every phrasing in French." in llm.last_prompt

    def test_hint_on_short_query_omits_clause(self):
        from src.query_expansion.multi_query import MultiQueryExpander

        llm = self._mock_llm("a\nb")
        ex = MultiQueryExpander(llm=llm, n_variants=2, language_hint=True)
        # Too short to detect → no hint appended, no crash.
        ex.expand("cells")
        assert "Write every phrasing in" not in llm.last_prompt
