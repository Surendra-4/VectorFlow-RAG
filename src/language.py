# src/language.py

"""
Advisory language detection (Phase 11).

**Strict scope**: detection output is *advisory metadata only*. It NEVER
drives retrieval routing, model selection, or any branching that affects
which chunks are returned. Multilingual retrieval is model-driven (shared
embedding space); language tags are for observability, filtering UIs, and
expansion-prompt hinting.

Backend: ``langid`` (pure Python, ~3 MB model, no native build). Lazy-imported
so the dependency is only loaded when detection is actually requested.
"""

from __future__ import annotations

from typing import Optional

from src.logging_setup import get_logger

logger = get_logger(__name__)


class LanguageDetector:
    """Thin wrapper over langid with graceful degradation."""

    def __init__(self) -> None:
        self._identifier = None  # lazy

    def _ensure_loaded(self) -> bool:
        if self._identifier is not None:
            return True
        try:
            import langid

            self._identifier = langid
            return True
        except Exception as exc:  # pragma: no cover - dependency missing
            logger.warning("langid unavailable; language detection disabled: %s", exc)
            return False

    def detect(self, text: str, *, min_chars: int = 20) -> Optional[str]:
        """
        Return an ISO 639-1 language code (e.g. "en", "fr", "zh"), or ``None``.

        ``None`` is returned when:
          * text is too short to classify reliably (< ``min_chars``)
          * langid is unavailable
          * detection raises for any reason

        Short-text guard matters: language detection on a 3-word query is
        noisy. Callers treat ``None`` as "unknown" and degrade gracefully.
        """
        if not text or len(text.strip()) < min_chars:
            return None
        if not self._ensure_loaded():
            return None
        try:
            lang, _confidence = self._identifier.classify(text)
            return lang
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("language detection failed: %s", exc)
            return None


# Process-wide singleton — the langid model is read-only after load.
_DETECTOR: Optional[LanguageDetector] = None


def get_language_detector() -> LanguageDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = LanguageDetector()
    return _DETECTOR


# Human-readable names for the most common codes — used to build expansion
# prompt hints ("Generate variants in French"). Unknown codes fall back to
# the raw code, which is still a usable hint for capable LLMs.
_LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish", "pl": "Polish", "sv": "Swedish",
}


def language_name(code: Optional[str]) -> Optional[str]:
    """Map an ISO code to an English language name for prompt hints."""
    if not code:
        return None
    return _LANG_NAMES.get(code, code)
