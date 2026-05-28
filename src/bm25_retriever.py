# src/bm25_retriever.py

"""
Sparse keyword retrieval using BM25.

Corpus forms (Phase 3.5):

* ``List[str]``            — legacy plain-text corpus (BC).
* ``List[Dict[str, Any]]`` — entries with at least ``"text"``, optionally
                             ``"chunk_id"`` and ``"metadata"``.

Tokenization (Phase 11):

* Text is NFC-normalized before tokenization. For ASCII this is a no-op, so
  the English path is byte-identical; for multilingual content it collapses
  composed/decomposed Unicode so equivalent strings tokenize identically.
* ``use_stemmer=True`` (English default) keeps the language-specific Snowball
  stemmer and English stopword list — unchanged behavior.
* ``use_stemmer=False`` (multilingual profile) tokenizes Unicode-aware
  without stemming or English stopwords, so no English morphology rules are
  applied to non-English tokens.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Union

import bm25s
import Stemmer

from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

CorpusEntry = Union[str, Dict[str, Any]]


def _nfc(text: str) -> str:
    """NFC-normalize. No-op for ASCII → English parity preserved."""
    return unicodedata.normalize("NFC", text)


class BM25Retriever:
    def __init__(
        self,
        corpus: Sequence[CorpusEntry],
        language: Optional[str] = None,
        use_stemmer: Optional[bool] = None,
    ):
        cfg = get_settings().bm25
        self.language = language or cfg.language
        self.use_stemmer = cfg.use_stemmer if use_stemmer is None else use_stemmer

        texts: List[str] = []
        chunk_ids: List[Optional[str]] = []
        metadatas: List[Optional[Dict[str, Any]]] = []

        for entry in corpus:
            if isinstance(entry, str):
                texts.append(entry)
                chunk_ids.append(None)
                metadatas.append(None)
            elif isinstance(entry, dict):
                if "text" not in entry:
                    raise ValueError("Corpus dict entries must include a 'text' key")
                texts.append(entry["text"])
                chunk_ids.append(entry.get("chunk_id"))
                metadatas.append(entry.get("metadata"))
            else:
                raise TypeError(f"Unsupported corpus entry type: {type(entry).__name__}")

        logger.debug(
            "Initializing BM25Retriever lang=%s use_stemmer=%s docs=%d",
            self.language, self.use_stemmer, len(texts),
        )

        # Stemmer only constructed when stemming is enabled.
        self.stemmer = Stemmer.Stemmer(self.language) if self.use_stemmer else None

        # Tokenize NFC-normalized text. Original text is kept for display.
        normalized = [_nfc(t) for t in texts]
        tokens = self._tokenize(normalized)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)

        # Public surface for legacy callers — list of (original) texts.
        self.corpus: List[str] = texts
        self._chunk_ids: List[Optional[str]] = chunk_ids
        self._metadatas: List[Optional[Dict[str, Any]]] = metadatas

    def _tokenize(self, text_or_texts):
        """
        Tokenize with the configured strategy.

        English (use_stemmer=True): Snowball stemmer + default English
        stopwords — identical to the pre-Phase-11 behavior.

        Multilingual (use_stemmer=False): no stemmer, no language-specific
        stopword removal — Unicode-aware whitespace tokenization that doesn't
        impose English morphology on other languages.
        """
        if self.use_stemmer:
            return bm25s.tokenize(text_or_texts, stemmer=self.stemmer, show_progress=False)
        return bm25s.tokenize(
            text_or_texts, stemmer=None, stopwords=None, show_progress=False
        )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k matches.

        Each result dict includes ``text``, ``score``, ``rank``,
        ``chunk_id`` (or None) and ``metadata`` (or None).
        """
        qt = self._tokenize(_nfc(query))
        docs, scores = self.retriever.retrieve(qt, k=k)
        results: List[Dict[str, Any]] = []
        for i in range(docs.shape[1]):
            idx = int(docs[0, i])
            score = float(scores[0, i])
            if score <= 0:
                continue
            results.append(
                {
                    "text": self.corpus[idx],
                    "score": score,
                    "rank": i + 1,
                    "chunk_id": self._chunk_ids[idx],
                    "metadata": self._metadatas[idx],
                }
            )
        return results


if __name__ == "__main__":
    c = ["Python language", "Machine learning", "Semantic search"]
    r = BM25Retriever(c)
    print(r.search("machine", 3))
