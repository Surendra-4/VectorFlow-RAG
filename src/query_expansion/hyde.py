# src/query_expansion/hyde.py

"""
HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

The LLM generates a short hypothetical answer document. The document's
embedding is then used as an additional vector-side retrieval query.
Documents tend to share vocabulary with their hypothetical answers more
than with their questions, which improves cosine similarity on hard
queries.

The hypothetical document is **never** shown to the user and **never**
fed back into the LLM as context — it's purely a retrieval signal.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Optional

from src.config import get_settings
from src.llm_client import OllamaClient
from src.logging_setup import get_logger
from src.query_expansion.base import ExpansionResult, sanitize_query

logger = get_logger(__name__)


_PROMPT_TEMPLATE = (
    "Write a short, plausible-sounding answer to the question below. "
    "Use 2-3 sentences. Do not refuse, do not say 'I don't know', and "
    "do not preface your answer. Use neutral, encyclopedic phrasing.\n\n"
    "Question: {query}\n\nAnswer:"
)


def _truncate(text: str, max_length: int) -> str:
    if max_length is None or len(text) <= max_length:
        return text
    return text[:max_length]


class HyDEExpander:
    """Generate hypothetical answer documents to use as vector queries."""

    name = "hyde"

    def __init__(
        self,
        llm: Optional[OllamaClient] = None,
        n_docs: Optional[int] = None,
        timeout_s: Optional[float] = None,
        max_query_length: Optional[int] = None,
        max_doc_length: Optional[int] = None,
    ):
        settings = get_settings()
        qcfg = settings.query_expansion

        self.n_docs = n_docs if n_docs is not None else qcfg.hyde_count
        self.timeout_s = timeout_s if timeout_s is not None else qcfg.timeout_s
        self.max_query_length = (
            max_query_length if max_query_length is not None else qcfg.max_query_length
        )
        self.max_doc_length = (
            max_doc_length if max_doc_length is not None else qcfg.max_variant_length
        )

        if self.n_docs < 1:
            raise ValueError(f"n_docs must be >= 1, got {self.n_docs}")

        if llm is not None:
            self._llm = llm
        else:
            model = qcfg.expansion_model or settings.llm.model
            self._llm = OllamaClient(model=model)

    def expand(self, query: str) -> ExpansionResult:
        sanitized = sanitize_query(query, max_length=self.max_query_length)
        if not sanitized:
            return ExpansionResult(strategy=self.name, error="empty query")

        docs: List[str] = []
        start = time.perf_counter()
        last_error: Optional[str] = None

        # Generate ``n_docs`` independently. Each call has its own timeout
        # so a single slow generation can't blow the total wall-clock budget.
        for i in range(self.n_docs):
            prompt = _PROMPT_TEMPLATE.format(query=sanitized)
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(
                        self._llm.generate,
                        prompt=prompt,
                        context=None,
                        max_tokens=256,
                        temperature=0.8,
                        stream=False,
                    )
                    output = future.result(timeout=self.timeout_s)
            except FuturesTimeoutError:
                last_error = "timeout"
                logger.warning("hyde[%d] timed out", i)
                continue
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("hyde[%d] failed: %s", i, exc)
                continue

            doc = (output or "").strip()
            if doc:
                docs.append(_truncate(doc, self.max_doc_length))

        elapsed = (time.perf_counter() - start) * 1000

        if not docs:
            return ExpansionResult(
                strategy=self.name,
                latency_ms=elapsed,
                error=last_error or "no usable hyde documents",
            )

        return ExpansionResult(
            strategy=self.name,
            hyde_documents=tuple(docs),
            latency_ms=elapsed,
        )
