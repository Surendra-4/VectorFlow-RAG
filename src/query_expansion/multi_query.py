# src/query_expansion/multi_query.py

"""
Multi-query expansion via local LLM.

A single LLM call produces N alternative phrasings of the user's query.
Each phrasing is used as an additional retrieval query, fused with the
original through RRF in the retrieval pipeline.

Failures (timeout, parse error, empty output) are contained — the
strategy returns an empty result with ``error`` populated, and
retrieval proceeds with the original query alone.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Optional

from src.config import get_settings
from src.llm_client import OllamaClient
from src.logging_setup import get_logger
from src.query_expansion.base import ExpansionResult, dedupe_preserve_order, sanitize_query

logger = get_logger(__name__)


# The prompt is intentionally minimal and constrains output shape. We
# avoid chain-of-thought style instructions per the Phase 5 brief.
_PROMPT_TEMPLATE = (
    "You rewrite search queries. Produce {n} alternative phrasings of the "
    "following question. Each phrasing must preserve the original intent. "
    "Output ONE phrasing per line, no numbering, no commentary.\n\n"
    "Question: {query}\n\nPhrasings:"
)

# A reasonable leading-prefix pattern we strip from each line — LLMs
# sometimes include "1. ", "- ", "* ", or quote-marks despite the prompt.
_LEADING_NOISE = re.compile(r"^\s*([\-*•]|\d+[\.\)])\s*")


def _parse_variants(output: str, max_variant_length: int) -> List[str]:
    """Split LLM output into clean variant strings."""
    if not output:
        return []
    lines: List[str] = []
    for raw in output.splitlines():
        line = _LEADING_NOISE.sub("", raw).strip()
        # Strip surrounding quotes if the LLM added them.
        if len(line) >= 2 and line[0] in '"\'' and line[-1] == line[0]:
            line = line[1:-1].strip()
        if not line:
            continue
        if len(line) > max_variant_length:
            line = line[:max_variant_length]
        lines.append(line)
    return lines


class MultiQueryExpander:
    """Generate N rewritten variants of a query via local LLM."""

    name = "multi_query"

    def __init__(
        self,
        llm: Optional[OllamaClient] = None,
        n_variants: Optional[int] = None,
        timeout_s: Optional[float] = None,
        max_query_length: Optional[int] = None,
        max_variant_length: Optional[int] = None,
        language_hint: Optional[bool] = None,
    ):
        settings = get_settings()
        qcfg = settings.query_expansion

        self.n_variants = n_variants if n_variants is not None else qcfg.multi_query_count
        self.timeout_s = timeout_s if timeout_s is not None else qcfg.timeout_s
        self.max_query_length = (
            max_query_length if max_query_length is not None else qcfg.max_query_length
        )
        self.max_variant_length = (
            max_variant_length if max_variant_length is not None else qcfg.max_variant_length
        )
        self.language_hint = qcfg.language_hint if language_hint is None else language_hint

        if self.n_variants < 1:
            raise ValueError(f"n_variants must be >= 1, got {self.n_variants}")

        # Lazy LLM instantiation: caller can inject a mock for tests.
        if llm is not None:
            self._llm = llm
        else:
            model = qcfg.expansion_model or settings.llm.model
            self._llm = OllamaClient(model=model)

    def expand(self, query: str) -> ExpansionResult:
        sanitized = sanitize_query(query, max_length=self.max_query_length)
        if not sanitized:
            return ExpansionResult(strategy=self.name, error="empty query")

        prompt = _PROMPT_TEMPLATE.format(n=self.n_variants, query=sanitized)
        # Optional same-language hint. Advisory only — appended to the prompt;
        # detection failure or short query simply omits the hint.
        if self.language_hint:
            from src.language import get_language_detector, language_name

            lang = language_name(get_language_detector().detect(sanitized))
            if lang:
                prompt += f"\n\nWrite every phrasing in {lang}."
        start = time.perf_counter()

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    self._llm.generate,
                    prompt=prompt,
                    context=None,
                    # Each variant is short; cap tokens generously to absorb
                    # the LLM's prefix/suffix noise without bloating latency.
                    max_tokens=max(256, 64 * self.n_variants),
                    temperature=0.7,
                    stream=False,
                )
                output = future.result(timeout=self.timeout_s)
        except FuturesTimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("multi_query timed out after %.0fms", elapsed)
            return ExpansionResult(strategy=self.name, latency_ms=elapsed, error="timeout")
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("multi_query failed: %s", exc)
            return ExpansionResult(strategy=self.name, latency_ms=elapsed, error=f"{type(exc).__name__}: {exc}")

        elapsed = (time.perf_counter() - start) * 1000

        parsed = _parse_variants(output or "", self.max_variant_length)
        # Drop variants equal (case-insensitive, whitespace-normalized) to the
        # original — they add no retrieval signal.
        norm_original = sanitized.lower().strip()
        unique = dedupe_preserve_order([
            v for v in parsed if v.lower().strip() != norm_original
        ])

        # Cap to requested count; LLM may have overproduced.
        variants = unique[: self.n_variants]

        if not variants:
            return ExpansionResult(
                strategy=self.name,
                latency_ms=elapsed,
                error="no usable variants in LLM output",
            )

        return ExpansionResult(
            strategy=self.name,
            queries=tuple(variants),
            latency_ms=elapsed,
        )
