# tests/test_encoding_safety.py

"""
End-to-end encoding-safety audit (Phase 11).

Multilingual systems often fail in the *observability* and *serialization*
layers before they fail in retrieval. This suite asserts that non-ASCII
content (CJK, Arabic RTL, Cyrillic, emoji, combining marks) survives every
serialization boundary intact:

  * JSON serialization (logs, API responses)
  * Prometheus text export
  * cache codec (pickle) round-trip
  * RetrievalTrace.to_dict() round-trip
  * SSE event encoding (UTF-8 bytes)
  * structured-log JSON formatter
"""

from __future__ import annotations

import json

import pytest

# A representative gauntlet of non-ASCII content.
SAMPLES = [
    "café société",                     # Latin + accents
    "光合作用将光能转化为化学能",        # Chinese
    "باريس هي عاصمة فرنسا",             # Arabic RTL
    "Париж — столица Франции",          # Cyrillic + em dash
    "光 + café + سين 🌍",               # mixed scripts + emoji
    "café",                             # decomposed (NFD) accent
]


class TestJsonSerialization:
    @pytest.mark.parametrize("text", SAMPLES)
    def test_json_round_trip_preserves_text(self, text):
        # ensure_ascii=False is the project convention — verify it round-trips.
        encoded = json.dumps({"text": text}, ensure_ascii=False)
        decoded = json.loads(encoded)
        assert decoded["text"] == text

    def test_json_default_ascii_also_round_trips(self):
        # Even ensure_ascii=True (escaped) must decode back to the original.
        for text in SAMPLES:
            assert json.loads(json.dumps({"t": text}))["t"] == text


class TestCacheCodecRoundTrip:
    @pytest.mark.parametrize("text", SAMPLES)
    def test_pickle_codec_preserves_unicode(self, text):
        from src.cache.codec import PickleCodec

        codec = PickleCodec()
        payload = {"chunk_id": "doc:0:é", "text": text, "metadata": {"document_name": text}}
        assert codec.decode(codec.encode(payload)) == payload


class TestRetrievalTraceEncoding:
    def test_trace_to_dict_json_serializable_with_unicode(self):
        from src.retrieval_trace import RetrievalTrace

        trace = RetrievalTrace(
            original_query="法国的首都是什么",
            sanitized_query="法国的首都是什么",
        )
        trace.expanded_queries = ["法国首都", "capital of France", "باريس"]
        trace.hyde_documents = ["巴黎是法国的首都。"]
        trace.final_results = [
            {"text": "Париж — столица", "document_name": "вики.txt", "chunk_id": "ru:0:x"}
        ]
        payload = trace.to_dict()
        # Must serialize with non-ASCII preserved.
        encoded = json.dumps(payload, ensure_ascii=False)
        assert "法国的首都是什么" in encoded
        assert json.loads(encoded)["original_query"] == "法国的首都是什么"


class TestPrometheusExportEncoding:
    def test_prometheus_export_is_valid_utf8(self):
        from src.observability.registry import MetricsRegistry
        from src.observability.prometheus_export import to_prometheus_text

        reg = MetricsRegistry()
        # Label values are normally bounded (endpoints/status), but verify the
        # exporter emits UTF-8-encodable text even if a label carries unicode.
        reg.requests_total.inc("/search/café", "200")
        text = to_prometheus_text(reg)
        # Must encode to UTF-8 without error.
        encoded = text.encode("utf-8")
        assert isinstance(encoded, bytes)
        assert "requests_total" in text


class TestStructuredLogEncoding:
    @pytest.mark.parametrize("text", SAMPLES)
    def test_json_log_formatter_preserves_unicode(self, text):
        import logging

        from src.logging_setup import JsonFormatter

        rec = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="query=%s", args=(text,), exc_info=None,
        )
        out = JsonFormatter().format(rec)
        # JsonFormatter uses ensure_ascii=False → text appears literally and
        # decodes back intact.
        decoded = json.loads(out)
        assert text in decoded["message"]


class TestSseEventEncoding:
    @pytest.mark.parametrize("text", SAMPLES)
    def test_sse_token_payload_round_trips(self, text):
        # The /ask SSE route emits `data: {json}` lines. Verify a token payload
        # with unicode encodes + decodes through the same path the client uses.
        line = f"data: {json.dumps({'token': text}, ensure_ascii=False)}"
        # Simulate the client-side parse (strip "data: ", json.loads).
        assert line.startswith("data: ")
        payload = json.loads(line[len("data: "):])
        assert payload["token"] == text
        # And the whole line must be UTF-8 transport-safe.
        assert line.encode("utf-8").decode("utf-8") == line
