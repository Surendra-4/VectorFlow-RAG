# tests/test_providers_online.py

"""Tests for the four online providers (Phase 12b).

All HTTP is mocked through ``src.providers.online_base.requests`` — no
network, no real keys. The fake responses are shaped exactly like the real
APIs documented in each provider module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
import requests as _real_requests

import src.providers.online_base as online_base
from src.providers import (
    ChatModelConfig,
    ModelNotFoundError,
    ProviderAuthError,
    ProviderError,
    ProviderUnavailableError,
    SecretStore,
    list_provider_capabilities,
    make_chat_provider,
)
from src.providers.anthropic import AnthropicProvider
from src.providers.gemini import GeminiProvider
from src.providers.openai_compat import (
    GroqProvider,
    OpenAIProvider,
    OpenRouterProvider,
)


# --------------------------------------------------------------------------- #
# Fake HTTP harness
# --------------------------------------------------------------------------- #


class FakeResp:
    def __init__(self, status_code: int = 200, json_data: Optional[Dict] = None,
                 lines: Optional[List[bytes]] = None, text: str = ""):
        self.status_code = status_code
        self._json = json_data
        self._lines = lines or []
        self.text = text

    def json(self):
        return self._json

    def iter_lines(self):
        for line in self._lines:
            yield line


class FakeReq:
    """A drop-in for the ``requests`` module that online_base uses."""

    def __init__(self):
        self.next_response: Optional[FakeResp] = None
        self.next_exc: Optional[Exception] = None
        # Keyed by (method, idx) when the test wants distinct responses per call.
        self.scripted: List[FakeResp] = []
        self.calls: List[Dict[str, Any]] = []
        # Real requests.exceptions so the base class catches our raises.
        self.exceptions = _real_requests.exceptions

    def request(self, method, url, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        if self.next_exc is not None:
            exc, self.next_exc = self.next_exc, None
            raise exc
        if self.scripted:
            return self.scripted.pop(0)
        return self.next_response


@pytest.fixture
def http(monkeypatch):
    fr = FakeReq()
    monkeypatch.setattr(online_base, "requests", fr)
    return fr


# --------------------------------------------------------------------------- #
# Helpers to instantiate each provider with a key
# --------------------------------------------------------------------------- #


def _openai(key="sk-test"):
    return OpenAIProvider(ChatModelConfig(provider="openai", model="gpt-4o-mini"), api_key=key)


def _groq(key="gsk-test"):
    return GroqProvider(ChatModelConfig(provider="groq", model="llama-3.1-8b-instant"), api_key=key)


def _openrouter(key="or-test"):
    return OpenRouterProvider(
        ChatModelConfig(provider="openrouter", model="openai/gpt-4o-mini"), api_key=key
    )


def _anthropic(key="sk-ant-test"):
    return AnthropicProvider(
        ChatModelConfig(provider="anthropic", model="claude-3-5-haiku-latest"), api_key=key
    )


def _gemini(key="goog-test"):
    return GeminiProvider(ChatModelConfig(provider="gemini", model="gemini-1.5-flash"), api_key=key)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_all_four_providers_registered():
    names = {c.name for c in list_provider_capabilities()}
    for required in {"ollama", "openai", "groq", "openrouter", "anthropic", "gemini"}:
        assert required in names


def test_make_chat_provider_injects_key_for_online_only(temp_dir):
    store = SecretStore(path=temp_dir / "s.json", key_path=temp_dir / "s.key")
    store.set_secret("openai", "sk-from-store")
    p = make_chat_provider(ChatModelConfig(provider="openai"), secret_store=store)
    assert isinstance(p, OpenAIProvider)
    assert p._api_key == "sk-from-store"

    # Ollama doesn't require a key → store is not consulted.
    store.set_secret("ollama", "should-not-be-used")
    p2 = make_chat_provider(ChatModelConfig(provider="ollama"), secret_store=store)
    assert p2._api_key is None


# --------------------------------------------------------------------------- #
# OpenAI-compatible: chat (non-stream + stream)
# --------------------------------------------------------------------------- #


def _openai_nonstream_resp(text="Hello, world."):
    return FakeResp(json_data={
        "choices": [{"message": {"role": "assistant", "content": text}}],
    })


def _openai_stream_lines(tokens):
    out = []
    for t in tokens:
        out.append(f'data: {{"choices":[{{"delta":{{"content":"{t}"}}}}]}}'.encode())
    out.append(b"data: [DONE]")
    return out


def test_openai_generate_non_stream(http):
    http.next_response = _openai_nonstream_resp("Hi")
    out = _openai().generate("Q?", context=["c"])
    assert out == "Hi"
    call = http.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["Authorization"].startswith("Bearer sk-")
    payload = call["json"]
    # System message folded in for RAG context (OpenAI-compat path).
    assert payload["messages"][0]["role"] == "system"
    assert "c" in payload["messages"][0]["content"]
    assert payload["messages"][-1] == {"role": "user", "content": "Q?"}
    assert payload["stream"] is False


def test_openai_stream_generate_yields_deltas(http):
    http.next_response = FakeResp(lines=_openai_stream_lines(["Hel", "lo"]))
    toks = list(_openai().stream_generate("Q?"))
    assert toks == ["Hel", "lo"]
    assert http.calls[0]["json"]["stream"] is True


def test_groq_uses_groq_base(http):
    http.next_response = _openai_nonstream_resp("ok")
    _groq().generate("Q?")
    assert "groq.com" in http.calls[0]["url"]


def test_openrouter_sends_attribution_header(http):
    http.next_response = _openai_nonstream_resp("ok")
    _openrouter().generate("Q?")
    assert http.calls[0]["headers"]["X-Title"] == "VectorFlow-RAG"


# --------------------------------------------------------------------------- #
# OpenAI-compatible: list_models with/without key, validate_connection
# --------------------------------------------------------------------------- #


def test_list_models_without_key_returns_static_catalog(http):
    p = OpenAIProvider(ChatModelConfig(provider="openai"))  # no key
    models = p.list_models()
    assert {m.id for m in models} == {m.id for m in OpenAIProvider.STATIC_MODELS}
    # No HTTP call was made.
    assert http.calls == []


def test_list_models_with_key_hits_api_and_merges_metadata(http):
    http.next_response = FakeResp(json_data={
        "data": [{"id": "gpt-4o-mini"}, {"id": "some-unknown-id"}],
    })
    models = _openai().list_models()
    by_id = {m.id: m for m in models}
    assert by_id["gpt-4o-mini"].context_window == 128_000  # merged from static
    assert by_id["some-unknown-id"].context_window is None


def test_list_models_falls_back_to_static_when_api_errors(http):
    http.next_response = FakeResp(status_code=500, text="boom")
    models = _openai().list_models()
    assert len(models) == len(OpenAIProvider.STATIC_MODELS)


def test_validate_connection_without_key():
    p = OpenAIProvider(ChatModelConfig(provider="openai"))
    s = p.validate_connection()
    assert s.ok is False
    assert "No API key" in s.message


def test_validate_connection_ok(http):
    http.next_response = FakeResp(json_data={"data": [{"id": "gpt-4o"}]})
    s = _openai().validate_connection()
    assert s.ok is True
    assert s.models_available == 1


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #


def test_401_maps_to_auth_error_stream(http):
    http.next_response = FakeResp(status_code=401, text="bad key")
    with pytest.raises(ProviderAuthError):
        list(_openai().stream_generate("Q?"))


def test_404_maps_to_model_not_found_stream(http):
    http.next_response = FakeResp(status_code=404, text="nope")
    with pytest.raises(ModelNotFoundError):
        list(_openai().stream_generate("Q?"))


def test_5xx_maps_to_unavailable_stream(http):
    http.next_response = FakeResp(status_code=503, text="down")
    with pytest.raises(ProviderUnavailableError):
        list(_openai().stream_generate("Q?"))


def test_connection_refused_maps_to_unavailable(http):
    http.next_exc = _real_requests.exceptions.ConnectionError("refused")
    with pytest.raises(ProviderUnavailableError):
        list(_openai().stream_generate("Q?"))


def test_nonstream_generate_returns_graceful_error_on_auth(http):
    # Parity with OllamaClient.generate: non-streaming path NEVER raises;
    # it returns an error string the API can pass through.
    http.next_response = FakeResp(status_code=401, text="bad key")
    out = _openai().generate("Q?")
    assert out.startswith("[Error communicating with OpenAI")


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


def test_anthropic_payload_uses_top_level_system(http):
    http.next_response = FakeResp(json_data={
        "content": [{"type": "text", "text": "ok"}],
    })
    _anthropic().generate("Q?", context=["alpha", "beta"])
    payload = http.calls[0]["json"]
    assert payload["model"] == "claude-3-5-haiku-latest"
    assert payload["messages"] == [{"role": "user", "content": "Q?"}]
    assert "system" in payload
    assert "alpha" in payload["system"] and "beta" in payload["system"]


def test_anthropic_headers(http):
    http.next_response = FakeResp(json_data={"content": []})
    _anthropic().generate("Q?")
    h = http.calls[0]["headers"]
    assert h["x-api-key"] == "sk-ant-test"
    assert h["anthropic-version"] == "2023-06-01"


def test_anthropic_stream_extracts_text_deltas(http):
    lines = [
        b'data: {"type":"message_start"}',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}',
        b'data: {"type":"message_stop"}',
    ]
    http.next_response = FakeResp(lines=lines)
    toks = list(_anthropic().stream_generate("Q?"))
    assert toks == ["Hel", "lo"]


def test_anthropic_nonstream_concatenates_text_blocks(http):
    http.next_response = FakeResp(json_data={
        "content": [
            {"type": "text", "text": "Hel"},
            {"type": "tool_use", "name": "x"},  # ignored
            {"type": "text", "text": "lo"},
        ],
    })
    assert _anthropic().generate("Q?") == "Hello"


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


def test_gemini_url_contains_key_and_picks_stream_path(http):
    http.next_response = FakeResp(json_data={"candidates": []})
    p = _gemini()

    p.generate("Q?")
    url1 = http.calls[-1]["url"]
    assert ":generateContent" in url1
    assert "key=goog-test" in url1
    assert "alt=sse" not in url1

    # Streaming uses a different path.
    http.next_response = FakeResp(lines=[b'data: {"candidates":[]}'])
    list(p.stream_generate("Q?"))
    url2 = http.calls[-1]["url"]
    assert ":streamGenerateContent" in url2
    assert "alt=sse" in url2
    assert "key=goog-test" in url2


def test_gemini_payload_shape(http):
    http.next_response = FakeResp(json_data={"candidates": []})
    _gemini().generate("Q?", context=["ctx"], max_tokens=64, temperature=0.1)
    payload = http.calls[-1]["json"]
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0]["text"] == "Q?"
    assert payload["generationConfig"]["maxOutputTokens"] == 64
    assert payload["generationConfig"]["temperature"] == 0.1
    assert "ctx" in payload["systemInstruction"]["parts"][0]["text"]


def test_gemini_nonstream_extracts_concatenated_text(http):
    http.next_response = FakeResp(json_data={
        "candidates": [{"content": {"parts": [{"text": "Hel"}, {"text": "lo"}]}}],
    })
    assert _gemini().generate("Q?") == "Hello"


def test_gemini_stream_yields_part_text(http):
    http.next_response = FakeResp(lines=[
        b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}',
        b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}',
    ])
    toks = list(_gemini().stream_generate("Q?"))
    assert toks == ["Hel", "lo"]


def test_gemini_parse_models_strips_prefix(http):
    http.next_response = FakeResp(json_data={
        "models": [
            {"name": "models/gemini-1.5-flash", "displayName": "Gemini 1.5 Flash",
             "inputTokenLimit": 1_000_000},
            {"name": "models/gemini-1.5-pro", "displayName": "Gemini 1.5 Pro"},
        ],
    })
    models = _gemini().list_models()
    by_id = {m.id: m for m in models}
    assert "gemini-1.5-flash" in by_id
    assert by_id["gemini-1.5-flash"].context_window == 1_000_000
