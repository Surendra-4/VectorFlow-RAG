# tests/test_providers_ollama.py

"""Tests for OllamaProvider + the provider registry/factory (Phase 12a).

All HTTP is mocked — no network, no running Ollama required.
"""

from __future__ import annotations

import pytest

import src.providers.ollama as ollama_mod
from src.providers import (
    ChatModelConfig,
    ProviderError,
    list_provider_capabilities,
    make_chat_provider,
)
from src.providers.base import ModelKind
from src.providers.ollama import OllamaProvider
from src.providers.registry import get_provider_capabilities


# --------------------------------------------------------------------------- #
# Fake HTTP plumbing
# --------------------------------------------------------------------------- #


class FakeResp:
    def __init__(self, json_data=None, lines=None, status_ok=True):
        self._json = json_data
        self._lines = lines or []
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._json

    def iter_lines(self):
        for line in self._lines:
            yield line


class FakeRequests:
    """Routes get/post/delete to preconfigured responses or raises."""

    def __init__(self):
        self.get_resp = None
        self.post_resp = None
        self.delete_resp = None
        self.raise_on = set()  # {"get", "post", "delete"}
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(("get", url, kw))
        if "get" in self.raise_on:
            raise ConnectionError("refused")
        return self.get_resp

    def post(self, url, **kw):
        self.calls.append(("post", url, kw))
        if "post" in self.raise_on:
            raise ConnectionError("refused")
        return self.post_resp

    def delete(self, url, **kw):
        self.calls.append(("delete", url, kw))
        if "delete" in self.raise_on:
            raise ConnectionError("refused")
        return self.delete_resp


@pytest.fixture
def fake_requests(monkeypatch):
    fr = FakeRequests()
    monkeypatch.setattr(ollama_mod, "requests", fr)
    return fr


@pytest.fixture
def provider():
    return OllamaProvider(ChatModelConfig(provider="ollama", model="tinyllama",
                                          base_url="http://localhost:11434"))


_TAGS = {
    "models": [
        {"name": "tinyllama:latest", "size": 637_000_000,
         "details": {"parameter_size": "1.1B", "quantization_level": "Q4_0", "family": "llama"}},
        {"name": "nomic-embed-text:latest", "size": 270_000_000,
         "details": {"parameter_size": "137M", "quantization_level": "F16", "family": "nomic-bert"}},
    ]
}


# --------------------------------------------------------------------------- #
# Connection validation
# --------------------------------------------------------------------------- #


def test_validate_connection_ok(provider, fake_requests):
    fake_requests.get_resp = FakeResp(json_data=_TAGS)
    status = provider.validate_connection()
    assert status.ok is True
    assert status.provider == "ollama"
    assert status.models_available == 2
    assert status.latency_ms is not None


def test_validate_connection_failure_does_not_raise(provider, fake_requests):
    fake_requests.raise_on.add("get")
    status = provider.validate_connection()
    assert status.ok is False
    assert "not reachable" in status.message


# --------------------------------------------------------------------------- #
# Model listing
# --------------------------------------------------------------------------- #


def test_list_models_parses_tags(provider, fake_requests):
    fake_requests.get_resp = FakeResp(json_data=_TAGS)
    models = provider.list_models()
    by_id = {m.id: m for m in models}

    chat = by_id["tinyllama:latest"]
    assert chat.kind == ModelKind.CHAT
    assert chat.installed is True
    assert chat.size_bytes == 637_000_000
    assert chat.parameter_size == "1.1B"
    assert chat.quantization == "Q4_0"
    assert chat.multilingual is True  # llama family heuristic

    emb = by_id["nomic-embed-text:latest"]
    assert emb.kind == ModelKind.EMBEDDING  # name contains "embed"


def test_list_models_failure_raises_provider_error(provider, fake_requests):
    fake_requests.raise_on.add("get")
    with pytest.raises(ProviderError) as ei:
        provider.list_models()
    assert ei.value.provider == "ollama"
    assert ei.value.retriable is True


def test_list_catalog_marks_installed(provider, fake_requests):
    fake_requests.get_resp = FakeResp(json_data={
        "models": [{"name": "tinyllama", "details": {}}]
    })
    catalog = provider.list_catalog()
    ids = {m.id for m in catalog}
    assert "tinyllama" in ids
    assert "mistral" in ids
    installed = {m.id: m.installed for m in catalog}
    assert installed["tinyllama"] is True
    assert installed["mistral"] is False


def test_list_catalog_survives_server_down(provider, fake_requests):
    fake_requests.raise_on.add("get")
    catalog = provider.list_catalog()
    assert len(catalog) > 0  # still shows the catalog
    assert all(m.installed is False for m in catalog)


# --------------------------------------------------------------------------- #
# Install (streaming progress) + delete
# --------------------------------------------------------------------------- #


def test_install_model_yields_normalized_progress(provider, fake_requests):
    fake_requests.post_resp = FakeResp(lines=[
        b'{"status":"pulling manifest"}',
        b'{"status":"downloading","total":100,"completed":50,"digest":"sha256:abc"}',
        b'not-json-ignored',
        b'{"status":"success"}',
    ])
    events = list(provider.install_model("tinyllama"))
    assert events[0]["status"] == "pulling manifest"
    mid = events[1]
    assert mid["percent"] == 50.0
    assert mid["digest"] == "sha256:abc"
    assert events[-1]["status"] == "success"
    assert events[-1]["percent"] == 100.0


def test_install_model_error_event_raises(provider, fake_requests):
    fake_requests.post_resp = FakeResp(lines=[b'{"error":"model not found"}'])
    with pytest.raises(ProviderError) as ei:
        list(provider.install_model("bogus"))
    assert "model not found" in str(ei.value)


def test_install_requires_name(provider):
    with pytest.raises(ProviderError):
        list(provider.install_model(""))


def test_delete_model_ok(provider, fake_requests):
    fake_requests.delete_resp = FakeResp(json_data={})
    provider.delete_model("tinyllama")
    assert any(c[0] == "delete" for c in fake_requests.calls)


def test_delete_model_failure_raises(provider, fake_requests):
    fake_requests.raise_on.add("delete")
    with pytest.raises(ProviderError):
        provider.delete_model("tinyllama")


# --------------------------------------------------------------------------- #
# Chat delegation (no network — inject a fake client)
# --------------------------------------------------------------------------- #


class FakeClient:
    def __init__(self):
        self.gen_calls = []
        self.stream_calls = []

    def generate(self, prompt, context=None, max_tokens=512, temperature=0.7, stream=True):
        self.gen_calls.append((prompt, context, max_tokens, temperature, stream))
        return "ANSWER"

    def stream_generate(self, prompt, context=None, max_tokens=512, temperature=0.7):
        self.stream_calls.append((prompt, context, max_tokens, temperature))
        yield "tok1"
        yield "tok2"


def test_generate_delegates_to_client(provider):
    fc = FakeClient()
    provider._client = fc
    out = provider.generate("Q", context=["c"], max_tokens=64, temperature=0.1)
    assert out == "ANSWER"
    assert fc.gen_calls == [("Q", ["c"], 64, 0.1, True)]


def test_stream_generate_delegates_to_client(provider):
    fc = FakeClient()
    provider._client = fc
    toks = list(provider.stream_generate("Q", context=["c"]))
    assert toks == ["tok1", "tok2"]
    assert fc.stream_calls[0][0] == "Q"


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #


def test_factory_builds_ollama_without_key():
    p = make_chat_provider(ChatModelConfig(provider="ollama", model="tinyllama"))
    assert isinstance(p, OllamaProvider)
    assert p._api_key is None  # offline → no key injected


def test_unknown_provider_raises():
    with pytest.raises(ProviderError):
        make_chat_provider(ChatModelConfig(provider="does-not-exist"))


def test_capabilities_listed_and_offline_first():
    caps = list_provider_capabilities()
    names = [c.name for c in caps]
    assert "ollama" in names
    # ollama is offline → must sort before any online provider
    assert names[0] == "ollama"


def test_ollama_capabilities_flags():
    caps = get_provider_capabilities("ollama")
    assert caps.requires_api_key is False
    assert caps.supports_install is True
    assert caps.location.value == "offline"
