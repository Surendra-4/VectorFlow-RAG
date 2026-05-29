# tests/test_providers_base.py

"""Unit tests for the provider abstraction core (Phase 12a)."""

from __future__ import annotations

import pytest

from src.providers.base import (
    ChatModelConfig,
    ConnectionStatus,
    EmbeddingModelConfig,
    ModelKind,
    ModelProvider,
    ProviderAuthError,
    ProviderCapabilities,
    ProviderError,
    ProviderLocation,
    ProviderModel,
    ProviderUnavailableError,
    RerankerModelConfig,
)


# --------------------------------------------------------------------------- #
# A minimal concrete provider for exercising the base class.
# --------------------------------------------------------------------------- #


class _DummyCaps:
    pass


class DummyProvider(ModelProvider):
    capabilities = ProviderCapabilities(
        name="dummy", label="Dummy", location=ProviderLocation.ONLINE,
        requires_api_key=True,
    )

    def validate_connection(self) -> ConnectionStatus:
        return ConnectionStatus(ok=True, provider="dummy")

    def list_models(self):
        return [ProviderModel(id="dummy-1")]

    def generate(self, prompt, context=None, max_tokens=512, temperature=0.7, stream=True):
        return self._build_prompt(prompt, context)

    def stream_generate(self, prompt, context=None, max_tokens=512, temperature=0.7):
        yield self._build_prompt(prompt, context)


# --------------------------------------------------------------------------- #
# Metadata value objects
# --------------------------------------------------------------------------- #


def test_provider_model_to_dict_serializes_enum():
    m = ProviderModel(id="x", kind=ModelKind.EMBEDDING, context_window=2048)
    d = m.to_dict()
    assert d["id"] == "x"
    assert d["kind"] == "embedding"  # enum → plain string for JSON
    assert d["context_window"] == 2048
    # optional fields present but None by default
    assert d["pricing"] is None


def test_provider_model_is_frozen():
    m = ProviderModel(id="x")
    with pytest.raises(Exception):
        m.id = "y"  # type: ignore[misc]


def test_capabilities_to_dict_serializes_location():
    caps = ProviderCapabilities(
        name="ollama", label="Ollama", location=ProviderLocation.OFFLINE,
        requires_api_key=False, supports_install=True,
    )
    d = caps.to_dict()
    assert d["location"] == "offline"
    assert d["requires_api_key"] is False
    assert d["supports_install"] is True


def test_connection_status_to_dict():
    s = ConnectionStatus(ok=False, provider="p", message="down")
    assert s.to_dict() == {
        "ok": False, "provider": "p", "message": "down",
        "models_available": None, "latency_ms": None,
    }


# --------------------------------------------------------------------------- #
# Config objects
# --------------------------------------------------------------------------- #


def test_chat_config_defaults_match_legacy():
    c = ChatModelConfig()
    assert c.provider == "ollama"
    assert c.model == "tinyllama"
    assert c.base_url is None
    assert c.max_tokens == 512
    assert c.temperature == 0.7
    assert c.request_timeout_s == 120


def test_chat_config_has_no_api_key_field():
    # Security: keys must never live on the config object.
    assert "api_key" not in ChatModelConfig.__dataclass_fields__


def test_config_from_dict_ignores_unknown_keys():
    c = ChatModelConfig.from_dict({"provider": "openai", "model": "gpt-4o", "junk": 1})
    assert c.provider == "openai"
    assert c.model == "gpt-4o"
    # round-trips
    assert ChatModelConfig.from_dict(c.to_dict()) == c


def test_embedding_and_reranker_defaults_preserve_english_baseline():
    e = EmbeddingModelConfig()
    assert e.provider == "sentence_transformers"
    assert e.model == "all-MiniLM-L6-v2"
    assert e.normalize is True

    r = RerankerModelConfig()
    assert r.enabled is False  # off by default
    assert r.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert r.top_n == 3


# --------------------------------------------------------------------------- #
# Prompt helpers — parity with legacy OllamaClient
# --------------------------------------------------------------------------- #


def test_build_prompt_with_context_matches_legacy_shape():
    out = ModelProvider._build_prompt("Q?", ["a", "b"])
    assert out == "a\n\nb\n\nQuestion: Q?\nAnswer:"


def test_build_prompt_without_context_is_passthrough():
    assert ModelProvider._build_prompt("just this", None) == "just this"
    assert ModelProvider._build_prompt("just this", []) == "just this"


def test_build_messages_folds_context_into_system():
    msgs = ModelProvider._build_messages("Q?", ["ctx"])
    assert msgs[0]["role"] == "system"
    assert "ctx" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "Q?"}


def test_build_messages_without_context_is_user_only():
    msgs = ModelProvider._build_messages("hi", None)
    assert msgs == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# Base behaviors
# --------------------------------------------------------------------------- #


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        ModelProvider(ChatModelConfig())  # type: ignore[abstract]


def test_model_property_back_compat_shim():
    p = DummyProvider(ChatModelConfig(provider="dummy", model="m1"))
    assert p.model == "m1"
    assert p.name == "dummy"


def test_api_key_is_not_a_plain_attribute():
    p = DummyProvider(ChatModelConfig(provider="dummy"), api_key="sk-secret")
    # No attribute literally named api_key / _api_key in instance __dict__.
    assert "api_key" not in p.__dict__
    assert "_api_key" not in p.__dict__
    # Accessor still works internally.
    assert p._api_key == "sk-secret"
    assert p._require_api_key() == "sk-secret"


def test_require_api_key_raises_when_missing():
    p = DummyProvider(ChatModelConfig(provider="dummy"), api_key=None)
    with pytest.raises(ProviderAuthError) as ei:
        p._require_api_key()
    assert ei.value.provider == "dummy"
    assert ei.value.retriable is False


def test_offline_ops_unsupported_by_default():
    p = DummyProvider(ChatModelConfig(provider="dummy"), api_key="k")
    assert p.list_catalog() == []
    with pytest.raises(ProviderError):
        list(p.install_model("x"))
    with pytest.raises(ProviderError):
        p.delete_model("x")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_error_hierarchy_and_flags():
    assert issubclass(ProviderUnavailableError, ProviderError)
    assert issubclass(ProviderAuthError, ProviderError)

    e = ProviderUnavailableError("down", provider="ollama")
    assert e.provider == "ollama"
    assert e.retriable is True

    a = ProviderAuthError("bad key", provider="openai")
    assert a.retriable is False
