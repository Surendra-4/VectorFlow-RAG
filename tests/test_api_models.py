# tests/test_api_models.py

"""
API contract tests for the model-management + runtime-config routes (12d).

No network and no model loads: HTTP to providers is mocked, and the pipeline
is a lightweight stub exposing only what the runtime-mutation methods touch.
The secret store and runtime config are pointed at temp paths.
"""

from __future__ import annotations

import json
import threading
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import src.providers.ollama as ollama_mod
import src.providers.online_base as online_base
import src.providers.secrets as secrets_mod
from src.api.app import create_app
from src.api.dependencies import (
    get_ingest_lock,
    get_pipeline,
    get_runtime_config,
)
from src.config import Settings
from src.providers.secrets import SecretStore
from src.rag_pipeline import RAGPipeline
from src.runtime_config import RuntimeConfigStore


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeResp:
    def __init__(self, status_code=200, json_data=None, lines=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self._lines = lines or []
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json

    def iter_lines(self):
        yield from self._lines


class FakeOllamaRequests:
    def __init__(self):
        self.get_resp = None
        self.post_resp = None
        self.delete_resp = None
        self.raise_on = set()
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(("get", url))
        if "get" in self.raise_on:
            raise ConnectionError("refused")
        return self.get_resp

    def post(self, url, **kw):
        self.calls.append(("post", url))
        if "post" in self.raise_on:
            raise ConnectionError("refused")
        return self.post_resp

    def delete(self, url, **kw):
        self.calls.append(("delete", url))
        if "delete" in self.raise_on:
            raise ConnectionError("refused")
        return self.delete_resp


class FakeOnlineRequests:
    def __init__(self):
        self.next_response = None
        self.next_exc = None
        self.calls = []
        import requests as _r
        self.exceptions = _r.exceptions

    def request(self, method, url, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        if self.next_exc is not None:
            exc, self.next_exc = self.next_exc, None
            raise exc
        return self.next_response


def _stub_pipeline(settings):
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.settings = settings
    pipe._owns_settings = False
    pipe._reranker = None
    pipe._expansion_pipeline = None
    pipe.enable_reranker = settings.reranker.enabled
    pipe.enable_expansion = settings.query_expansion.enabled

    # Minimal llm with a model attr so /models/active works before any switch.
    class _LLM:
        name = "ollama"
        model = settings.llm.model
    pipe.llm = _LLM()
    return pipe


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def temp_secret_store(temp_dir, monkeypatch):
    store = SecretStore(path=temp_dir / "secrets.json", key_path=temp_dir / "secret.key")
    # All routes resolve the singleton via get_secret_store() → reads _STORE.
    monkeypatch.setattr(secrets_mod, "_STORE", store)
    return store


@pytest.fixture
def fake_ollama(monkeypatch):
    fr = FakeOllamaRequests()
    monkeypatch.setattr(ollama_mod, "requests", fr)
    return fr


@pytest.fixture
def fake_online(monkeypatch):
    fr = FakeOnlineRequests()
    monkeypatch.setattr(online_base, "requests", fr)
    return fr


@pytest.fixture
def client(temp_dir, temp_secret_store) -> Iterator[TestClient]:
    settings = Settings()
    pipe = _stub_pipeline(settings)
    runtime = RuntimeConfigStore(settings, path=temp_dir / "runtime_config.json")
    app = create_app(init_pipeline=False)
    shared_lock = threading.Lock()
    app.dependency_overrides[get_pipeline] = lambda: pipe
    app.dependency_overrides[get_ingest_lock] = lambda: shared_lock
    app.dependency_overrides[get_runtime_config] = lambda: runtime
    with TestClient(app) as c:
        c._pipe = pipe          # type: ignore[attr-defined]
        c._runtime = runtime    # type: ignore[attr-defined]
        yield c
    app.dependency_overrides.clear()


_TAGS = {"models": [
    {"name": "tinyllama:latest", "size": 637_000_000,
     "details": {"parameter_size": "1.1B", "quantization_level": "Q4_0", "family": "llama"}},
]}


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


def test_list_providers(client):
    r = client.get("/api/v1/models/providers")
    assert r.status_code == 200
    body = r.json()
    names = {p["name"] for p in body["providers"]}
    assert {"ollama", "openai", "anthropic", "gemini", "groq", "openrouter"} <= names
    # offline-first ordering preserved
    assert body["providers"][0]["name"] == "ollama"
    # no key configured yet
    for p in body["providers"]:
        assert p["key_configured"] is False


def test_list_providers_reflects_configured_key(client, temp_secret_store):
    temp_secret_store.set_secret("openai", "sk-xyz12345")
    r = client.get("/api/v1/models/providers")
    by_name = {p["name"]: p for p in r.json()["providers"]}
    assert by_name["openai"]["key_configured"] is True
    assert by_name["openai"]["key_hint"] == "****2345"
    assert by_name["ollama"]["key_configured"] is False


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #


def test_offline_installed(client, fake_ollama):
    fake_ollama.get_resp = FakeResp(json_data=_TAGS)
    r = client.get("/api/v1/models/offline/installed")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "ollama"
    assert body["models"][0]["id"] == "tinyllama:latest"
    assert body["models"][0]["installed"] is True


def test_offline_installed_unavailable_maps_to_503(client, fake_ollama):
    fake_ollama.raise_on.add("get")
    r = client.get("/api/v1/models/offline/installed")
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "provider_unavailable"
    assert body["details"]["provider"] == "ollama"


def test_offline_catalog(client, fake_ollama):
    fake_ollama.get_resp = FakeResp(json_data={"models": [{"name": "tinyllama", "details": {}}]})
    r = client.get("/api/v1/models/offline/catalog")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["models"]}
    assert "tinyllama" in ids and "mistral" in ids


def test_offline_install_streams_sse(client, fake_ollama):
    fake_ollama.post_resp = FakeResp(lines=[
        b'{"status":"downloading","total":100,"completed":50}',
        b'{"status":"success"}',
    ])
    r = client.post("/api/v1/models/offline/install", json={"name": "tinyllama"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text
    assert "event: progress" in text
    assert "event: done" in text


def test_offline_install_error_emits_error_event(client, fake_ollama):
    fake_ollama.post_resp = FakeResp(lines=[b'{"error":"manifest not found"}'])
    r = client.post("/api/v1/models/offline/install", json={"name": "bogus"})
    assert r.status_code == 200  # SSE opened; failure is in-band
    assert "event: error" in r.text
    assert "manifest not found" in r.text


def test_offline_delete(client, fake_ollama):
    fake_ollama.delete_resp = FakeResp(json_data={})
    r = client.request("DELETE", "/api/v1/models/offline/tinyllama:latest")
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] is True
    assert body["name"] == "tinyllama:latest"


# --------------------------------------------------------------------------- #
# Online providers
# --------------------------------------------------------------------------- #


def test_set_get_delete_api_key(client):
    # set
    r = client.put("/api/v1/models/online/openai/key", json={"api_key": "sk-secret-123"})
    assert r.status_code == 200
    assert r.json()["configured"] is True
    assert r.json()["hint"] == "****-123"
    # The raw key never appears in the response.
    assert "sk-secret-123" not in r.text

    # status
    r = client.get("/api/v1/models/online/openai/key")
    assert r.json()["configured"] is True

    # delete
    r = client.request("DELETE", "/api/v1/models/online/openai/key")
    assert r.json()["configured"] is False

    r = client.get("/api/v1/models/online/openai/key")
    assert r.json()["configured"] is False


def test_set_api_key_rejected_for_offline_provider(client):
    r = client.put("/api/v1/models/online/ollama/key", json={"api_key": "x"})
    # ProviderError (not auth/unavailable/not-found) → 502 provider_error
    assert r.status_code == 502
    assert r.json()["code"] == "provider_error"


def test_online_models_with_key(client, fake_online, temp_secret_store):
    temp_secret_store.set_secret("openai", "sk-live")
    fake_online.next_response = FakeResp(json_data={"data": [{"id": "gpt-4o-mini"}]})
    r = client.get("/api/v1/models/online/openai")
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["models"]}
    assert by_id["gpt-4o-mini"]["context_window"] == 128_000  # merged metadata


def test_online_models_without_key_returns_static(client):
    r = client.get("/api/v1/models/online/openai")
    assert r.status_code == 200
    assert len(r.json()["models"]) > 0  # static catalog


def test_online_unknown_provider_404(client):
    r = client.get("/api/v1/models/online/nonsense")
    assert r.status_code == 404
    assert r.json()["code"] == "provider_not_found"


def test_validate_connection_ok(client, fake_online, temp_secret_store):
    temp_secret_store.set_secret("openai", "sk-live")
    fake_online.next_response = FakeResp(json_data={"data": [{"id": "gpt-4o"}]})
    r = client.post("/api/v1/models/online/openai/validate")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["models_available"] == 1


def test_validate_connection_without_key_returns_ok_false(client):
    r = client.post("/api/v1/models/online/openai/validate")
    assert r.status_code == 200  # validate never raises
    assert r.json()["ok"] is False
    assert "No API key" in r.json()["message"]


# --------------------------------------------------------------------------- #
# Active model + select
# --------------------------------------------------------------------------- #


def test_active_model_default(client):
    r = client.get("/api/v1/models/active")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "ollama"
    assert body["location"] == "offline"


def test_select_model_switches_provider(client):
    r = client.post("/api/v1/models/select", json={"provider": "ollama", "model": "llama3.2:1b"})
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "ollama"
    assert body["model"] == "llama3.2:1b"
    # The stub pipeline's llm was swapped to a real OllamaProvider.
    assert client._pipe.llm.model == "llama3.2:1b"  # type: ignore[attr-defined]
    # Persisted in runtime config.
    assert client._runtime.live.chat.model == "llama3.2:1b"  # type: ignore[attr-defined]


def test_select_online_model_persists(client, temp_secret_store):
    temp_secret_store.set_secret("openai", "sk-live")
    r = client.post("/api/v1/models/select", json={"provider": "openai", "model": "gpt-4o-mini"})
    assert r.status_code == 200
    assert client._runtime.live.chat.provider == "openai"  # type: ignore[attr-defined]
    # /active now reflects the online model.
    r2 = client.get("/api/v1/models/active")
    assert r2.json()["provider"] == "openai"
    assert r2.json()["location"] == "online"


def test_select_unknown_provider_400(client):
    r = client.post("/api/v1/models/select", json={"provider": "ghost", "model": "x"})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_request"
    assert r.json()["details"]["field"] == "chat.provider"


# --------------------------------------------------------------------------- #
# Runtime config routes
# --------------------------------------------------------------------------- #


def test_get_runtime_config(client):
    r = client.get("/api/v1/config/runtime")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    assert "live" in body and "staged_index" in body and "active_index" in body


def test_patch_live_settings_applies_to_pipeline(client):
    r = client.patch("/api/v1/config/runtime/live",
                     json={"reranker_enabled": True, "retrieval_rrf_k": 80})
    assert r.status_code == 200
    live = r.json()["live"]
    assert live["reranker_enabled"] is True
    assert live["retrieval_rrf_k"] == 80
    # Applied to the live pipeline stub.
    assert client._pipe.enable_reranker is True            # type: ignore[attr-defined]
    assert client._pipe.settings.retrieval.rrf_k == 80     # type: ignore[attr-defined]


def test_patch_live_validation_error(client):
    r = client.patch("/api/v1/config/runtime/live", json={"retrieval_rrf_k": 0})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_request"
    assert r.json()["details"]["field"] == "retrieval_rrf_k"


def test_patch_staged_index_does_not_rebuild(client):
    r = client.patch("/api/v1/config/runtime/index",
                     json={"chunk_size": 1000, "chunk_overlap": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["staged_index"]["chunk_size"] == 1000
    assert body["rebuild_required"] is True
    # active index is untouched (no silent rebuild).
    assert client._runtime.active_index.chunk_size == 500  # type: ignore[attr-defined]


def test_patch_staged_index_validation(client):
    r = client.patch("/api/v1/config/runtime/index", json={"vector_backend": "annoy"})
    assert r.status_code == 400
    assert r.json()["details"]["field"] == "vector_backend"


def test_reset_staged_index(client):
    client.patch("/api/v1/config/runtime/index", json={"chunk_size": 999})
    r = client.post("/api/v1/config/runtime/index/reset")
    assert r.status_code == 200
    assert r.json()["rebuild_required"] is False
    assert r.json()["staged_index"]["chunk_size"] == 500


# --------------------------------------------------------------------------- #
# OpenAPI coverage
# --------------------------------------------------------------------------- #


def test_new_routes_in_openapi(client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    assert "/api/v1/models/providers" in paths
    assert "/api/v1/models/select" in paths
    assert "/api/v1/models/offline/installed" in paths
    assert "/api/v1/config/runtime" in paths
    assert "/api/v1/config/runtime/live" in paths
