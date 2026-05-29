# src/providers/ollama.py

"""
Ollama (offline/local) provider.

Chat generation delegates to the existing :class:`src.llm_client.OllamaClient`
so answer behavior is byte-identical to the pre-Phase-12 path — that protects
the existing tests and the English-quality gate. Everything else (model
listing, install with streaming progress, delete, connection validation) is
new provider surface that talks to the Ollama HTTP API directly.

Endpoints used (Ollama default ``http://localhost:11434``):

* ``GET  /api/tags``    — installed models
* ``POST /api/pull``    — download a model (streams progress JSON lines)
* ``POST /api/delete``  — remove an installed model
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import requests

from src.logging_setup import get_logger
from src.providers.base import (
    ChatModelConfig,
    ConnectionStatus,
    ModelKind,
    ModelProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderLocation,
    ProviderModel,
)
from src.providers.registry import register_provider

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"

OLLAMA_CAPABILITIES = ProviderCapabilities(
    name="ollama",
    label="Ollama (local)",
    location=ProviderLocation.OFFLINE,
    requires_api_key=False,
    supports_chat=True,
    supports_streaming=True,
    supports_model_listing=True,
    supports_install=True,
    supports_embeddings=True,
    base_url_configurable=True,
    default_base_url=DEFAULT_BASE_URL,
    docs_url="https://ollama.com/library",
    notes="Runs fully offline. No API key. Models are downloaded once and cached locally.",
)


# Curated catalog of popular, freely-pullable models. Ollama has no public
# "list everything downloadable" API, so we ship a sensible, RAM-annotated
# shortlist. ``size_bytes`` is the approximate download size; ``ram_estimate``
# is a rough resident-set guide for non-expert users picking a model.
_GB = 1024 ** 3
_CATALOG: List[ProviderModel] = [
    ProviderModel(
        id="tinyllama", kind=ModelKind.CHAT, label="TinyLlama 1.1B",
        context_window=2048, parameter_size="1.1B", quantization="Q4_0",
        size_bytes=int(0.64 * _GB), ram_estimate_bytes=1 * _GB, multilingual=False,
        description="Smallest practical chat model; great for low-RAM machines and CI.",
    ),
    ProviderModel(
        id="llama3.2:1b", kind=ModelKind.CHAT, label="Llama 3.2 1B",
        context_window=131072, parameter_size="1B", quantization="Q4_K_M",
        size_bytes=int(1.3 * _GB), ram_estimate_bytes=2 * _GB, multilingual=True,
        description="Fast, capable small model with a very large context window.",
    ),
    ProviderModel(
        id="llama3.2:3b", kind=ModelKind.CHAT, label="Llama 3.2 3B",
        context_window=131072, parameter_size="3B", quantization="Q4_K_M",
        size_bytes=int(2.0 * _GB), ram_estimate_bytes=4 * _GB, multilingual=True,
        description="Strong quality-for-size; good default on 8GB+ machines.",
    ),
    ProviderModel(
        id="phi3:mini", kind=ModelKind.CHAT, label="Phi-3 Mini 3.8B",
        context_window=131072, parameter_size="3.8B", quantization="Q4_0",
        size_bytes=int(2.2 * _GB), ram_estimate_bytes=4 * _GB, multilingual=False,
        description="Microsoft Phi-3; reasoning-focused compact model.",
    ),
    ProviderModel(
        id="qwen2.5:3b", kind=ModelKind.CHAT, label="Qwen2.5 3B",
        context_window=32768, parameter_size="3B", quantization="Q4_K_M",
        size_bytes=int(1.9 * _GB), ram_estimate_bytes=4 * _GB, multilingual=True,
        description="Strong multilingual small model (incl. CJK).",
    ),
    ProviderModel(
        id="mistral", kind=ModelKind.CHAT, label="Mistral 7B",
        context_window=32768, parameter_size="7B", quantization="Q4_0",
        size_bytes=int(4.1 * _GB), ram_estimate_bytes=8 * _GB, multilingual=True,
        description="Well-rounded 7B; needs ~8GB RAM.",
    ),
    ProviderModel(
        id="llama3.1:8b", kind=ModelKind.CHAT, label="Llama 3.1 8B",
        context_window=131072, parameter_size="8B", quantization="Q4_K_M",
        size_bytes=int(4.7 * _GB), ram_estimate_bytes=10 * _GB, multilingual=True,
        description="High-quality general model; large context.",
    ),
    ProviderModel(
        id="nomic-embed-text", kind=ModelKind.EMBEDDING, label="Nomic Embed Text",
        context_window=8192, parameter_size="137M", quantization="F16",
        size_bytes=int(0.27 * _GB), ram_estimate_bytes=1 * _GB, multilingual=False,
        description="Local embedding model servable through Ollama.",
    ),
]


class OllamaProvider(ModelProvider):
    """Offline provider backed by a local Ollama server."""

    capabilities = OLLAMA_CAPABILITIES

    def __init__(self, config: ChatModelConfig, *, api_key: Optional[str] = None):
        super().__init__(config, api_key=api_key)
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")
        self.request_timeout_s = config.request_timeout_s
        # Lazy: the delegate client is created on first generation so listing /
        # validation work even if the chat model isn't pulled yet.
        self._client = None

    # -- chat (delegated to OllamaClient for byte-parity) -------------------- #

    def _get_client(self):
        if self._client is None:
            from src.llm_client import OllamaClient

            self._client = OllamaClient(
                model=self.config.model,
                base_url=self.base_url,
                request_timeout_s=self.request_timeout_s,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str:
        return self._get_client().generate(
            prompt=prompt, context=context, max_tokens=max_tokens,
            temperature=temperature, stream=stream,
        )

    def stream_generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        return self._get_client().stream_generate(
            prompt=prompt, context=context, max_tokens=max_tokens, temperature=temperature,
        )

    # -- connection + listing ----------------------------------------------- #

    def validate_connection(self) -> ConnectionStatus:
        import time

        t0 = time.perf_counter()
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return ConnectionStatus(
                ok=True, provider=self.name,
                message=f"Connected to Ollama at {self.base_url}",
                models_available=len(models),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            return ConnectionStatus(
                ok=False, provider=self.name,
                message=f"Ollama not reachable at {self.base_url}: {exc}",
            )

    def list_models(self) -> List[ProviderModel]:
        """Installed models, parsed from ``/api/tags``."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            raise ProviderError(
                f"Could not list Ollama models: {exc}", provider=self.name, retriable=True
            ) from exc

        out: List[ProviderModel] = []
        for entry in resp.json().get("models", []):
            details = entry.get("details", {}) or {}
            name = entry.get("name") or entry.get("model") or ""
            family = (details.get("family") or "").lower()
            kind = ModelKind.EMBEDDING if "embed" in name.lower() else ModelKind.CHAT
            out.append(ProviderModel(
                id=name,
                kind=kind,
                label=name,
                installed=True,
                size_bytes=entry.get("size"),
                parameter_size=details.get("parameter_size"),
                quantization=details.get("quantization_level"),
                # Heuristic: known multilingual families.
                multilingual=any(f in family for f in ("qwen", "gemma", "llama", "mistral")),
            ))
        return out

    def list_catalog(self) -> List[ProviderModel]:
        """Curated downloadable models, annotated with which are installed."""
        import dataclasses

        installed_ids: set = set()
        try:
            installed_ids = {m.id for m in self.list_models()}
        except ProviderError:
            pass  # server down → still show the catalog, just unmarked
        return [
            dataclasses.replace(m, installed=(m.id in installed_ids)) for m in _CATALOG
        ]

    # -- install / delete --------------------------------------------------- #

    def install_model(self, name: str) -> Iterator[Dict[str, Any]]:
        """Pull a model, yielding normalized progress dicts.

        Each yielded dict: ``{"status", "total", "completed", "percent", "digest"}``.
        The final event has ``status="success"`` and ``percent=100``.
        """
        if not name:
            raise ProviderError("model name is required", provider=self.name)
        try:
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": name, "stream": True},
                stream=True,
                timeout=self.request_timeout_s,
            )
            resp.raise_for_status()
        except Exception as exc:
            raise ProviderError(
                f"Failed to start install for {name!r}: {exc}", provider=self.name, retriable=True
            ) from exc

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if "error" in data:
                raise ProviderError(
                    f"Ollama install error: {data['error']}", provider=self.name
                )
            total = data.get("total")
            completed = data.get("completed")
            percent = None
            if total:
                percent = round((completed or 0) / total * 100, 1)
            status = data.get("status", "")
            if status == "success":
                percent = 100.0
            yield {
                "status": status,
                "total": total,
                "completed": completed,
                "percent": percent,
                "digest": data.get("digest"),
            }

    def delete_model(self, name: str) -> None:
        if not name:
            raise ProviderError("model name is required", provider=self.name)
        try:
            resp = requests.delete(
                f"{self.base_url}/api/delete",
                json={"name": name},
                timeout=self.request_timeout_s,
            )
            resp.raise_for_status()
        except Exception as exc:
            raise ProviderError(
                f"Failed to delete {name!r}: {exc}", provider=self.name, retriable=True
            ) from exc
        logger.info("Deleted Ollama model %s", name)


# Register at import so the factory can resolve "ollama".
register_provider(OLLAMA_CAPABILITIES, OllamaProvider)
