# src/providers/online_base.py

"""
Shared base for online (remote API) chat providers.

All online providers — OpenAI/Groq/OpenRouter (OpenAI-compatible), Anthropic,
Gemini — share the same shape: authenticated HTTP, a JSON chat call, an SSE
streaming variant, and a model-listing call. This base captures the common
plumbing (requests, error mapping, SSE line parsing, graceful generate) so the
concrete providers only declare the parts that actually differ.

No third-party SDKs: everything goes through ``requests`` (already a dep).
This keeps the dependency surface flat and the failure modes uniform.

Error mapping (consistent across providers):

* connection refused / timeout      → ProviderUnavailableError (retriable)
* 401 / 403                          → ProviderAuthError (not retriable)
* 404 (unknown model)               → ModelNotFoundError
* 429 / 5xx                          → ProviderUnavailableError (retriable)
* other 4xx                          → ProviderError
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, Iterator, List, Optional

import requests

from src.logging_setup import get_logger
from src.providers.base import (
    ChatModelConfig,
    ConnectionStatus,
    ModelNotFoundError,
    ModelProvider,
    ProviderAuthError,
    ProviderError,
    ProviderModel,
    ProviderUnavailableError,
)

logger = get_logger(__name__)


class HTTPChatProvider(ModelProvider):
    """Base class for remote chat providers driven over HTTP.

    Subclasses set ``capabilities``, ``DEFAULT_BASE_URL``, ``STATIC_MODELS``
    and implement the small set of provider-specific hooks below.
    """

    DEFAULT_BASE_URL: str = ""
    STATIC_MODELS: List[ProviderModel] = []

    def __init__(self, config: ChatModelConfig, *, api_key: Optional[str] = None):
        super().__init__(config, api_key=api_key)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.request_timeout_s = config.request_timeout_s

    # -- provider-specific hooks (override) ---------------------------------- #

    def _auth_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    def _chat_url(self, stream: bool = False) -> str:
        """The chat endpoint URL. ``stream`` is honored by providers (e.g.
        Gemini) that use a different path for streaming."""
        raise NotImplementedError

    def _models_url(self) -> str:
        raise NotImplementedError

    def _build_chat_payload(
        self, prompt: str, context: Optional[List[str]],
        max_tokens: int, temperature: float, stream: bool,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _extract_nonstream_text(self, data: Dict[str, Any]) -> str:
        raise NotImplementedError

    def _extract_stream_delta(self, event: Dict[str, Any]) -> str:
        """Return the text delta from one parsed SSE JSON event ("" if none)."""
        raise NotImplementedError

    def _parse_models(self, data: Dict[str, Any]) -> List[ProviderModel]:
        raise NotImplementedError

    # -- shared HTTP plumbing ------------------------------------------------ #

    def _map_status_error(self, status_code: int, body: str) -> ProviderError:
        if status_code in (401, 403):
            return ProviderAuthError(
                f"{self.capabilities.label} rejected the API key (HTTP {status_code}).",
                provider=self.name,
            )
        if status_code == 404:
            return ModelNotFoundError(
                f"{self.capabilities.label}: model {self.config.model!r} not found (HTTP 404).",
                provider=self.name,
            )
        if status_code == 429 or status_code >= 500:
            return ProviderUnavailableError(
                f"{self.capabilities.label} temporarily unavailable (HTTP {status_code}).",
                provider=self.name,
            )
        return ProviderError(
            f"{self.capabilities.label} request failed (HTTP {status_code}): {body[:200]}",
            provider=self.name,
        )

    def _request(self, method: str, url: str, *, json_body=None, stream=False, timeout=None):
        headers = self._auth_headers()
        try:
            resp = requests.request(
                method, url, headers=headers, json=json_body, stream=stream,
                timeout=timeout or self.request_timeout_s,
            )
        except requests.exceptions.RequestException as exc:
            raise ProviderUnavailableError(
                f"{self.capabilities.label} unreachable: {exc}", provider=self.name
            ) from exc
        if resp.status_code >= 400:
            body = ""
            try:
                body = resp.text
            except Exception:  # pragma: no cover - defensive
                pass
            raise self._map_status_error(resp.status_code, body)
        return resp

    def _iter_sse_json(self, response) -> Iterator[Dict[str, Any]]:
        """Yield parsed JSON objects from ``data:`` SSE lines (skips [DONE])."""
        for raw in response.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue

    # -- public interface ---------------------------------------------------- #

    def validate_connection(self) -> ConnectionStatus:
        import time

        if not self._api_key:
            return ConnectionStatus(
                ok=False, provider=self.name,
                message=f"No API key configured for {self.capabilities.label}.",
            )
        t0 = time.perf_counter()
        try:
            resp = self._request("GET", self._models_url(), timeout=10)
            models = self._parse_models(resp.json())
            return ConnectionStatus(
                ok=True, provider=self.name,
                message=f"Connected to {self.capabilities.label}",
                models_available=len(models),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except ProviderError as exc:
            return ConnectionStatus(ok=False, provider=self.name, message=exc.message)

    def list_models(self) -> List[ProviderModel]:
        """Live models when a key is set; otherwise the static curated catalog.

        When a key is present we hit the API and enrich each live id with any
        static metadata (context window, multilingual flag) we know about.
        """
        if not self._api_key:
            return list(self.STATIC_MODELS)
        try:
            resp = self._request("GET", self._models_url(), timeout=15)
            live = self._parse_models(resp.json())
        except ProviderError as exc:
            logger.warning("list_models for %s failed, using static catalog: %s", self.name, exc)
            return list(self.STATIC_MODELS)
        return self._merge_static_metadata(live)

    def _merge_static_metadata(self, live: List[ProviderModel]) -> List[ProviderModel]:
        static_by_id = {m.id: m for m in self.STATIC_MODELS}
        merged: List[ProviderModel] = []
        for m in live:
            ref = static_by_id.get(m.id)
            if ref is None:
                merged.append(m)
                continue
            merged.append(dataclasses.replace(
                m,
                label=m.label or ref.label,
                context_window=m.context_window or ref.context_window,
                supports_tools=ref.supports_tools,
                multilingual=ref.multilingual,
                description=ref.description,
            ))
        return merged

    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str:
        """Non-streaming generation. Degrades to an error string (parity with
        OllamaClient.generate) so the non-streaming /ask path never 500s."""
        try:
            self._require_api_key()
            payload = self._build_chat_payload(prompt, context, max_tokens, temperature, stream=False)
            resp = self._request("POST", self._chat_url(stream=False), json_body=payload)
            text = self._extract_nonstream_text(resp.json())
            return text.strip() or "No response generated."
        except ProviderError as exc:
            logger.error("%s generate failed: %s", self.name, exc)
            return f"[Error communicating with {self.capabilities.label}: {exc.message}]"

    def stream_generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Streaming generation. Raises ProviderError on failure (the SSE /ask
        route catches and emits an ``error`` event) — parity with OllamaClient."""
        self._require_api_key()
        payload = self._build_chat_payload(prompt, context, max_tokens, temperature, stream=True)
        resp = self._request("POST", self._chat_url(stream=True), json_body=payload, stream=True)
        for event in self._iter_sse_json(resp):
            delta = self._extract_stream_delta(event)
            if delta:
                yield delta
