# src/providers/gemini.py

"""
Google Gemini (Generative Language API) provider.

Three quirks the base needs to know about:

* Auth is a ``?key=`` **query parameter**, not a header — ``_auth_headers``
  returns only ``Content-Type`` and the key is folded into each URL.
* The model id is part of the **path**: ``/models/<model>:generateContent``.
* Streaming uses a **different path** (``:streamGenerateContent?alt=sse``)
  from the non-streaming one — that's exactly what the ``stream`` flag on
  :meth:`_chat_url` is for.

Everything else (SSE ``data:`` parsing, error mapping, generate parity) reuses
the :class:`HTTPChatProvider` plumbing.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.providers.base import (
    ModelKind,
    ProviderCapabilities,
    ProviderLocation,
    ProviderModel,
)
from src.providers.online_base import HTTPChatProvider
from src.providers.registry import register_provider

GEMINI_CAPABILITIES = ProviderCapabilities(
    name="gemini", label="Google Gemini", location=ProviderLocation.ONLINE,
    requires_api_key=True, supports_install=False, base_url_configurable=True,
    default_base_url="https://generativelanguage.googleapis.com/v1beta",
    docs_url="https://ai.google.dev/gemini-api/docs",
    notes="Generative Language API. The key is sent as a URL parameter.",
)

_STATIC: List[ProviderModel] = [
    ProviderModel(id="gemini-1.5-flash", label="Gemini 1.5 Flash",
                  context_window=1_000_000, supports_tools=True, multilingual=True,
                  description="Fast, long-context model."),
    ProviderModel(id="gemini-1.5-pro", label="Gemini 1.5 Pro",
                  context_window=2_000_000, supports_tools=True, multilingual=True,
                  description="Highest-quality long-context model."),
    ProviderModel(id="gemini-2.0-flash", label="Gemini 2.0 Flash",
                  context_window=1_000_000, supports_tools=True, multilingual=True,
                  description="Next-generation Flash model."),
    ProviderModel(id="gemini-2.0-flash-lite", label="Gemini 2.0 Flash-Lite",
                  context_window=1_000_000, multilingual=True),
]


def _strip_models_prefix(name: str) -> str:
    return name[len("models/"):] if name.startswith("models/") else name


class GeminiProvider(HTTPChatProvider):
    capabilities = GEMINI_CAPABILITIES
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    STATIC_MODELS = _STATIC

    def _auth_headers(self) -> Dict[str, str]:
        # Key travels in the URL — the only fixed header is content type. We
        # still call ``_require_api_key`` here so missing-key requests fail
        # consistently with the rest of the providers.
        self._require_api_key()
        return {"Content-Type": "application/json"}

    def _chat_url(self, stream: bool = False) -> str:
        model = self.config.model
        path = "streamGenerateContent" if stream else "generateContent"
        url = f"{self.base_url}/models/{model}:{path}?key={self._require_api_key()}"
        if stream:
            url += "&alt=sse"
        return url

    def _models_url(self) -> str:
        return f"{self.base_url}/models?key={self._require_api_key()}"

    def _build_chat_payload(self, prompt, context, max_tokens, temperature, stream):
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if context:
            payload["systemInstruction"] = {
                "parts": [{"text": (
                    "Answer the question using only the context below. "
                    "If the answer is not in the context, say so.\n\n"
                    "Context:\n" + "\n\n".join(context)
                )}],
            }
        return payload

    def _extract_nonstream_text(self, data: Dict[str, Any]) -> str:
        return self._extract_candidates_text(data)

    def _extract_stream_delta(self, event: Dict[str, Any]) -> str:
        return self._extract_candidates_text(event)

    @staticmethod
    def _extract_candidates_text(data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))

    def _parse_models(self, data: Dict[str, Any]) -> List[ProviderModel]:
        items = data.get("models", []) if isinstance(data, dict) else []
        out: List[ProviderModel] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            raw_name = entry.get("name") or ""
            mid = _strip_models_prefix(raw_name)
            if not mid:
                continue
            out.append(ProviderModel(
                id=mid,
                kind=ModelKind.CHAT,
                label=entry.get("displayName") or mid,
                context_window=entry.get("inputTokenLimit"),
            ))
        return out


register_provider(GEMINI_CAPABILITIES, GeminiProvider)
