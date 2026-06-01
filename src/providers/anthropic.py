# src/providers/anthropic.py

"""
Anthropic Claude provider.

Wire shape differs from OpenAI in three ways the base needs to know about:

* auth header is ``x-api-key`` (not ``Authorization: Bearer``) plus a required
  ``anthropic-version``
* ``system`` is a top-level field, not a message with role=system
* SSE events are typed (``content_block_delta`` carries text deltas)

Everything else (model listing, error mapping, generate parity) reuses the
:class:`HTTPChatProvider` plumbing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.providers.base import (
    ModelKind,
    ProviderCapabilities,
    ProviderLocation,
    ProviderModel,
)
from src.providers.online_base import HTTPChatProvider
from src.providers.registry import register_provider


ANTHROPIC_API_VERSION = "2023-06-01"

ANTHROPIC_CAPABILITIES = ProviderCapabilities(
    name="anthropic", label="Anthropic", location=ProviderLocation.ONLINE,
    requires_api_key=True, supports_install=False, base_url_configurable=True,
    default_base_url="https://api.anthropic.com/v1",
    docs_url="https://docs.anthropic.com",
    notes="Anthropic Messages API. system prompt is sent as a top-level field.",
)

_STATIC: List[ProviderModel] = [
    ProviderModel(id="claude-3-5-sonnet-latest", label="Claude 3.5 Sonnet",
                  context_window=200_000, supports_tools=True, multilingual=True,
                  description="Strong general-purpose model."),
    ProviderModel(id="claude-3-5-haiku-latest", label="Claude 3.5 Haiku",
                  context_window=200_000, supports_tools=True, multilingual=True,
                  description="Fast, cost-efficient model."),
    ProviderModel(id="claude-3-opus-latest", label="Claude 3 Opus",
                  context_window=200_000, supports_tools=True, multilingual=True,
                  description="Highest-capability Opus generation."),
    ProviderModel(id="claude-3-haiku-20240307", label="Claude 3 Haiku",
                  context_window=200_000, supports_tools=True, multilingual=True),
]


class AnthropicProvider(HTTPChatProvider):
    capabilities = ANTHROPIC_CAPABILITIES
    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    STATIC_MODELS = _STATIC

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._require_api_key(),
            "anthropic-version": ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
        }

    def _chat_url(self, stream: bool = False) -> str:
        return f"{self.base_url}/messages"

    def _models_url(self) -> str:
        return f"{self.base_url}/models"

    def _build_chat_payload(self, prompt, context, max_tokens, temperature, stream):
        # Anthropic wants system as a TOP-LEVEL field; messages contains only
        # user/assistant turns. We fold any retrieved context into system.
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "messages": [{"role": "user", "content": prompt}],
        }
        if context:
            payload["system"] = (
                "Answer the question using only the context below. "
                "If the answer is not in the context, say so.\n\n"
                "Context:\n" + "\n\n".join(context)
            )
        return payload

    def _extract_nonstream_text(self, data: Dict[str, Any]) -> str:
        blocks = data.get("content", []) if isinstance(data, dict) else []
        out: List[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", "") or "")
        return "".join(out)

    def _extract_stream_delta(self, event: Dict[str, Any]) -> str:
        if event.get("type") != "content_block_delta":
            return ""
        delta = event.get("delta") or {}
        if delta.get("type") != "text_delta":
            return ""
        return delta.get("text", "") or ""

    def _parse_models(self, data: Dict[str, Any]) -> List[ProviderModel]:
        items = data.get("data", []) if isinstance(data, dict) else []
        out: List[ProviderModel] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("id")
            if not mid:
                continue
            out.append(ProviderModel(
                id=mid, kind=ModelKind.CHAT,
                label=entry.get("display_name") or mid,
            ))
        return out


register_provider(ANTHROPIC_CAPABILITIES, AnthropicProvider)
