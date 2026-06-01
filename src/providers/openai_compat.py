# src/providers/openai_compat.py

"""
OpenAI-compatible chat providers: OpenAI, Groq, OpenRouter.

These three services all speak the same wire format (``/chat/completions``,
``/models``, SSE ``data:`` deltas), so they share one implementation and only
differ in base URL, label, and curated catalog. Auth is ``Authorization:
Bearer <key>`` everywhere.

OpenRouter additionally accepts (but does not require) ``HTTP-Referer`` and
``X-Title`` headers for attribution — we send a polite ``X-Title`` so usage
reports identify the caller.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.providers.base import (
    ChatModelConfig,
    ModelKind,
    ProviderCapabilities,
    ProviderLocation,
    ProviderModel,
)
from src.providers.online_base import HTTPChatProvider
from src.providers.registry import register_provider


# --------------------------------------------------------------------------- #
# Shared OpenAI-compatible implementation
# --------------------------------------------------------------------------- #


class OpenAICompatibleProvider(HTTPChatProvider):
    """Speaks the OpenAI chat-completions wire format."""

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._require_api_key()}",
            "Content-Type": "application/json",
        }

    def _chat_url(self, stream: bool = False) -> str:
        return f"{self.base_url}/chat/completions"

    def _models_url(self) -> str:
        return f"{self.base_url}/models"

    def _build_chat_payload(self, prompt, context, max_tokens, temperature, stream):
        return {
            "model": self.config.model,
            "messages": self._build_messages(prompt, context),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

    def _extract_nonstream_text(self, data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def _extract_stream_delta(self, event: Dict[str, Any]) -> str:
        try:
            return event["choices"][0]["delta"].get("content", "") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def _parse_models(self, data: Dict[str, Any]) -> List[ProviderModel]:
        items = data.get("data", []) if isinstance(data, dict) else []
        out: List[ProviderModel] = []
        for entry in items:
            mid = entry.get("id") if isinstance(entry, dict) else None
            if not mid:
                continue
            out.append(ProviderModel(id=mid, kind=ModelKind.CHAT, label=mid))
        return out


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


OPENAI_CAPABILITIES = ProviderCapabilities(
    name="openai", label="OpenAI", location=ProviderLocation.ONLINE,
    requires_api_key=True, supports_install=False, base_url_configurable=True,
    default_base_url="https://api.openai.com/v1",
    docs_url="https://platform.openai.com/docs",
    notes="Standard OpenAI chat-completions API. Bring your own key.",
)

_OPENAI_STATIC: List[ProviderModel] = [
    ProviderModel(id="gpt-4o-mini", label="GPT-4o mini", context_window=128_000,
                  supports_tools=True, multilingual=True,
                  description="Fast, low-cost general model."),
    ProviderModel(id="gpt-4o", label="GPT-4o", context_window=128_000,
                  supports_tools=True, multilingual=True,
                  description="Flagship multimodal general model."),
    ProviderModel(id="gpt-4.1-mini", label="GPT-4.1 mini", context_window=1_000_000,
                  supports_tools=True, multilingual=True,
                  description="Long-context successor; cost-efficient."),
    ProviderModel(id="o3-mini", label="o3-mini", context_window=200_000,
                  supports_tools=True, multilingual=True,
                  description="Reasoning-tuned model."),
    ProviderModel(id="gpt-3.5-turbo", label="GPT-3.5 Turbo", context_window=16_385,
                  supports_tools=True, multilingual=True,
                  description="Legacy compact chat model."),
]


class OpenAIProvider(OpenAICompatibleProvider):
    capabilities = OPENAI_CAPABILITIES
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    STATIC_MODELS = _OPENAI_STATIC


register_provider(OPENAI_CAPABILITIES, OpenAIProvider)


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #


GROQ_CAPABILITIES = ProviderCapabilities(
    name="groq", label="Groq", location=ProviderLocation.ONLINE,
    requires_api_key=True, supports_install=False, base_url_configurable=True,
    default_base_url="https://api.groq.com/openai/v1",
    docs_url="https://console.groq.com/docs",
    notes="Ultra-low-latency inference for popular open-weights models.",
)

_GROQ_STATIC: List[ProviderModel] = [
    ProviderModel(id="llama-3.3-70b-versatile", label="Llama 3.3 70B (versatile)",
                  context_window=131_072, multilingual=True,
                  description="High-quality general model on Groq."),
    ProviderModel(id="llama-3.1-8b-instant", label="Llama 3.1 8B (instant)",
                  context_window=131_072, multilingual=True,
                  description="Very fast small model."),
    ProviderModel(id="mixtral-8x7b-32768", label="Mixtral 8x7B",
                  context_window=32_768, multilingual=True,
                  description="MoE model; broad capability."),
    ProviderModel(id="gemma2-9b-it", label="Gemma 2 9B Instruct",
                  context_window=8_192, multilingual=True),
]


class GroqProvider(OpenAICompatibleProvider):
    capabilities = GROQ_CAPABILITIES
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    STATIC_MODELS = _GROQ_STATIC


register_provider(GROQ_CAPABILITIES, GroqProvider)


# --------------------------------------------------------------------------- #
# OpenRouter — same wire format, plus polite attribution headers
# --------------------------------------------------------------------------- #


OPENROUTER_CAPABILITIES = ProviderCapabilities(
    name="openrouter", label="OpenRouter", location=ProviderLocation.ONLINE,
    requires_api_key=True, supports_install=False, base_url_configurable=True,
    default_base_url="https://openrouter.ai/api/v1",
    docs_url="https://openrouter.ai/docs",
    notes="Single key, many providers. Optional attribution headers are sent.",
)

_OPENROUTER_STATIC: List[ProviderModel] = [
    ProviderModel(id="openai/gpt-4o-mini", label="OpenAI · GPT-4o mini",
                  context_window=128_000, supports_tools=True, multilingual=True),
    ProviderModel(id="anthropic/claude-3.5-sonnet", label="Anthropic · Claude 3.5 Sonnet",
                  context_window=200_000, supports_tools=True, multilingual=True),
    ProviderModel(id="meta-llama/llama-3.1-8b-instruct",
                  label="Meta · Llama 3.1 8B Instruct",
                  context_window=131_072, multilingual=True),
    ProviderModel(id="google/gemini-flash-1.5", label="Google · Gemini 1.5 Flash",
                  context_window=1_000_000, supports_tools=True, multilingual=True),
    ProviderModel(id="mistralai/mistral-7b-instruct",
                  label="Mistral · 7B Instruct", context_window=32_768, multilingual=True),
]


class OpenRouterProvider(OpenAICompatibleProvider):
    capabilities = OPENROUTER_CAPABILITIES
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    STATIC_MODELS = _OPENROUTER_STATIC

    def _auth_headers(self) -> Dict[str, str]:
        headers = super()._auth_headers()
        headers["X-Title"] = "VectorFlow-RAG"
        return headers


register_provider(OPENROUTER_CAPABILITIES, OpenRouterProvider)
