# src/providers/base.py

"""
Model-provider abstraction layer for VectorFlow-RAG (Phase 12).

A *provider* is a source of models — local/offline (Ollama) or remote/online
(OpenAI, Anthropic, Gemini, Groq, OpenRouter). The rest of the system talks
to a provider through one narrow interface, so the pipeline, the API, and the
frontend never embed provider-specific HTTP details.

Design rules (these are load-bearing — keep them):

* **Frontend never learns provider internals.** It only ever sees the JSON
  shapes of :class:`ProviderCapabilities` (what a provider can do) and
  :class:`ProviderModel` (one model's metadata). No URLs, no SDK names.
* **API keys are never held on a config object.** They are resolved at
  construction time from the backend-only ``SecretStore`` and kept in a
  private attribute. Redaction helpers refuse to print them.
* **Everything degrades gracefully.** Connection / list / generate failures
  raise typed :class:`ProviderError` subclasses; callers convert them to
  structured API errors or SSE ``error`` events — they never crash a worker.
* **Chat parity.** ``generate`` / ``stream_generate`` keep the exact
  signatures the legacy ``OllamaClient`` exposes, so ``pipeline.llm`` can be a
  provider with zero changes at the call sites (``RAGPipeline.ask`` / the
  ``/ask`` SSE route).
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ProviderError(Exception):
    """Base class for all provider-layer failures.

    ``retriable`` lets callers decide whether a transient retry is sensible
    (timeouts, 5xx) versus a permanent misconfiguration (bad key, unknown
    model). ``provider`` carries the canonical provider name for logs/metrics.
    """

    def __init__(self, message: str, *, provider: Optional[str] = None, retriable: bool = False):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retriable = retriable


class ProviderUnavailableError(ProviderError):
    """Provider endpoint is unreachable or the local runtime isn't running."""

    def __init__(self, message: str, *, provider: Optional[str] = None):
        super().__init__(message, provider=provider, retriable=True)


class ProviderAuthError(ProviderError):
    """Authentication failed — missing/invalid API key. Not retriable."""

    def __init__(self, message: str, *, provider: Optional[str] = None):
        super().__init__(message, provider=provider, retriable=False)


class ModelNotFoundError(ProviderError):
    """The requested model id is not available from this provider."""

    def __init__(self, message: str, *, provider: Optional[str] = None):
        super().__init__(message, provider=provider, retriable=False)


class UnknownProviderError(ProviderError):
    """The named provider is not registered. Maps to 404 at the API layer."""

    def __init__(self, message: str, *, provider: Optional[str] = None):
        super().__init__(message, provider=provider, retriable=False)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class ModelKind(str, Enum):
    """What a model is for. ``str`` mixin → serializes as a plain string."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


class ProviderLocation(str, Enum):
    OFFLINE = "offline"  # runs locally (Ollama) — no API key, full privacy
    ONLINE = "online"    # remote API — requires a key, leaves the machine


# --------------------------------------------------------------------------- #
# Metadata value objects (frozen → immutable, hashable, safe to share)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderModel:
    """Metadata describing a single model offered by a provider.

    This is the *only* model shape the frontend sees. Offline-only fields
    (size/quantization/RAM) and online-only fields (pricing) are both optional
    so one schema serves every provider.
    """

    id: str
    kind: ModelKind = ModelKind.CHAT
    label: Optional[str] = None
    context_window: Optional[int] = None
    supports_streaming: bool = True
    supports_tools: bool = False
    multilingual: bool = False

    # Offline (Ollama) metadata
    installed: Optional[bool] = None
    size_bytes: Optional[int] = None
    parameter_size: Optional[str] = None     # e.g. "1.1B", "7B"
    quantization: Optional[str] = None        # e.g. "Q4_0"
    ram_estimate_bytes: Optional[int] = None  # rough working-set estimate

    # Online metadata (optional, advisory only — never used for routing)
    pricing: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider, as a whole, can do. Drives which UI controls appear."""

    name: str            # canonical id: "ollama", "openai", ...
    label: str           # human label: "Ollama (local)", "OpenAI", ...
    location: ProviderLocation
    requires_api_key: bool
    supports_chat: bool = True
    supports_streaming: bool = True
    supports_model_listing: bool = True
    supports_install: bool = False       # only Ollama can pull/delete models
    supports_embeddings: bool = False
    base_url_configurable: bool = False
    default_base_url: Optional[str] = None
    docs_url: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["location"] = self.location.value
        return d


@dataclass(frozen=True)
class ConnectionStatus:
    """Result of :meth:`ModelProvider.validate_connection`."""

    ok: bool
    provider: str
    message: str = ""
    models_available: Optional[int] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Config objects (mutable — runtime config mutates these; api_key never here)
# --------------------------------------------------------------------------- #


@dataclass
class ChatModelConfig:
    """Selection of a chat model + decoding params.

    Note the deliberate absence of any ``api_key`` field: keys live only in
    the backend ``SecretStore`` and are injected into the provider instance.
    """

    provider: str = "ollama"
    model: str = "tinyllama"
    base_url: Optional[str] = None        # None → provider default
    max_tokens: int = 512
    temperature: float = 0.7
    request_timeout_s: int = 120

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatModelConfig":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class EmbeddingModelConfig:
    """Selection of an embedding model.

    Defaults to the local sentence-transformers path so the validated English
    baseline is untouched. Online embedding providers are forward-compatible
    here but not the default.
    """

    provider: str = "sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None
    normalize: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingModelConfig":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class RerankerModelConfig:
    """Selection of a reranker model. Disabled by default (English baseline)."""

    provider: str = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enabled: bool = False
    top_n: int = 3
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RerankerModelConfig":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


# --------------------------------------------------------------------------- #
# Provider base class
# --------------------------------------------------------------------------- #


class ModelProvider(abc.ABC):
    """Abstract base for chat-capable model providers.

    Concrete providers implement connection validation, model listing, and the
    two generation methods. Offline providers additionally implement install /
    delete (guarded by ``capabilities.supports_install``).
    """

    capabilities: ProviderCapabilities  # set as a class attribute by subclasses

    def __init__(self, config: ChatModelConfig, *, api_key: Optional[str] = None):
        self.config = config
        # Private + name-mangled so it never lands in __dict__ dumps that get
        # logged or serialized. Redaction helpers are the only readers.
        self.__api_key = api_key

    # -- identity ------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def model(self) -> str:
        """Back-compat shim: ``LLMClientProtocol`` expects a ``.model`` str."""
        return self.config.model

    @property
    def _api_key(self) -> Optional[str]:
        """Subclass-internal accessor for the injected secret."""
        return self.__api_key

    def _require_api_key(self) -> str:
        if not self.__api_key:
            raise ProviderAuthError(
                f"{self.capabilities.label} requires an API key but none is configured.",
                provider=self.name,
            )
        return self.__api_key

    # -- abstract surface ---------------------------------------------------- #

    @abc.abstractmethod
    def validate_connection(self) -> ConnectionStatus:
        """Cheap reachability/auth probe. Never raises — returns a status."""

    @abc.abstractmethod
    def list_models(self) -> List[ProviderModel]:
        """Return models this provider can serve right now."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str:
        """Generate a complete answer string (signature matches OllamaClient)."""

    @abc.abstractmethod
    def stream_generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Yield answer tokens as they arrive (signature matches OllamaClient)."""

    # -- shared prompt helpers (parity with the legacy client) --------------- #

    @staticmethod
    def _build_prompt(prompt: str, context: Optional[List[str]]) -> str:
        """Single-string prompt — byte-identical to the legacy OllamaClient."""
        if context:
            return "\n\n".join(context) + f"\n\nQuestion: {prompt}\nAnswer:"
        return prompt

    @staticmethod
    def _build_messages(prompt: str, context: Optional[List[str]]) -> List[Dict[str, str]]:
        """Chat-style messages for online providers that use a messages API.

        We fold retrieved context into a system message and keep the user's
        question clean, which is the conventional RAG shaping for chat models.
        """
        messages: List[Dict[str, str]] = []
        if context:
            joined = "\n\n".join(context)
            messages.append({
                "role": "system",
                "content": (
                    "Answer the question using only the context below. "
                    "If the answer is not in the context, say so.\n\n"
                    f"Context:\n{joined}"
                ),
            })
        messages.append({"role": "user", "content": prompt})
        return messages

    # -- optional offline operations (default: unsupported) ------------------ #

    def list_catalog(self) -> List[ProviderModel]:
        """Downloadable/free models not necessarily installed. Offline only."""
        return []

    def install_model(self, name: str) -> Iterator[Dict[str, Any]]:
        """Yield progress dicts while installing a model. Offline only."""
        raise ProviderError(
            f"{self.capabilities.label} does not support model installation.",
            provider=self.name,
        )

    def delete_model(self, name: str) -> None:
        """Delete an installed model. Offline only."""
        raise ProviderError(
            f"{self.capabilities.label} does not support model deletion.",
            provider=self.name,
        )
